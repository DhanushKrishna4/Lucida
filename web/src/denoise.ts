/** GPU à-trous denoising. Mirrors `crates/gpu/src/denoise.rs`.
 *
 * Runs over the finished accumulation buffer into a separate output shaped like
 * `Accum`, so the display pass reads it with no change at all — switching the
 * denoiser on swaps which buffer is bound and nothing else.
 *
 * The accumulation is never written. Denoising is a display-time product: one
 * more sample still converges to the unbiased answer.
 */
import { ACCUM_BYTES_PER_PIXEL } from './generated/layout';

export interface DenoiseSettings {
  /** À-trous passes. Each doubles the tap spacing. Zero disables the filter. */
  iterations: number;
  sigmaNormal: number;
  sigmaDepth: number;
  sigmaLuminance: number;
  sigmaAlbedo: number;
}

/** Matches `DenoiseParams::default()` in crates/core/src/denoise.rs. */
export const DEFAULT_DENOISE: DenoiseSettings = {
  iterations: 4,
  sigmaNormal: 64,
  sigmaDepth: 0.08,
  sigmaLuminance: 4,
  sigmaAlbedo: 0.1,
};

const PARAMS_BYTES = 32;

export class Denoiser {
  private layout: GPUBindGroupLayout;
  private demodulate: GPUComputePipeline;
  private measure: GPUComputePipeline;
  private atrous: GPUComputePipeline;
  private modulate: GPUComputePipeline;

  private a?: GPUBuffer;
  private b?: GPUBuffer;
  private deviation?: GPUBuffer;
  private params: GPUBuffer[] = [];
  private pixels = 0;

  /** Accum-shaped; what the display pass binds when denoising is on. */
  output?: GPUBuffer;

  private constructor(
    private device: GPUDevice,
    layout: GPUBindGroupLayout,
    pipelines: GPUComputePipeline[],
  ) {
    this.layout = layout;
    [this.demodulate, this.measure, this.atrous, this.modulate] = pipelines;
  }

  static async create(device: GPUDevice, module: GPUShaderModule): Promise<Denoiser> {
    const sto = (binding: number, readOnly: boolean): GPUBindGroupLayoutEntry => ({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: readOnly ? 'read-only-storage' : 'storage' },
    });
    const layout = device.createBindGroupLayout({
      label: 'denoise',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        sto(1, true),
        sto(2, false),
        sto(3, false),
        sto(4, false),
        sto(5, false),
      ],
    });
    const pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const stage = (entryPoint: string) =>
      device.createComputePipelineAsync({ label: entryPoint, layout: pl, compute: { module, entryPoint } });
    const pipelines = await Promise.all([
      stage('demodulate'),
      stage('measure'),
      stage('atrous_pass'),
      stage('modulate'),
    ]);
    return new Denoiser(device, layout, pipelines);
  }

  /** Reallocate for a new render size. */
  resize(pixels: number): void {
    if (pixels === this.pixels) return;
    for (const b of [this.a, this.b, this.deviation, this.output]) b?.destroy();
    const mk = (label: string, bytes: number) =>
      this.device.createBuffer({ label, size: bytes, usage: GPUBufferUsage.STORAGE });
    this.a = mk('denoise a', pixels * 16);
    this.b = mk('denoise b', pixels * 16);
    this.deviation = mk('denoise deviation', pixels * 4);
    this.output = mk('denoise output', pixels * ACCUM_BYTES_PER_PIXEL);
    this.pixels = pixels;
  }

  private paramsFor(index: number, width: number, height: number, stride: number, s: DenoiseSettings): GPUBuffer {
    while (this.params.length <= index) {
      this.params.push(
        this.device.createBuffer({
          label: 'denoise params',
          size: PARAMS_BYTES,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        }),
      );
    }
    const buf = new ArrayBuffer(PARAMS_BYTES);
    new Uint32Array(buf, 0, 3).set([width, height, stride]);
    new Float32Array(buf, 12, 4).set([s.sigmaNormal, s.sigmaDepth, s.sigmaLuminance, s.sigmaAlbedo]);
    this.device.queue.writeBuffer(this.params[index], 0, buf);
    return this.params[index];
  }

  /** Record the filter chain. Returns the buffer the display should read. */
  run(
    encoder: GPUCommandEncoder,
    accum: GPUBuffer,
    width: number,
    height: number,
    settings: DenoiseSettings,
  ): GPUBuffer {
    this.resize(width * height);
    const gx = Math.ceil(width / 8);
    const gy = Math.ceil(height / 8);

    const group = (u: GPUBuffer, a: GPUBuffer, b: GPUBuffer) =>
      this.device.createBindGroup({
        layout: this.layout,
        entries: [
          { binding: 0, resource: { buffer: u } },
          { binding: 1, resource: { buffer: accum } },
          { binding: 2, resource: { buffer: a } },
          { binding: 3, resource: { buffer: b } },
          { binding: 4, resource: { buffer: this.deviation! } },
          { binding: 5, resource: { buffer: this.output! } },
        ],
      });
    const pass = encoder.beginComputePass({ label: 'denoise' });
    const dispatch = (p: GPUComputePipeline, g: GPUBindGroup) => {
      pass.setPipeline(p);
      pass.setBindGroup(0, g);
      pass.dispatchWorkgroups(gx, gy, 1);
    };

    const g0 = group(this.paramsFor(0, width, height, 1, settings), this.a!, this.b!);
    dispatch(this.demodulate, g0);
    dispatch(this.measure, g0);

    // Ping-pong by swapping which buffer is bound where, so the shader always
    // reads `buf_a` and writes `buf_b` and needs no notion of which pass it is
    // on.
    let inA = true;
    for (let level = 0; level < settings.iterations; level++) {
      const u = this.paramsFor(level + 1, width, height, 1 << level, settings);
      const g = inA ? group(u, this.a!, this.b!) : group(u, this.b!, this.a!);
      dispatch(this.atrous, g);
      inA = !inA;
    }

    // `modulate` reads `buf_a`, so bind whichever buffer the last pass wrote.
    const uf = this.paramsFor(settings.iterations + 1, width, height, 1, settings);
    dispatch(this.modulate, inA ? group(uf, this.a!, this.b!) : group(uf, this.b!, this.a!));
    pass.end();
    return this.output!;
  }
}
