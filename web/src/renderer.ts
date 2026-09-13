/**
 * The browser renderer.
 *
 * Architecture, deliberately simple at this stage:
 *
 *   compute pass (megakernel)  ->  accum: array<vec4<f32>>  ->  render pass (display)
 *
 * Radiance is summed into an f32 storage buffer (xyz = sum, w = sample count)
 * and divided only at display time. Two consequences worth stating:
 *
 *   * The accumulation arithmetic is identical to the CPU reference tracer's —
 *     sum in sample order, divide once — so the browser can be compared against
 *     the native renderers at the same sharpness the native comparison enjoys.
 *   * Exposure and tone mapping are free. They re-run the display pass, not the
 *     integrator, which is what makes the raw-vs-denoised and operator toggles
 *     instant later on.
 *
 * The accumulator is a storage *buffer* rather than a storage texture because
 * read-modify-write storage textures are not core WebGPU; `rgba32float` is
 * write-only there.
 */
import { ACCUM_BYTES_PER_PIXEL, UNIFORMS_SIZE, writeUniforms } from './generated/layout';

import { resolveCamera } from './camera';
import type { CameraDef } from './generated/scenes';
import type { SceneData } from './sceneLoader';
import { captureErrors, createShaderModule } from './webgpu';
import { ConvergenceMeter } from './stats';
import { Denoiser, DEFAULT_DENOISE } from './denoise';
import denoiseSource from '../../shaders/denoise/atrous.wgsl';
import convergenceSource from '../../shaders/stats/convergence.wgsl';
import { WavefrontPass } from './wavefront';

/** Must match `SamplingMode::index()` in crates/core/src/integrator.rs. */
const SAMPLING_INDEX: Record<SamplingMode, number> = { bsdf: 0, nee: 1, mis: 2 };
/** Must match `SamplerKind::index()` in crates/core/src/sobol.rs. */
const SAMPLER_INDEX: Record<SamplerKind, number> = { independent: 0, sobol: 1 };
/** Must match `RenderMode::index()` in crates/core/src/diagnostic.rs. */
const DIAGNOSTIC_INDEX: Record<DiagnosticMode, number> = {
  beauty: 0,
  normal: 1,
  albedo: 2,
  depth: 3,
  heat: 4,
};

/** Must match `Tonemap::index()` in crates/core/src/tonemap.rs. */
const TONEMAP_INDEX: Record<Tonemap, number> = {
  clamp: 0,
  reinhard: 1,
  aces: 2,
  agx: 3,
};

import megakernelSource from '../../shaders/trace/megakernel.wgsl';
import gradientSource from '../../shaders/gradient.wgsl';
import displaySource from '../../shaders/display.wgsl';

export type RenderMode = 'gradient' | 'pathtrace';
export type Tonemap = 'clamp' | 'reinhard' | 'aces' | 'agx';

export interface RenderSettings {
  mode: RenderMode;
  width: number;
  height: number;
  /** Samples per pixel taken by a single dispatch. */
  samplesPerFrame: number;
  maxDepth: number;
  frameSeed: number;
  exposure: number;
  tonemap: Tonemap;
  /** Stop accumulating once this many samples have landed. 0 = never stop. */
  targetSamples: number;
  architecture: Architecture;
  /**
   * Frame-time budget, in milliseconds, that the interactive resolution aims
   * for while the camera is moving.
   */
  targetFrameMs: number;
  /** Allow dropping internal resolution during interaction. */
  adaptiveResolution: boolean;
  /**
   * How light is found. Both estimate the same integral and converge to the
   * same image; they differ enormously in variance depending on the scene.
   */
  sampling: SamplingMode;
  /** Which point set to draw from. See crates/core/src/sobol.rs. */
  sampler: SamplerKind;
  /** À-trous denoise passes applied at display time. Zero is off. */
  denoisePasses: number;
  /** Diagnostic display mode. See crates/core/src/diagnostic.rs. */
  diagnostic: DiagnosticMode;
}

/** Must match `SamplingMode::index()` in crates/core/src/integrator.rs. */
export type SamplingMode = 'bsdf' | 'nee' | 'mis';
export type SamplerKind = 'independent' | 'sobol';
export type DiagnosticMode = 'beauty' | 'normal' | 'albedo' | 'depth' | 'heat';

/**
 * Which GPU path-tracing architecture to run.
 *
 * Both are kept because neither dominates: the megakernel wins on geometrically
 * trivial scenes, the wavefront on heavy ones and at high bounce limits. The
 * numbers are in `crates/gpu/src/wavefront.rs`.
 */
export type Architecture = 'megakernel' | 'wavefront';

export const DEFAULT_SETTINGS: RenderSettings = {
  mode: 'pathtrace',
  width: 512,
  height: 512,
  samplesPerFrame: 2,
  maxDepth: 8,
  frameSeed: 0x5eed1234,
  exposure: 1.0,
  tonemap: 'clamp',
  targetSamples: 0,
  targetFrameMs: 16,
  adaptiveResolution: true,
  sampling: 'mis',
  sampler: 'sobol',
  denoisePasses: 0,
  diagnostic: 'beauty',
  // The megakernel is the default because the browser runs few samples per
  // frame at interactive depths, which is exactly where it wins.
  architecture: 'megakernel',
};

/**
 * Resolution divisors the adaptive controller can select, coarsest last.
 *
 * Powers of two only. A divisor that does not divide the width evenly still
 * works (the dispatch rounds up and the shader bounds-checks), but keeping them
 * powers of two means the upscale in the display pass lands on clean pixel
 * boundaries instead of shimmering.
 */
const SCALES = [1, 2, 4, 8] as const;

/** Frames between convergence measurements. See `Renderer.framesSinceMeasure`. */
const MEASURE_EVERY = 8;

export interface Stats {
  samples: number;
  /** Resolution divisor in effect for the last frame. */
  scale: number;
  renderWidth: number;
  renderHeight: number;
  lastFrameMs: number;
  /** Exponential moving average, so the readout does not flicker. */
  avgFrameMs: number;
  /**
   * Upper bound on rays traced per second.
   *
   * It assumes every path runs the full `maxDepth` bounces, which none do —
   * paths that escape the box or hit the (black) emitter terminate early, and
   * the fraction that do varies with the scene and the bounce limit. A true
   * count needs a per-ray atomic counter written back to the host; the traversal
   * counter added for the BVH heatmap counts *node visits*, which is a different
   * quantity. Reported with a `<=` rather than quietly rounded, because an
   * unlabelled upper bound on a throughput figure is a way of claiming a number
   * the renderer did not achieve.
   */
  maxRaysPerSecond: number;
  gpuBytes: number;
  /**
   * Measured mean relative standard error over the image, or a negative number
   * when nothing has been measured yet. See `web/src/stats.ts`.
   */
  convergence: number;
  /**
   * Milliseconds spent actually accumulating since the last reset.
   *
   * Wall-clock time since the reset would include every frame the renderer sat
   * idle on a converged image, which would make the remaining-time estimate
   * grow without bound while nothing was happening.
   */
  accumMs: number;
}



export class Renderer {
  readonly device: GPUDevice;
  private readonly canvas: HTMLCanvasElement;
  private readonly context: GPUCanvasContext;
  private readonly format: GPUTextureFormat;

  private computeLayout!: GPUBindGroupLayout;
  private displayLayout!: GPUBindGroupLayout;
  private tracePipeline!: GPUComputePipeline;
  private gradientPipeline!: GPUComputePipeline;
  private displayPipeline!: GPURenderPipeline;

  private uniformBuffer!: GPUBuffer;
  private displayUniformBuffer!: GPUBuffer;
  private accumBuffer!: GPUBuffer;
  private materialBuffer!: GPUBuffer;
  private primitiveBuffer!: GPUBuffer;
  private lightBuffer!: GPUBuffer;
  private positionBuffer!: GPUBuffer;
  private vertexAttrBuffer!: GPUBuffer;
  private triangleBuffer!: GPUBuffer;
  private bvhNodeBuffer!: GPUBuffer;
  private envRadianceTexture?: GPUTexture;
  private envCdfTexture?: GPUTexture;
  private envRadianceView!: GPUTextureView;
  private envCdfView!: GPUTextureView;
  private computeBindGroup!: GPUBindGroup;
  private displayBindGroup!: GPUBindGroup;
  private denoiser!: Denoiser;
  private meter!: ConvergenceMeter;

  private settings: RenderSettings = { ...DEFAULT_SETTINGS };
  private scene!: SceneData;
  private camera!: CameraDef;

  private accumulatedSamples = 0;
  private allocatedPixels = 0;
  private stats: Stats = {
    samples: 0,
    scale: 1,
    renderWidth: 0,
    renderHeight: 0,
    lastFrameMs: 0,
    avgFrameMs: 0,
    maxRaysPerSecond: 0,
    gpuBytes: 0,
    convergence: -1,
    accumMs: 0,
  };
  private frameStart = 0;
  private raysInFlight = 0;
  /**
   * Frames since the last convergence measurement.
   *
   * The reduction reads the whole accumulation buffer — 16 MB at 512x512 — so
   * running it every frame would spend real bandwidth on a number that moves far
   * too slowly to need 60 updates a second. Every eighth frame is several
   * readings per second, which is faster than anyone reads it.
   */
  private framesSinceMeasure = 0;

  /** Current resolution divisor; 1 means full resolution. */
  private scale = 1;
  private interacting = false;
  /**
   * Consecutive frames that have argued for the same scale change. The
   * controller only acts after a couple of them, because a single slow frame
   * (a shader recompile, another tab waking up) is not evidence about steady
   * -state cost, and reacting to it makes the resolution visibly flap.
   */
  private scaleVotes = 0;
  private scaleVoteDirection = 0;
  /** When false, the shader tests every triangle instead of traversing. */
  private bvhEnabled = true;
  private wavefront?: WavefrontPass;

  private constructor(device: GPUDevice, canvas: HTMLCanvasElement) {
    this.device = device;
    this.canvas = canvas;
    const ctx = canvas.getContext('webgpu');
    if (!ctx) throw new Error('canvas.getContext("webgpu") returned null');
    this.context = ctx;
    // The display shader writes sRGB-encoded values itself, so the canvas must
    // be a *non*-sRGB format. Configuring a `-srgb` format here would apply the
    // transfer function a second time and wash the image out.
    this.format = navigator.gpu.getPreferredCanvasFormat();
    ctx.configure({ device, format: this.format, alphaMode: 'opaque' });
  }

  static async create(
    device: GPUDevice,
    canvas: HTMLCanvasElement,
    scene: SceneData,
    settings: Partial<RenderSettings> = {},
  ): Promise<Renderer> {
    const r = new Renderer(device, canvas);
    r.settings = { ...DEFAULT_SETTINGS, ...settings };
    await r.buildPipelines();
    // Built eagerly: pipeline creation is asynchronous, and switching
    // architecture must not stall a frame partway through a render loop.
    r.wavefront = await WavefrontPass.create(device);
    r.setScene(scene);
    return r;
  }

  private async buildPipelines(): Promise<void> {
    const device = this.device;

    // An explicit bind group layout rather than `layout: 'auto'`. With 'auto'
    // each pipeline gets a layout derived from the bindings it happens to use,
    // so the gradient kernel (which touches only 0 and 4) and the path tracer
    // would need separate, incompatible bind groups for the same buffers.
    // Declaring the layout once lets both pipelines share one bind group.
    this.computeLayout = device.createBindGroupLayout({
      label: 'trace bindings',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        // The environment map arrives as textures rather than storage buffers.
        // That is forced: the wavefront's SHADE stage already binds exactly
        // eight storage buffers, WebGPU's guaranteed per-stage limit counted
        // across every bind group. `float` and not `unfilterable-float` because
        // the shader only ever calls textureLoad, which needs no filtering —
        // and must not have any, since the CDF describes a piecewise-constant
        // image and a filtered radiance would not be the quantity the density
        // is proportional to.
        {
          binding: 90,
          visibility: GPUShaderStage.COMPUTE,
          texture: { sampleType: 'unfilterable-float', viewDimension: '2d' },
        },
        {
          binding: 91,
          visibility: GPUShaderStage.COMPUTE,
          texture: { sampleType: 'unfilterable-float', viewDimension: '2d' },
        },
      ],
    });
    this.displayLayout = device.createBindGroupLayout({
      label: 'display bindings',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      ],
    });

    const [traceModule, gradientModule, displayModule, denoiseModule, statsModule] =
      await Promise.all([
        createShaderModule(device, 'megakernel.wgsl', megakernelSource),
        createShaderModule(device, 'gradient.wgsl', gradientSource),
        createShaderModule(device, 'display.wgsl', displaySource),
        createShaderModule(device, 'atrous.wgsl', denoiseSource),
        createShaderModule(device, 'convergence.wgsl', convergenceSource),
      ]);
    this.denoiser = await Denoiser.create(device, denoiseModule);
    this.meter = await ConvergenceMeter.create(device, statsModule);

    const computePipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.computeLayout],
    });

    await captureErrors(device, 'creating pipelines', () => {
      this.tracePipeline = device.createComputePipeline({
        label: 'megakernel',
        layout: computePipelineLayout,
        compute: { module: traceModule, entryPoint: 'main' },
      });
      this.gradientPipeline = device.createComputePipeline({
        label: 'gradient',
        layout: computePipelineLayout,
        compute: { module: gradientModule, entryPoint: 'main' },
      });
      this.displayPipeline = device.createRenderPipeline({
        label: 'display',
        layout: device.createPipelineLayout({ bindGroupLayouts: [this.displayLayout] }),
        vertex: { module: displayModule, entryPoint: 'vs' },
        fragment: { module: displayModule, entryPoint: 'fs', targets: [{ format: this.format }] },
        primitive: { topology: 'triangle-list' },
      });
    });

    this.uniformBuffer = device.createBuffer({
      label: 'uniforms',
      size: UNIFORMS_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.displayUniformBuffer = device.createBuffer({
      label: 'display uniforms',
      // 32 bytes: the diagnostic mode and the scene's depth scale joined the
      // original four words.
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  setScene(scene: SceneData): void {
    this.scene = scene;
    this.camera = { ...scene.camera };
    for (const b of [
      this.materialBuffer,
      this.primitiveBuffer,
      this.lightBuffer,
      this.positionBuffer,
      this.vertexAttrBuffer,
      this.triangleBuffer,
      this.bvhNodeBuffer,
    ]) {
      b?.destroy();
    }
    // Scene arrays arrive pre-packed from Rust, so there is no TypeScript code
    // here that could lay a struct out incorrectly.
    this.materialBuffer = this.uploadStorage('materials', scene.materials, 80);
    this.primitiveBuffer = this.uploadStorage('primitives', scene.primitives, 64);
    this.lightBuffer = this.uploadStorage('lights', scene.lights, 64);
    this.positionBuffer = this.uploadStorage('positions', scene.positions, 16);
    this.vertexAttrBuffer = this.uploadStorage('vertexAttrs', scene.vertexAttrs, 32);
    this.triangleBuffer = this.uploadStorage('triangles', scene.triangles, 16);
    this.bvhNodeBuffer = this.uploadStorage('bvhNodes', scene.bvhNodes, 32);
    this.uploadEnvMap(scene);
    this.allocatedPixels = 0; // force reallocation, which rebuilds the bind group
    this.reset();
  }

  /**
   * WebGPU forbids zero-sized bindings, so an empty primitive array still needs
   * one element's worth of zeroes; `num_spheres == 0` stops the shader's loop.
   */
  /**
   * Upload the environment map's two textures, or 1x1 stand-ins.
   *
   * WebGPU has no optional bindings, so something must be bound even for a
   * scene with no sky; `env_width == 0` in the uniforms keeps the shader from
   * ever reading it. The bytes are packed in Rust — including the CDF
   * rectangle's layout — so there is no TypeScript that could lay the
   * distribution out differently from the native harness.
   */
  private uploadEnvMap(scene: SceneData): void {
    const device = this.device;
    const has = scene.envWidth > 0 && scene.envTotalWeight > 0;
    const make = (label: string, w: number, h: number, format: GPUTextureFormat) =>
      device.createTexture({
        label,
        size: { width: Math.max(w, 1), height: Math.max(h, 1) },
        format,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });

    this.envRadianceTexture?.destroy();
    this.envCdfTexture?.destroy();

    if (!has) {
      this.envRadianceTexture = make('env radiance', 1, 1, 'rgba32float');
      this.envCdfTexture = make('env cdf', 1, 1, 'r32float');
      device.queue.writeTexture(
        { texture: this.envRadianceTexture },
        new Float32Array(4),
        { bytesPerRow: 16 },
        { width: 1, height: 1 },
      );
    } else {
      const w = scene.envWidth;
      const h = scene.envHeight;
      this.envRadianceTexture = make('env radiance', w, h, 'rgba32float');
      device.queue.writeTexture(
        { texture: this.envRadianceTexture },
        scene.envRadiance,
        { bytesPerRow: w * 16, rowsPerImage: h },
        { width: w, height: h },
      );
      // Same rectangle Rust packed: rows 0..h-1 conditional, row h marginal.
      const cdfW = Math.max(w + 1, h + 1);
      const cdfH = h + 1;
      this.envCdfTexture = make('env cdf', cdfW, cdfH, 'r32float');
      device.queue.writeTexture(
        { texture: this.envCdfTexture },
        scene.envCdf,
        { bytesPerRow: cdfW * 4, rowsPerImage: cdfH },
        { width: cdfW, height: cdfH },
      );
    }
    this.envRadianceView = this.envRadianceTexture.createView();
    this.envCdfView = this.envCdfTexture.createView();
  }

  private uploadStorage(
    label: string,
    data: Uint8Array<ArrayBuffer>,
    stride: number,
  ): GPUBuffer {
    const size = Math.max(data.byteLength, stride);
    const buf = this.device.createBuffer({
      label,
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (data.byteLength > 0) this.device.queue.writeBuffer(buf, 0, data);
    return buf;
  }

  getCamera(): CameraDef {
    return { ...this.camera };
  }

  getScene(): SceneData {
    return this.scene;
  }

  setCamera(camera: CameraDef): void {
    this.camera = camera;
    this.reset();
  }

  getSettings(): RenderSettings {
    return { ...this.settings };
  }

  /**
   * Toggle BVH traversal. With it off the shader brute-forces every triangle —
   * the same reference path the correctness tests compare against, and a direct
   * demonstration of what the acceleration structure is worth.
   */
  setBvhEnabled(enabled: boolean): void {
    if (enabled === this.bvhEnabled) return;
    this.bvhEnabled = enabled;
    this.reset();
  }

  isBvhEnabled(): boolean {
    return this.bvhEnabled;
  }

  /**
   * Resolution actually being dispatched, which is the display resolution
   * divided by the current interactive scale.
   */
  get renderWidth(): number {
    return Math.max(1, Math.ceil(this.settings.width / this.scale));
  }
  get renderHeight(): number {
    return Math.max(1, Math.ceil(this.settings.height / this.scale));
  }
  get renderScale(): number {
    return this.scale;
  }

  /**
   * Tell the renderer whether the user is currently moving the camera.
   *
   * Entering interaction lets the adaptive controller drop resolution; leaving
   * it returns to full resolution immediately, so the moment the user lets go
   * the image starts converging at full quality.
   */
  setInteracting(interacting: boolean): void {
    if (interacting === this.interacting) return;
    this.interacting = interacting;
    if (!interacting && this.scale !== 1) {
      this.setScale(1);
    }
    this.scaleVotes = 0;
    this.scaleVoteDirection = 0;
  }

  private setScale(scale: number): void {
    if (scale === this.scale) return;
    this.scale = scale;
    // A different render resolution means a different pixel grid, so nothing
    // accumulated so far is reusable.
    this.reset();
  }

  /**
   * Pick a resolution divisor that keeps the frame inside the time budget.
   *
   * Only runs during interaction. Stepping up (coarser) is allowed to happen
   * sooner than stepping down, because being briefly too soft is much less
   * noticeable than being briefly too slow, and asymmetric thresholds are what
   * stop the controller oscillating between two neighbouring scales.
   */
  private adaptScale(): void {
    if (!this.interacting || !this.settings.adaptiveResolution) return;
    const ms = this.stats.avgFrameMs;
    if (ms <= 0) return;

    const target = this.settings.targetFrameMs;
    const i = SCALES.indexOf(this.scale as (typeof SCALES)[number]);
    if (i < 0) return;

    let want = 0;
    if (ms > target * 1.15 && i < SCALES.length - 1) {
      want = +1;
    } else if (ms < target * 0.5 && i > 0) {
      // Halving the divisor quadruples the pixel count, so only step down when
      // there is at least 2x headroom — otherwise the step down immediately
      // triggers a step back up.
      want = -1;
    }

    if (want === 0 || want !== this.scaleVoteDirection) {
      this.scaleVoteDirection = want;
      this.scaleVotes = want === 0 ? 0 : 1;
      return;
    }
    if (++this.scaleVotes < 3) return;

    this.scaleVotes = 0;
    this.scaleVoteDirection = 0;
    this.setScale(SCALES[i + want]);
  }

  /**
   * Apply settings. Anything that changes what is being *integrated* resets the
   * accumulation; anything that only changes how it is *displayed* does not.
   * Getting that distinction wrong either throws away converged samples on an
   * exposure tweak, or shows a stale image after a depth change.
   */
  setSettings(patch: Partial<RenderSettings>): void {
    const prev = this.settings;
    const next = { ...prev, ...patch };
    const invalidates =
      next.mode !== prev.mode ||
      next.width !== prev.width ||
      next.height !== prev.height ||
      next.maxDepth !== prev.maxDepth ||
      next.frameSeed !== prev.frameSeed ||
      next.sampling !== prev.sampling;
    this.settings = next;
    if (invalidates) this.reset();
  }

  /** Discard accumulated samples and start the estimate again. */
  reset(): void {
    this.accumulatedSamples = 0;
    // Both belong to the estimate being discarded. Leaving the convergence
    // figure standing would show the old image's error against the new image's
    // sample count, which reads as a sudden collapse in quality that never
    // happened.
    this.stats.convergence = -1;
    this.stats.accumMs = 0;
    this.meter.value = -1;
    this.framesSinceMeasure = 0;
    this.ensureAccumBuffer();
    const encoder = this.device.createCommandEncoder({ label: 'reset accum' });
    encoder.clearBuffer(this.accumBuffer);
    this.device.queue.submit([encoder.finish()]);
  }

  private ensureAccumBuffer(): void {
    const { width, height } = this.settings;
    // Always allocated at *full* resolution. A reduced-scale render writes a
    // compact prefix of the same buffer — a 1/4-scale frame needs 1/16 of the
    // pixels, so it always fits — which means changing the interactive scale
    // costs nothing: no reallocation, no second buffer, no bind group rebuild.
    const pixels = width * height;
    if (pixels === this.allocatedPixels) return;

    this.accumBuffer?.destroy();
    this.accumBuffer = this.device.createBuffer({
      label: 'accum',
      size: pixels * ACCUM_BYTES_PER_PIXEL,
      // COPY_DST is required by `clearBuffer`, which is how `reset()` discards
      // accumulated samples. Omitting it does not fail loudly: WebGPU
      // zero-initialises new buffers, so a fresh allocation still renders
      // correctly and only *resets* silently do nothing — leaving stale samples
      // mixed into the estimate after a depth or seed change.
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.allocatedPixels = pixels;

    this.canvas.width = width;
    this.canvas.height = height;

    this.computeBindGroup = this.device.createBindGroup({
      label: 'trace bindings',
      layout: this.computeLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.materialBuffer } },
        { binding: 2, resource: { buffer: this.primitiveBuffer } },
        { binding: 3, resource: { buffer: this.lightBuffer } },
        { binding: 4, resource: { buffer: this.accumBuffer } },
        { binding: 5, resource: { buffer: this.positionBuffer } },
        { binding: 6, resource: { buffer: this.vertexAttrBuffer } },
        { binding: 7, resource: { buffer: this.triangleBuffer } },
        { binding: 8, resource: { buffer: this.bvhNodeBuffer } },
        { binding: 90, resource: this.envRadianceView },
        { binding: 91, resource: this.envCdfView },
      ],
    });
    this.displayBindGroup = this.device.createBindGroup({
      label: 'display bindings',
      layout: this.displayLayout,
      entries: [
        { binding: 0, resource: { buffer: this.displayUniformBuffer } },
        { binding: 1, resource: { buffer: this.accumBuffer } },
      ],
    });

    this.wavefront?.setScene(
      {
        uniforms: this.uniformBuffer,
        materials: this.materialBuffer,
        primitives: this.primitiveBuffer,
        lights: this.lightBuffer,
        positions: this.positionBuffer,
        vertexAttrs: this.vertexAttrBuffer,
        triangles: this.triangleBuffer,
        bvhNodes: this.bvhNodeBuffer,
        envRadiance: this.envRadianceView,
        envCdf: this.envCdfView,
      },
      this.accumBuffer,
    );

    this.stats.gpuBytes =
      pixels * ACCUM_BYTES_PER_PIXEL +
      this.materialBuffer.size +
      this.primitiveBuffer.size +
      this.lightBuffer.size +
      this.positionBuffer.size +
      this.vertexAttrBuffer.size +
      this.triangleBuffer.size +
      this.bvhNodeBuffer.size +
      UNIFORMS_SIZE +
      16;
  }

  get isConverged(): boolean {
    const t = this.settings.targetSamples;
    return t > 0 && this.accumulatedSamples >= t;
  }

  getStats(): Stats {
    return {
      ...this.stats,
      // Added at read time, not in ensureAccumBuffer: the wavefront allocates
      // its pool lazily on its first render, which is after that runs.
      gpuBytes: this.stats.gpuBytes + (this.wavefront?.gpuBytes ?? 0),
      samples: this.accumulatedSamples,
      scale: this.scale,
      renderWidth: this.renderWidth,
      renderHeight: this.renderHeight,
      convergence: this.meter.value,
    };
  }

  /** Write the uniform block for the next dispatch. */
  private writeUniforms(samplesThisDispatch: number, sampleOffset?: number): void {
    const s = this.settings;
    const [rw, rh] = [this.renderWidth, this.renderHeight];
    // Aspect comes from the *display* resolution, not the render resolution:
    // ceil() when dividing can shift the render aspect by a fraction of a
    // percent, and letting that reach the camera would make the framing jitter
    // as the interactive scale changes.
    const cam = resolveCamera(this.camera, s.width / s.height);
    const buf = writeUniforms({
      ...cam,
      width: rw,
      height: rh,
      sampleOffset: sampleOffset ?? this.accumulatedSamples,
      samplesPerLaunch: samplesThisDispatch,
      maxDepth: s.maxDepth,
      frameSeed: s.frameSeed,
      numPrimitives: this.scene.numPrimitives,
      numLights: this.scene.numLights,
      samplingMode: SAMPLING_INDEX[s.sampling],
      samplerKind: SAMPLER_INDEX[s.sampler],
      numTriangles: this.scene.numTriangles,
      // Zero disables traversal and makes the shader test every triangle. That
      // is not a fallback for weak hardware — it is the reference path the BVH
      // is validated against, exposed so it can be toggled from the console.
      numBvhNodes: this.bvhEnabled ? this.scene.numBvhNodes : 0,
      renderMode: 0,
      background: this.scene.background,
      envWidth: this.scene.envWidth,
      envHeight: this.scene.envHeight,
      envTotalWeight: this.scene.envTotalWeight,
      numInstances: this.scene.numInstances,
      tlasRoot: this.scene.tlasRoot,
    });
    this.device.queue.writeBuffer(this.uniformBuffer, 0, buf);

    const d = new ArrayBuffer(32);
    // The display pass reads the render grid and stretches it over the
    // full-resolution canvas, which is what upscales a reduced-scale frame.
    new Uint32Array(d, 0, 2).set([rw, rh]);
    new Float32Array(d, 8, 1)[0] = s.exposure;
    new Uint32Array(d, 12, 1)[0] = TONEMAP_INDEX[s.tonemap];
    new Uint32Array(d, 16, 1)[0] = DIAGNOSTIC_INDEX[s.diagnostic];
    // The scene's own scale, so the depth mode reads the same whether the scene
    // is measured in hundreds (the Cornell box) or in ones (a sphere).
    new Float32Array(d, 20, 1)[0] = this.scene.depthScale;
    this.device.queue.writeBuffer(this.displayUniformBuffer, 0, d);
  }

  /**
   * Advance the render by one dispatch and present.
   *
   * Returns the number of samples added, which is zero when the target sample
   * count has been reached — the display pass still runs, so exposure changes
   * remain live on a converged image.
   */
  renderFrame(): number {
    this.ensureAccumBuffer();
    const s = this.settings;

    let samplesThisDispatch = 0;
    if (s.mode === 'gradient') {
      // The gradient kernel overwrites rather than accumulates, so running it
      // repeatedly is pointless; one dispatch is enough.
      samplesThisDispatch = this.accumulatedSamples === 0 ? 1 : 0;
    } else if (!this.isConverged) {
      samplesThisDispatch = s.samplesPerFrame;
      if (s.targetSamples > 0) {
        samplesThisDispatch = Math.min(
          samplesThisDispatch,
          s.targetSamples - this.accumulatedSamples,
        );
      }
    }

    const t0 = performance.now();

    // The wavefront submits its own command buffers, one per batch, because
    // `queue.writeBuffer` is flushed ahead of the command buffers it is
    // submitted with — so a batch's uniforms have to be paired with its own
    // submit, or every batch renders the last batch's seed and the image never
    // converges. Its work is queued *before* the display encoder below, which
    // is the order the display pass needs anyway.
    const useWavefront =
      s.mode === 'pathtrace' && s.architecture === 'wavefront' && this.wavefront !== undefined;

    if (useWavefront && samplesThisDispatch > 0) {
      this.wavefront!.render({
        width: this.renderWidth,
        height: this.renderHeight,
        samples: samplesThisDispatch,
        maxDepth: s.maxDepth,
        sampleOffset: this.accumulatedSamples,
        writeUniforms: (offset, perLaunch) => this.writeUniforms(perLaunch, offset),
      });
    }
    // Written (or rewritten, after the wavefront's per-batch updates) so the
    // display pass sees the current render grid, exposure and tone map.
    this.writeUniforms(Math.max(samplesThisDispatch, 1));

    const encoder = this.device.createCommandEncoder({ label: 'frame' });

    if (samplesThisDispatch > 0 && !useWavefront) {
      const pass = encoder.beginComputePass({ label: s.mode });
      pass.setPipeline(s.mode === 'gradient' ? this.gradientPipeline : this.tracePipeline);
      pass.setBindGroup(0, this.computeBindGroup);
      // Workgroup size is 8x8; round up and let the shader bounds-check.
      pass.dispatchWorkgroups(Math.ceil(this.renderWidth / 8), Math.ceil(this.renderHeight / 8));
      pass.end();
    }

    // Convergence, measured on the accumulation the passes above have just
    // finished writing. Recorded into the same encoder, so it costs one
    // reduction and no extra submit; the readback is asynchronous and never
    // stalls this frame.
    //
    // Skipped while the gradient kernel is up: it writes a fixed image with no
    // per-sample variance at all, so the figure would be a meaningless zero.
    // Nothing new to measure on a frame that added no samples, so a converged
    // image stops paying for the reduction rather than re-deriving the same
    // number sixty times a second forever. The `value < 0` clause covers the
    // case where the target was hit before a measurement ever landed.
    const worthMeasuring = samplesThisDispatch > 0 || this.meter.value < 0;
    if (s.mode !== 'gradient' && worthMeasuring && ++this.framesSinceMeasure >= MEASURE_EVERY) {
      if (this.meter.record(encoder, this.accumBuffer, this.renderWidth * this.renderHeight)) {
        this.framesSinceMeasure = 0;
      }
    }

    // Denoising, if asked for. It reads the finished accumulation and writes an
    // Accum-shaped buffer of its own, so the display pass below is unchanged and
    // only the bind group differs. The accumulation is never written: one more
    // sample still converges to the unbiased answer.
    //
    // Off by default. A denoiser trades variance for bias, and measured on the
    // Cornell box it helps the typical pixel 2.4x at 4 spp and *hurts* above
    // roughly 32 — so what this renderer shows unasked is what it computed.
    let displayGroup = this.displayBindGroup;
    if (s.mode !== 'gradient' && s.denoisePasses > 0) {
      const out = this.denoiser.run(
        encoder,
        this.accumBuffer,
        this.renderWidth,
        this.renderHeight,
        { ...DEFAULT_DENOISE, iterations: s.denoisePasses },
      );
      displayGroup = this.device.createBindGroup({
        label: 'display (denoised)',
        layout: this.displayLayout,
        entries: [
          { binding: 0, resource: { buffer: this.displayUniformBuffer } },
          { binding: 1, resource: { buffer: out } },
        ],
      });
    }

    const view = this.context.getCurrentTexture().createView();
    const pass = encoder.beginRenderPass({
      label: 'display',
      colorAttachments: [
        { view, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } },
      ],
    });
    pass.setPipeline(this.displayPipeline);
    pass.setBindGroup(0, displayGroup);
    pass.draw(3); // fullscreen triangle
    pass.end();

    this.device.queue.submit([encoder.finish()]);
    // After the submit, never before: mapping a staging buffer whose copy has
    // not been queued is a validation error.
    this.meter.collect();
    this.accumulatedSamples += samplesThisDispatch;

    this.frameStart = t0;
    this.raysInFlight = this.renderWidth * this.renderHeight * samplesThisDispatch * s.maxDepth;
    return samplesThisDispatch;
  }

  /**
   * Wait for the submitted frame to finish on the GPU, and record its timing.
   *
   * Awaiting this before submitting the next frame is not only how we get an
   * honest number — `performance.now()` around `submit()` measures how long it
   * took to *record* the commands, which is microseconds regardless of how long
   * the GPU then spends, and yields absurd throughput figures. It also bounds
   * the queue depth to one frame, which keeps the page responsive and keeps any
   * single dispatch short enough to stay clear of the driver watchdog.
   *
   * Timestamp queries would give true per-pass GPU time, but they are gated
   * behind a browser flag often enough that this has to work without them, and
   * the overlay says which kind of timing it is showing rather than letting the
   * two be confused. The native harness measures per-stage cost instead, where
   * the feature is always available — see `PT_LBVH_TIMING` in
   * `crates/gpu/src/lbvh.rs`.
   */
  async waitForGpu(): Promise<void> {
    await this.device.queue.onSubmittedWorkDone();
    const dt = performance.now() - this.frameStart;
    this.stats.lastFrameMs = dt;
    // Only frames that actually added samples count towards the accumulation
    // time, so the remaining-time estimate does not grow while the renderer sits
    // idle on a finished image.
    if (this.raysInFlight > 0) this.stats.accumMs += dt;
    this.stats.avgFrameMs =
      this.stats.avgFrameMs === 0 ? dt : this.stats.avgFrameMs * 0.9 + dt * 0.1;
    if (this.raysInFlight > 0 && this.stats.avgFrameMs > 0) {
      this.stats.maxRaysPerSecond = (this.raysInFlight / this.stats.avgFrameMs) * 1000;
    }
    this.adaptScale();
  }

  /**
   * Distance from the eye to whatever is under a display-space pixel, or 0 if
   * that ray hit nothing.
   *
   * This is what makes click-to-focus cost nothing. The depth is already in the
   * accumulation buffer: the denoiser needs a noise-free first-hit distance as
   * a guide channel, and it has been written on every path since build step 16.
   * Focusing the lens is one 32-byte readback of a number that was there anyway.
   *
   * It is an *average* over the samples that landed in the pixel, so a click
   * exactly on a silhouette returns a blend of the near and far surface. That is
   * a fraction of a pixel's worth of ambiguity in a control whose whole job is
   * "roughly there", so it is left alone rather than special-cased.
   */
  async probeDepth(displayX: number, displayY: number): Promise<number> {
    if (this.accumulatedSamples === 0) return 0;
    // The buffer holds a render-resolution image, which during interaction is
    // smaller than the canvas the click arrived in.
    const rx = Math.min(
      this.renderWidth - 1,
      Math.max(0, Math.floor((displayX * this.renderWidth) / this.settings.width)),
    );
    const ry = Math.min(
      this.renderHeight - 1,
      Math.max(0, Math.floor((displayY * this.renderHeight) / this.settings.height)),
    );
    const index = ry * this.renderWidth + rx;

    // 32 bytes covers radiance, the sample count and the guides through depth.
    const staging = this.device.createBuffer({
      label: 'depth probe',
      size: 32,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({ label: 'depth probe' });
    encoder.copyBufferToBuffer(this.accumBuffer, index * ACCUM_BYTES_PER_PIXEL, staging, 0, 32);
    this.device.queue.submit([encoder.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    const f = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();

    // Accum: radiance[0..2], samples[3], albedo[4..6], depth[7].
    return f[7] / Math.max(f[3], 1);
  }

  /**
   * Read the accumulated HDR buffer back to the CPU, already divided by the
   * sample count.
   *
   * This is how the browser gets compared against the native renderers: export
   * a PFM here, and run it through `cargo run -p pt-cli --bin compare`.
   */
  async readbackHDR(): Promise<{ width: number; height: number; data: Float32Array }> {
    // Reports the *render* resolution, which is what is actually in the buffer.
    // Labelling a quarter-resolution frame with the display size would quietly
    // corrupt any comparison made against it.
    const [width, height] = [this.renderWidth, this.renderHeight];
    const size = width * height * ACCUM_BYTES_PER_PIXEL;
    const staging = this.device.createBuffer({
      label: 'readback',
      size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({ label: 'readback' });
    encoder.copyBufferToBuffer(this.accumBuffer, 0, staging, 0, size);
    this.device.queue.submit([encoder.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    const src = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();

    // Take radiance and the sample count out of the Accum struct and divide,
    // matching what the display pass and the native readback both do.
    //
    // The stride is generated, not written here. It was 16 bytes when this file
    // was first written and is 64 now — guide channels for the denoiser, then
    // the sum of squares for the convergence meter — and a hardcoded stride
    // does not fail loudly when the struct grows: the export silently reads
    // every fourth pixel's radiance and calls it an image.
    const floats = ACCUM_BYTES_PER_PIXEL / 4;
    const out = new Float32Array(width * height * 3);
    for (let i = 0; i < width * height; i++) {
      const b = i * floats;
      const n = Math.max(src[b + 3], 1);
      out[i * 3 + 0] = src[b + 0] / n;
      out[i * 3 + 1] = src[b + 1] / n;
      out[i * 3 + 2] = src[b + 2] / n;
    }
    return { width, height, data: out };
  }

  destroy(): void {
    for (const b of [
      this.accumBuffer,
      this.materialBuffer,
      this.primitiveBuffer,
      this.lightBuffer,
      this.positionBuffer,
      this.vertexAttrBuffer,
      this.triangleBuffer,
      this.bvhNodeBuffer,
      this.uniformBuffer,
      this.displayUniformBuffer,
    ]) {
      b?.destroy();
    }
    this.meter?.destroy();
  }
}
