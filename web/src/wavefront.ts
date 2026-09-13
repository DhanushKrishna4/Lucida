/**
 * Wavefront path tracing in the browser.
 *
 * The same six WGSL kernels the native harness runs, driven from TypeScript.
 * Nothing here reimplements light transport — that all lives in the shaders and
 * is validated against the CPU reference in `cargo test`. What this file owns is
 * the plumbing the split introduces: the per-path buffers, the six bind group
 * layouts, and the batching.
 *
 * # Why six bind group layouts and not one
 *
 * WebGPU guarantees eight storage buffers per shader *stage*, counted across
 * every bind group. The union of what these kernels touch is fourteen, so a
 * single shared layout fails outright at pipeline creation. Each kernel
 * therefore declares exactly what it uses, split into a scene group (0) and a
 * wavefront-state group (1). The layouts here must match the bindings in
 * `shaders/wavefront/*.wgsl` exactly, and mirror `crates/gpu/src/wavefront.rs`.
 *
 * # Why this submits its own command buffers
 *
 * `queue.writeBuffer` is not ordered against encoder commands: the queue flushes
 * every pending write *before* the command buffers submitted alongside them. A
 * batch needs its own uniform values (sample offset, batch size), so recording
 * several batches into one encoder would apply every uniform write first and
 * then render every batch with the last one's seed — the same samples, over and
 * over, averaged. The image looks right and simply never converges. Submitting
 * per batch is what keeps each batch's uniforms paired with its dispatches.
 */
import generateSource from '../../shaders/wavefront/generate.wgsl';
import extendSource from '../../shaders/wavefront/extend.wgsl';
import shadeSource from '../../shaders/wavefront/shade.wgsl';
import connectSource from '../../shaders/wavefront/connect.wgsl';
import resetSource from '../../shaders/wavefront/reset.wgsl';
import resolveSource from '../../shaders/wavefront/resolve.wgsl';

import { captureErrors, createShaderModule } from './webgpu';
import {
  DISPATCH_ARGS_SIZE,
  HIT_RECORD_SIZE,
  PATH_STATE_SIZE,
  SHADOW_RAY_SIZE,
  WAVEFRONT_COUNTERS_SIZE,
} from './generated/layout';

/** Must match `@workgroup_size` in every wavefront kernel. */
const WORKGROUP = 64;

// Strides for the per-path buffers, generated from `crates/core/src/gpu_layout.rs`.
//
// These used to be literals under a comment naming that file as the source of
// truth, which reads as generated and drifts like a copy. `PathState` grew from
// 80 to 112 bytes when the denoiser's guide channels moved into it and the
// literal stayed at 80, so this allocated 71% of the pool the shader indexed —
// silently, because WGSL clamps an out-of-bounds index rather than faulting, so
// paths collided on the last valid slot. The browser's wavefront then differed
// from its own megakernel by 446% while still looking like a noisy render.
const PATH_STATE_BYTES = PATH_STATE_SIZE;
const HIT_RECORD_BYTES = HIT_RECORD_SIZE;
const SHADOW_RAY_BYTES = SHADOW_RAY_SIZE;
const COUNTERS_BYTES = WAVEFRONT_COUNTERS_SIZE;
const DISPATCH_ARGS_BYTES = DISPATCH_ARGS_SIZE;

/**
 * Base path-pool budget, scaled by the depth limit.
 *
 * The bounce loop costs four compute passes per bounce whether the queue still
 * holds anything or not, and the host cannot tell that it has emptied without a
 * readback that would stall for longer than the architecture saves. Batching k
 * samples divides that fixed cost by k; the price is that every per-path buffer
 * grows by k. Measured on an M3, the useful pool grows with the depth limit and
 * flattens once launch overhead stops dominating — see the comment in
 * `crates/gpu/src/wavefront.rs`, which carries the numbers and must stay in
 * step with this.
 */
const BASE_PATH_BUDGET = 1 << 18;

/** WebGPU's *guaranteed* storage-binding limit. Not the adapter's, so that the
 *  browser behaves like the native benchmark predicts. */
const MAX_BINDING_BYTES = 128 * 1024 * 1024;

/** Scene-side buffers, owned by the Renderer and bound read-only here. */
export interface WavefrontScene {
  uniforms: GPUBuffer;
  materials: GPUBuffer;
  primitives: GPUBuffer;
  lights: GPUBuffer;
  positions: GPUBuffer;
  vertexAttrs: GPUBuffer;
  triangles: GPUBuffer;
  bvhNodes: GPUBuffer;
  /// The environment map arrives as textures, not storage buffers: SHADE
  /// already binds exactly eight of those, WebGPU's guaranteed per-stage limit.
  envRadiance: GPUTextureView;
  envCdf: GPUTextureView;
}

export interface WavefrontFrame {
  width: number;
  height: number;
  /** Total samples to take this frame, across however many batches it needs. */
  samples: number;
  maxDepth: number;
  /** Samples already accumulated, so each batch seeds from the right offset. */
  sampleOffset: number;
  /**
   * Write the uniform block for one batch. Called once per batch, immediately
   * before that batch is submitted — see the note on ordering above.
   */
  writeUniforms(sampleOffset: number, samplesPerLaunch: number): void;
}

function storage(binding: number, readOnly: boolean): GPUBindGroupLayoutEntry {
  return {
    binding,
    visibility: GPUShaderStage.COMPUTE,
    buffer: { type: readOnly ? 'read-only-storage' : 'storage' },
  };
}
const ro = (b: number) => storage(b, true);
const rw = (b: number) => storage(b, false);
const tex = (b: number): GPUBindGroupLayoutEntry => ({
  binding: b,
  visibility: GPUShaderStage.COMPUTE,
  // `unfilterable-float` because the shader only ever calls textureLoad, which
  // needs no filtering — and must not have any, since the CDF describes a
  // piecewise-constant image.
  texture: { sampleType: 'unfilterable-float', viewDimension: '2d' },
});
const uni = (b: number): GPUBindGroupLayoutEntry => ({
  binding: b,
  visibility: GPUShaderStage.COMPUTE,
  buffer: { type: 'uniform' },
});

export class WavefrontPass {
  private readonly device: GPUDevice;
  private scene!: WavefrontScene;
  private accum!: GPUBuffer;

  private genSceneL!: GPUBindGroupLayout;
  private genStateL!: GPUBindGroupLayout;
  private extSceneL!: GPUBindGroupLayout;
  private extStateL!: GPUBindGroupLayout;
  private shadeSceneL!: GPUBindGroupLayout;
  private shadeStateL!: GPUBindGroupLayout;
  private connSceneL!: GPUBindGroupLayout;
  private connStateL!: GPUBindGroupLayout;
  private resetSceneL!: GPUBindGroupLayout;
  private resetStateL!: GPUBindGroupLayout;
  private resSceneL!: GPUBindGroupLayout;
  private resStateL!: GPUBindGroupLayout;

  private generate!: GPUComputePipeline;
  private extend!: GPUComputePipeline;
  private shade!: GPUComputePipeline;
  private connect!: GPUComputePipeline;
  private reset!: GPUComputePipeline;
  private resolve!: GPUComputePipeline;

  // Per-path state. Reallocated only when the pool size changes.
  private paths?: GPUBuffer;
  private hits?: GPUBuffer;
  private shadowRays?: GPUBuffer;
  private counters?: GPUBuffer;
  private dispatchArgs?: GPUBuffer;
  private queueA?: GPUBuffer;
  private queueB?: GPUBuffer;
  private allocatedPaths = 0;

  private genScene!: GPUBindGroup;
  private genStateB!: GPUBindGroup;
  private extScene!: GPUBindGroup;
  private extStateA!: GPUBindGroup;
  private extStateB!: GPUBindGroup;
  private shadeScene!: GPUBindGroup;
  private shadeStateAB!: GPUBindGroup;
  private shadeStateBA!: GPUBindGroup;
  private connScene!: GPUBindGroup;
  private connState!: GPUBindGroup;
  private resetScene!: GPUBindGroup;
  private resetStateA!: GPUBindGroup;
  private resetStateB!: GPUBindGroup;
  private resScene!: GPUBindGroup;
  private resState!: GPUBindGroup;

  private constructor(device: GPUDevice) {
    this.device = device;
  }

  static async create(device: GPUDevice): Promise<WavefrontPass> {
    const p = new WavefrontPass(device);
    await p.buildPipelines();
    return p;
  }

  /** Bytes currently held in per-path state, for the stats readout. */
  get gpuBytes(): number {
    if (this.allocatedPaths === 0) return 0;
    return (
      this.allocatedPaths *
        (PATH_STATE_BYTES + HIT_RECORD_BYTES + SHADOW_RAY_BYTES + 4 + 4) +
      COUNTERS_BYTES +
      DISPATCH_ARGS_BYTES
    );
  }

  private async buildPipelines(): Promise<void> {
    const d = this.device;
    const layout = (label: string, entries: GPUBindGroupLayoutEntry[]) =>
      d.createBindGroupLayout({ label, entries });

    // These must match the @group/@binding declarations in each kernel, and the
    // Rust harness in crates/gpu/src/wavefront.rs. A mismatch is a pipeline
    // creation error rather than a wrong image, which is the good outcome.
    this.genSceneL = layout('generate scene', [uni(0)]);
    this.genStateL = layout('generate state', [rw(0), rw(1)]);
    this.extSceneL = layout('extend scene', [uni(0), ro(1), ro(2), ro(3), ro(4), ro(5)]);
    this.extStateL = layout('extend state', [ro(0), rw(1), ro(2)]);
    this.shadeSceneL = layout('shade scene', [uni(0), ro(1), ro(2), tex(90), tex(91)]);
    this.shadeStateL = layout('shade state', [rw(0), ro(1), rw(2), rw(3), ro(4), rw(5)]);
    this.connSceneL = layout('connect scene', [uni(0), ro(1), ro(2), ro(3), ro(4)]);
    // Binding 2 is read-write because the counters struct contains atomics,
    // which WGSL requires even for a kernel that only reads one plain field.
    this.connStateL = layout('connect state', [ro(0), rw(1), rw(2)]);
    this.resetSceneL = layout('reset scene', [uni(0)]);
    this.resetStateL = layout('reset state', [rw(0), rw(1), rw(2)]);
    this.resSceneL = layout('resolve scene', [uni(0)]);
    this.resStateL = layout('resolve state', [ro(0), rw(1)]);

    const [gen, ext, sha, con, res, rsv] = await Promise.all([
      createShaderModule(d, 'wavefront/generate.wgsl', generateSource),
      createShaderModule(d, 'wavefront/extend.wgsl', extendSource),
      createShaderModule(d, 'wavefront/shade.wgsl', shadeSource),
      createShaderModule(d, 'wavefront/connect.wgsl', connectSource),
      createShaderModule(d, 'wavefront/reset.wgsl', resetSource),
      createShaderModule(d, 'wavefront/resolve.wgsl', resolveSource),
    ]);

    const pipeline = (
      label: string,
      module: GPUShaderModule,
      groups: GPUBindGroupLayout[],
    ) =>
      d.createComputePipeline({
        label,
        layout: d.createPipelineLayout({ bindGroupLayouts: groups }),
        compute: { module, entryPoint: 'main' },
      });

    await captureErrors(d, 'creating wavefront pipelines', () => {
      this.generate = pipeline('generate', gen, [this.genSceneL, this.genStateL]);
      this.extend = pipeline('extend', ext, [this.extSceneL, this.extStateL]);
      this.shade = pipeline('shade', sha, [this.shadeSceneL, this.shadeStateL]);
      this.connect = pipeline('connect', con, [this.connSceneL, this.connStateL]);
      this.reset = pipeline('reset', res, [this.resetSceneL, this.resetStateL]);
      this.resolve = pipeline('resolve', rsv, [this.resSceneL, this.resStateL]);
    });
  }

  /**
   * Point the pass at the scene and accumulator it should use.
   *
   * Called whenever either is reallocated — a scene load, or a resize. The
   * per-path buffers are keyed separately, on the pool size, so a scene change
   * alone does not throw them away.
   */
  setScene(scene: WavefrontScene, accum: GPUBuffer): void {
    this.scene = scene;
    this.accum = accum;
    this.allocatedPaths = 0; // force the state bind groups to be rebuilt
  }

  /**
   * How many whole samples to keep in flight, given the pool budget and the
   * hard binding limit. Mirrors `crates/gpu/src/wavefront.rs`.
   */
  private batchFor(pixels: number, maxDepth: number, samples: number): number {
    const maxPaths = Math.floor(MAX_BINDING_BYTES / PATH_STATE_BYTES);
    const budget = Math.min(BASE_PATH_BUDGET * Math.ceil(Math.max(maxDepth, 1) / 8), maxPaths);
    const byBudget = Math.floor(budget / Math.max(pixels, 1));
    const byLimit = Math.max(Math.floor(maxPaths / Math.max(pixels, 1)), 1);
    return Math.max(1, Math.min(byBudget, byLimit, Math.max(samples, 1)));
  }

  private ensureBuffers(inFlight: number): void {
    if (inFlight === this.allocatedPaths) return;
    const d = this.device;

    for (const b of [
      this.paths,
      this.hits,
      this.shadowRays,
      this.counters,
      this.dispatchArgs,
      this.queueA,
      this.queueB,
    ]) {
      b?.destroy();
    }

    const S = GPUBufferUsage.STORAGE;
    const buf = (label: string, size: number, usage: number) =>
      d.createBuffer({ label, size, usage });

    this.paths = buf('path state', inFlight * PATH_STATE_BYTES, S);
    this.hits = buf('hit records', inFlight * HIT_RECORD_BYTES, S);
    this.shadowRays = buf('shadow rays', inFlight * SHADOW_RAY_BYTES, S);
    this.counters = buf('wavefront counters', COUNTERS_BYTES, S | GPUBufferUsage.COPY_DST);
    // Separate from the counters, and INDIRECT. They cannot share a buffer:
    // WebGPU forbids one being both a read-write binding and the indirect
    // source within a single dispatch, and SHADE writes counters while being
    // dispatched indirectly.
    this.dispatchArgs = buf(
      'dispatch args',
      DISPATCH_ARGS_BYTES,
      S | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    );
    this.queueA = buf('queue a', inFlight * 4, S);
    this.queueB = buf('queue b', inFlight * 4, S);
    this.allocatedPaths = inFlight;

    const group = (label: string, l: GPUBindGroupLayout, bufs: GPUBuffer[]) =>
      d.createBindGroup({
        label,
        layout: l,
        entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })),
      });

    const s = this.scene;
    this.genScene = group('generate scene', this.genSceneL, [s.uniforms]);
    this.genStateB = group('generate -> b', this.genStateL, [this.paths, this.queueB]);

    this.extScene = group('extend scene', this.extSceneL, [
      s.uniforms,
      s.primitives,
      s.positions,
      s.vertexAttrs,
      s.triangles,
      s.bvhNodes,
    ]);
    this.extStateA = group('extend from a', this.extStateL, [this.paths, this.hits, this.queueA]);
    this.extStateB = group('extend from b', this.extStateL, [this.paths, this.hits, this.queueB]);

    // Not via `group`, which assumes sequential buffer bindings: the two
    // environment textures sit at 90 and 91 to stay clear of the storage range.
    this.shadeScene = d.createBindGroup({
      label: 'shade scene',
      layout: this.shadeSceneL,
      entries: [
        { binding: 0, resource: { buffer: s.uniforms } },
        { binding: 1, resource: { buffer: s.materials } },
        { binding: 2, resource: { buffer: s.lights } },
        { binding: 90, resource: s.envRadiance },
        { binding: 91, resource: s.envCdf },
      ],
    });
    this.shadeStateAB = group('shade a->b', this.shadeStateL, [
      this.paths,
      this.hits,
      this.shadowRays,
      this.counters,
      this.queueA,
      this.queueB,
    ]);
    this.shadeStateBA = group('shade b->a', this.shadeStateL, [
      this.paths,
      this.hits,
      this.shadowRays,
      this.counters,
      this.queueB,
      this.queueA,
    ]);

    // No vertex attributes: occlusion never shades a surface. That is worth a
    // binding, and bindings are the scarce resource.
    this.connScene = group('connect scene', this.connSceneL, [
      s.uniforms,
      s.primitives,
      s.positions,
      s.triangles,
      s.bvhNodes,
    ]);
    this.connState = group('connect state', this.connStateL, [
      this.shadowRays,
      this.paths,
      this.counters,
    ]);

    this.resetScene = group('reset scene', this.resetSceneL, [s.uniforms]);
    this.resetStateA = group('reset -> a', this.resetStateL, [
      this.counters,
      this.dispatchArgs,
      this.queueA,
    ]);
    this.resetStateB = group('reset -> b', this.resetStateL, [
      this.counters,
      this.dispatchArgs,
      this.queueB,
    ]);

    this.resScene = group('resolve scene', this.resSceneL, [s.uniforms]);
    this.resState = group('resolve state', this.resStateL, [this.paths, this.accum]);
  }

  /**
   * Take `frame.samples` samples, submitting one command buffer per batch.
   *
   * Returns without submitting anything when there is nothing to do, so the
   * caller can still run its display pass on a converged image.
   */
  render(frame: WavefrontFrame): void {
    if (frame.samples <= 0) return;
    const d = this.device;
    const pixels = frame.width * frame.height;
    const batch = this.batchFor(pixels, frame.maxDepth, frame.samples);
    this.ensureBuffers(pixels * batch);

    const pixelGroups = Math.ceil(pixels / WORKGROUP);

    for (let base = 0; base < frame.samples; base += batch) {
      const thisBatch = Math.min(batch, frame.samples - base);
      const poolGroups = Math.ceil((pixels * thisBatch) / WORKGROUP);

      frame.writeUniforms(frame.sampleOffset + base, thisBatch);

      // Every path starts alive, so the first queue is the identity and its
      // length is known here — no atomics needed to build it.
      d.queue.writeBuffer(this.counters!, 0, new Uint32Array([0, 0, 0, 0]));
      d.queue.writeBuffer(
        this.dispatchArgs!,
        0,
        // trace = (poolGroups,1,1), shadow = (0,1,1). Scalar u32 fields, not
        // vec3: a vec3<u32> has alignment 16 in WGSL, which would push `shadow`
        // to offset 16 and silently misread every dispatch after the first.
        new Uint32Array([poolGroups, 1, 1, 0, 1, 1, 0, 0]),
      );

      const encoder = d.createCommandEncoder({ label: 'wavefront batch' });

      {
        const pass = encoder.beginComputePass({ label: 'generate' });
        pass.setPipeline(this.generate);
        pass.setBindGroup(0, this.genScene);
        pass.setBindGroup(1, this.genStateB);
        pass.dispatchWorkgroups(poolGroups, 1, 1);
        pass.end();
      }

      for (let bounce = 0; bounce < frame.maxDepth; bounce++) {
        // GENERATE filled queue_b, so bounce 0 reads b and writes a.
        const readsB = bounce % 2 === 0;
        {
          const pass = encoder.beginComputePass({ label: 'extend' });
          pass.setPipeline(this.extend);
          pass.setBindGroup(0, this.extScene);
          pass.setBindGroup(1, readsB ? this.extStateB : this.extStateA);
          pass.dispatchWorkgroupsIndirect(this.dispatchArgs!, 0);
          pass.end();
        }
        {
          const pass = encoder.beginComputePass({ label: 'shade' });
          pass.setPipeline(this.shade);
          pass.setBindGroup(0, this.shadeScene);
          pass.setBindGroup(1, readsB ? this.shadeStateBA : this.shadeStateAB);
          pass.dispatchWorkgroupsIndirect(this.dispatchArgs!, 0);
          pass.end();
        }
        // RESET runs between SHADE and CONNECT, so CONNECT can be sized from the
        // shadow rays SHADE just appended. It also rolls the path queue forward
        // for the next bounce, which nothing before the next EXTEND reads.
        {
          const pass = encoder.beginComputePass({ label: 'reset' });
          pass.setPipeline(this.reset);
          pass.setBindGroup(0, this.resetScene);
          pass.setBindGroup(1, readsB ? this.resetStateA : this.resetStateB);
          pass.dispatchWorkgroups(1, 1, 1);
          pass.end();
        }
        {
          const pass = encoder.beginComputePass({ label: 'connect' });
          pass.setPipeline(this.connect);
          pass.setBindGroup(0, this.connScene);
          pass.setBindGroup(1, this.connState);
          // Offset 12: the shadow dispatch, past the three trace words.
          pass.dispatchWorkgroupsIndirect(this.dispatchArgs!, 12);
          pass.end();
        }
      }

      {
        const pass = encoder.beginComputePass({ label: 'resolve' });
        pass.setPipeline(this.resolve);
        pass.setBindGroup(0, this.resScene);
        pass.setBindGroup(1, this.resState);
        pass.dispatchWorkgroups(pixelGroups, 1, 1);
        pass.end();
      }

      d.queue.submit([encoder.finish()]);
    }
  }

  destroy(): void {
    for (const b of [
      this.paths,
      this.hits,
      this.shadowRays,
      this.counters,
      this.dispatchArgs,
      this.queueA,
      this.queueB,
    ]) {
      b?.destroy();
    }
    this.allocatedPaths = 0;
  }
}
