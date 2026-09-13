// GENERATE: one camera ray per path, and a full queue.
//
// Runs once per *batch* of samples. The path pool is a flat range of
//
//   index -> (pixel, sample) = (index % pixels, sample_offset + index / pixels)
//
// so a batch of k samples puts k * pixels paths in flight at once. Batching
// matters because the bounce loop costs four compute passes whether or not the
// queue still holds anything, and the host cannot know that it is empty without
// a readback that would stall for longer than the whole architecture saves. One
// batch of k samples runs the loop once instead of k times.
//
// The batch is bounded by memory, not by taste: every per-path buffer has to fit
// WebGPU's guaranteed 128 MiB storage-binding limit, which the host divides out.
//
// Declares only the bindings it uses. WebGPU guarantees eight storage buffers
// per shader *stage*, counted across every bind group, so one layout shared by
// all six kernels would need fourteen and fail outright.

//!include "common/generated.wgsl"

@group(0) @binding(0) var<uniform> U: Uniforms;
@group(1) @binding(0) var<storage, read_write> paths: array<PathState>;
@group(1) @binding(1) var<storage, read_write> queue_out: array<u32>;

//!include "common/math.wgsl"
//!include "common/rng.wgsl"
//!include "common/ray.wgsl"
//!include "common/camera.wgsl"

// Marks a queue slot that holds no path. EXTEND checks for it instead of
// reading a queue length, which saves that kernel a binding it cannot spare.
const INVALID_PATH: u32 = 0xFFFFFFFFu;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  let pixels = U.width * U.height;
  let total = pixels * U.samples_per_launch;
  if (index >= total) {
    // The dispatch rounds up to whole workgroups, so the tail slots would
    // otherwise hold whatever was in memory.
    if (index < arrayLength(&queue_out)) {
      queue_out[index] = INVALID_PATH;
    }
    return;
  }

  // Unflatten the path pool. The pixel varies fastest so that a workgroup of 64
  // consecutive paths covers 64 neighbouring pixels of one sample — the same
  // coherence the single-sample version had — rather than 64 samples of one
  // pixel.
  let pixel = index % pixels;
  let sample = U.sample_offset + index / pixels;
  let x = pixel % U.width;
  let y = pixel / U.width;

  // Seeded exactly as the megakernel seeds it, so the two architectures draw
  // identical random streams and their images can be compared at
  // float-reassociation level rather than merely statistically. Note the seed is
  // (pixel, sample), never the path index — batching must not move the stream.
  var rng = rng_init(U.frame_seed, pixel, sample);

  // Draw order is part of the contract: pixel jitter, then the lens sample,
  // then whatever the path consumes per bounce.
  let pixel_uv = rng_next_vec2(&rng);
  let lens_uv = rng_next_vec2(&rng);
  let ray = generate_ray(x, y, pixel_uv, lens_uv);

  var p: PathState;
  p.origin = ray.origin;
  p.direction = ray.dir;
  p.pixel = pixel;
  p.rng_state = rng.state;
  // Only the dimension needs carrying; `index` and `seed` are rederived in
  // SHADE from the pixel and the uniforms.
  p.sampler_dim = rng.dim;
  p.throughput = vec3<f32>(1.0, 1.0, 1.0);
  p.radiance = vec3<f32>(0.0, 0.0, 0.0);
  p.depth = 0u;
  p.prev_bsdf_pdf = 0.0;
  p.prev_position = ray.origin;
  paths[index] = p;

  // Every path starts alive, so the first queue is the identity and needs no
  // atomics to build.
  queue_out[index] = index;
}
