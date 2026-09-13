// RESOLVE: fold finished path radiance into the accumulator.
//
// Runs once per batch, after the bounce loop. One thread per *pixel*, summing
// that pixel's `samples_per_launch` paths — path `pixel + j * pixels` is the
// j-th sample of it. Threading by pixel rather than by path is what keeps this
// free of atomics now that a batch holds several samples of the same pixel.

//!include "common/generated.wgsl"

@group(0) @binding(0) var<uniform> U: Uniforms;
@group(1) @binding(0) var<storage, read> paths: array<PathState>;
@group(1) @binding(1) var<storage, read_write> accum: array<Accum>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pixels = U.width * U.height;
  if (gid.x >= pixels) {
    return;
  }
  // Summed in sample order, which is the order the megakernel's inner loop
  // accumulates them — so the two agree to float reassociation rather than
  // merely in expectation.
  var sum = vec3<f32>(0.0, 0.0, 0.0);
  var sum_sq = vec3<f32>(0.0);
  var albedo = vec3<f32>(0.0);
  var normal = vec3<f32>(0.0);
  var depth = 0.0;
  for (var j: u32 = 0u; j < U.samples_per_launch; j = j + 1u) {
    let q = paths[gid.x + j * pixels];
    sum = sum + q.radiance;
    sum_sq = sum_sq + q.radiance * q.radiance;
    albedo = albedo + q.guide_albedo;
    normal = normal + q.guide_normal;
    depth = depth + q.guide_depth;
  }
  var a = accum[gid.x];
  // `samples` carries the accumulated count, exactly as the megakernel writes
  // it: the display pass divides by it, so progressive rendering in the browser
  // depends on the two architectures agreeing here. The native readback divides
  // by the host's own sample count and ignores it, so this costs it nothing.
  a.radiance = a.radiance + sum;
  a.samples = a.samples + f32(U.samples_per_launch);
  a.albedo = a.albedo + albedo;
  a.normal = a.normal + normal;
  a.depth = a.depth + depth;
  a.radiance_sq = a.radiance_sq + sum_sq;
  accum[gid.x] = a;
}
