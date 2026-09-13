// Generic numerical-evaluation kernel for GPU-side unit tests.
//
// Reads a buffer of inputs, applies one selectable function, writes a buffer of
// outputs. That is enough to check any pure vec3 -> vec3 shader function against
// its Rust twin, which is how transcription errors get caught — a transposed
// colour matrix or a mistyped polynomial coefficient produces a plausible image
// and is essentially invisible to the eye.
//
// The same harness serves the chi-squared BSDF sampling tests at build step 7,
// where sample() and pdf() have to be checked for agreement on the GPU.

//!include "common/tonemap.wgsl"

struct EvalParams {
  mode: u32,
  count: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> P: EvalParams;
@group(0) @binding(1) var<storage, read> inputs: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> outputs: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.count) {
    return;
  }
  let c = inputs[gid.x].xyz;
  outputs[gid.x] = vec4<f32>(tonemap(c, P.mode), 0.0);
}
