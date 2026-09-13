// Evaluates the sRGB transfer function, for the CPU/GPU agreement test.
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
  outputs[gid.x] = vec4<f32>(linear_to_srgb(inputs[gid.x].xyz), 0.0);
}
