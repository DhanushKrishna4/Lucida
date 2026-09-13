// Evaluate the diagnostic colour ramp over a list of inputs, so the WGSL can be
// diffed against its Rust twin. See crates/gpu/src/eval.rs.
//
// A colour ramp is exactly the kind of thing that looks fine while being wrong:
// a mistyped stop or a transposed interpolation produces a perfectly plausible
// gradient that nobody can spot by eye.

//!include "common/generated.wgsl"
//!include "common/math.wgsl"
//!include "common/diagnostic.wgsl"

struct EvalParams {
  mode  : u32,
  count : u32,
  _pad0 : u32,
  _pad1 : u32,
};

@group(0) @binding(0) var<uniform> P: EvalParams;
@group(0) @binding(1) var<storage, read>       inputs  : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> outputs : array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.count) {
    return;
  }
  // x carries t; the rest is unused.
  outputs[i] = vec4<f32>(heat_ramp(inputs[i].x), 0.0);
}
