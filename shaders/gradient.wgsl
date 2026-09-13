// Build step 1: prove the whole pipeline works before anything else exists.
//
// Writes a known pattern into the same accumulation buffer the path tracer uses
// and is presented by the same display pass, so a correct gradient on screen
// means device init, bind groups, the storage buffer, the display pipeline, the
// swap chain and the colour-space handling are all wired up. It stays in the
// build as a smoke test — when the tracer shows black, this tells you in one
// click whether the problem is the tracer or the plumbing.

//!include "common/generated.wgsl"

@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(4) var<storage, read_write> accum: array<vec4<f32>>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= U.width || gid.y >= U.height) {
    return;
  }
  let uv = vec2<f32>(f32(gid.x) / f32(U.width), f32(gid.y) / f32(U.height));

  // Values are *linear* radiance, so the display pass's sRGB encode will make
  // the ramp look perceptually even. If the on-screen gradient looks washed out
  // and top-heavy instead, the display transform is being applied twice; if it
  // looks dark and bottom-heavy, it is not being applied at all.
  var c = vec3<f32>(uv.x, 1.0 - uv.y, 0.25);

  // A checkerboard in the corner: an unmistakable orientation and aspect marker.
  // Row 0 must appear at the TOP of the image, so this block belongs top-left.
  if (gid.x < 64u && gid.y < 64u) {
    let cell = ((gid.x / 8u) + (gid.y / 8u)) % 2u;
    c = select(vec3<f32>(0.02, 0.02, 0.02), vec3<f32>(0.9, 0.9, 0.9), cell == 0u);
  }

  accum[gid.y * U.width + gid.x] = vec4<f32>(c, 1.0);
}
