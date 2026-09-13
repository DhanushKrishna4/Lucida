// Display pass: resolve the HDR accumulation buffer to the swap chain.
//
// The renderer works in linear HDR end to end; this is the only place a display
// transform is applied. Keeping it in a separate pass rather than tone mapping
// inside the integrator is what makes exposure and operator changes free — they
// re-run this pass, not the light transport, so a converged image stays
// converged while you scrub exposure.
//
// Reading the accumulation buffer in a *fragment* shader, rather than blitting
// through a storage texture, avoids needing read-write storage textures, which
// are not core WebGPU (`rgba32float` is write-only there).

//!include "common/generated.wgsl"
//!include "common/diagnostic.wgsl"
//!include "common/tonemap.wgsl"

struct DisplayUniforms {
  width: u32,
  height: u32,
  exposure: f32,
  // Matches the TONEMAP_* constants in common/tonemap.wgsl and
  // `Tonemap::index()` in crates/core/src/tonemap.rs.
  tonemap_mode: u32,
  // Matches `RenderMode::index()` in crates/core/src/diagnostic.rs.
  render_mode: u32,
  // Scene scale for the depth mode, so one image works for a Cornell box
  // measured in hundreds and a sphere measured in ones.
  depth_scale: f32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> D: DisplayUniforms;
@group(0) @binding(1) var<storage, read> accum: array<Accum>;

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

// Fullscreen triangle, not a quad: one primitive instead of two, no diagonal
// seam where the halves meet, and no vertex buffer at all.
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
  var out: VsOut;
  let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;  // -1, 3, -1
  let y = f32(vi & 2u) * 2.0 - 1.0;          // -1, -1, 3
  out.pos = vec4<f32>(x, y, 0.0, 1.0);
  // Clip space has +y up, image space has +y down, hence the flip.
  out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
  return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
  let x = min(u32(in.uv.x * f32(D.width)), D.width - 1u);
  let y = min(u32(in.uv.y * f32(D.height)), D.height - 1u);
  let s = accum[y * D.width + x];

  // w is the accumulated sample count. Guard the very first frame, before any
  // dispatch has run, rather than emitting NaN across the whole screen.
  let n = max(s.samples, 1.0);

  // Exposure is a pure scale in *scene-linear*, applied before the curve — that
  // is what makes it behave like a camera stop rather than like a brightness
  // slider, and it is why the filmic operators roll off differently as you
  // change it.
  // Diagnostic modes short-circuit before tone mapping.
  //
  // Deliberately: exposure and a filmic curve are for *radiance*, and putting a
  // normal map through one would make a diagnostic misreport its own values,
  // which is the one thing a diagnostic cannot do. sRGB still applies, because
  // these are being looked at on a display.
  //
  // Mirrors `diagnostic::shade` in Rust; the two are diffed in `cargo test`.
  if (D.render_mode != MODE_BEAUTY) {
    let inv = 1.0 / n;
    var c = vec3<f32>(0.0);
    if (D.render_mode == MODE_NORMAL) {
      // Remapped rather than shown raw: half a normal's range is negative, and
      // clamping it to zero would make the two hemispheres identical.
      let nrm = s.normal * inv;
      let len2 = dot(nrm, nrm);
      let unit = select(vec3<f32>(0.0), nrm * inverseSqrt(len2), len2 > 1.0e-12);
      c = unit * 0.5 + vec3<f32>(0.5);
    } else if (D.render_mode == MODE_ALBEDO) {
      // Clamped: the stored albedo deliberately includes emission, which is
      // what keeps the denoiser from demodulating a light fixture into a
      // division by a near-black base colour. A diagnostic bypasses tone
      // mapping, so it has to arrive display-ready; an emitter reads as white.
      c = clamp(s.albedo * inv, vec3<f32>(0.0), vec3<f32>(1.0));
    } else if (D.render_mode == MODE_DEPTH) {
      let d = s.depth * inv;
      // Inverted so near is bright, which reads as depth rather than as fog.
      let v = pow(1.0 - clamp(d / max(D.depth_scale, 1.0e-6), 0.0, 1.0), 0.75);
      c = vec3<f32>(select(0.0, v, d > 0.0));
    } else {
      c = heat_ramp((s.traversal * inv) / HEAT_SCALE);
    }
    return vec4<f32>(linear_to_srgb(c), 1.0);
  }

  // Exposure is a pure scale in *scene-linear*, applied before the curve — that
  // is what makes it behave like a camera stop rather than like a brightness
  // slider, and it is why the filmic operators roll off differently as you
  // change it.
  let hdr = (s.radiance / n) * D.exposure;

  return vec4<f32>(linear_to_srgb(tonemap(hdr, D.tonemap_mode)), 1.0);
}
