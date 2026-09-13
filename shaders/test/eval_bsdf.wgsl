// Evaluates individual BSDF terms for the Rust/WGSL agreement test.
//
// Testing the microfacet math numerically rather than through an image: a
// mistyped coefficient in D, a swapped eta/k, or a differently-interpolated
// energy table all produce a perfectly plausible highlight and are invisible to
// the eye.

//!include "common/generated.wgsl"
//!include "common/math.wgsl"
//!include "common/bsdf.wgsl"

struct EvalParams {
  mode: u32,
  count: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> P: EvalParams;
@group(0) @binding(1) var<storage, read> inputs: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> outputs: array<vec4<f32>>;

const MODE_GGX_D: u32 = 0u;
const MODE_SMITH_G1: u32 = 1u;
const MODE_SMITH_G2: u32 = 2u;
const MODE_FRESNEL_SCHLICK: u32 = 3u;
const MODE_FRESNEL_CONDUCTOR: u32 = 4u;
const MODE_DIRECTIONAL_ALBEDO: u32 = 5u;
const MODE_VNDF_SAMPLE: u32 = 6u;
const MODE_VNDF_PDF: u32 = 7u;
const MODE_SPECULAR_WEIGHT: u32 = 8u;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.count) {
    return;
  }
  let v = inputs[gid.x].xyz;
  var out = vec3<f32>(0.0);

  switch (P.mode) {
    case 0u: { out = vec3<f32>(ggx_d(v.x, v.y)); }
    case 1u: { out = vec3<f32>(smith_g1(v.x, v.y)); }
    case 2u: { out = vec3<f32>(smith_g2(v.x, v.y, v.z)); }
    case 3u: { out = fresnel_schlick(vec3<f32>(v.x, v.x * 0.6, v.x * 0.3), v.y); }
    case 4u: {
      // Copper's measured constants, so eta and k are exercised per channel.
      out = fresnel_conductor(
        v.y,
        vec3<f32>(0.200438, 0.924033, 1.102212),
        vec3<f32>(3.912400, 2.447530, 2.137640),
      );
    }
    case 5u: { out = vec3<f32>(ggx_directional_albedo(v.x, v.y)); }
    case 6u: {
      // v = (alpha, cos_theta_o, u.x); u.y comes from the w component.
      let sin_o = sqrt(max(0.0, 1.0 - v.y * v.y));
      let wo = vec3<f32>(sin_o, 0.0, v.y);
      out = sample_ggx_vndf(wo, v.x, vec2<f32>(v.z, inputs[gid.x].w));
    }
    case 7u: {
      let sin_o = sqrt(max(0.0, 1.0 - v.y * v.y));
      let wo = vec3<f32>(sin_o, 0.0, v.y);
      let m = sample_ggx_vndf(wo, v.x, vec2<f32>(v.z, inputs[gid.x].w));
      out = vec3<f32>(ggx_vndf_pdf(v.x, wo, m));
    }
    case 8u: {
      // The quantity the renderer actually multiplies into throughput:
      // F * G2 / G1(wo), with F white. D cancels between the BRDF and the VNDF
      // pdf, so unlike either of them alone this is well conditioned even at
      // MIN_ALPHA.
      let sin_o = sqrt(max(0.0, 1.0 - v.y * v.y));
      let wo = vec3<f32>(sin_o, 0.0, v.y);
      let m = sample_ggx_vndf(wo, v.x, vec2<f32>(v.z, inputs[gid.x].w));
      let wi = reflect_about(wo, m);
      if (wi.z <= 0.0) {
        out = vec3<f32>(0.0);
      } else {
        out = vec3<f32>(smith_g2(v.x, wo.z, wi.z) / smith_g1(v.x, wo.z));
      }
    }
    default: {}
  }
  outputs[gid.x] = vec4<f32>(out, 0.0);
}
