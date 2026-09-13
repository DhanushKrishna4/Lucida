// Microfacet BSDFs. Mirrors `crates/core/src/bsdf.rs`.
//
// Everything is in the **local shading frame**, where the surface normal is +Z
// and both `wo` (toward the previous path vertex) and `wi` (toward the next)
// point away from the surface. Most microfacet bugs are sign and direction
// bugs rather than algebra, so the convention is stated once and kept.
//
// Requires: math.wgsl (PI, INV_PI, luminance), generated.wgsl (Material),
// ggx_energy.wgsl (the directional albedo table).

//!include "common/ggx_energy.wgsl"

// Smallest GGX width f32 can carry without destroying energy.
//
// The GGX denominator needs sin^2(theta), but the call site only has
// cos(theta) as an f32. Near the lobe centre cos(theta) is within one ULP of
// 1.0, so 1 - cos^2 underflows to zero and D blows up. Measured, the NDF
// integrates to 6.1 instead of 1 at alpha = 1e-4 — a near-mirror surface would
// emit six times the light that fell on it. 2e-3 is roughness 0.045, still a
// sharp highlight; a true mirror needs a delta lobe, which arrives with smooth
// dielectrics at build step 12.
const MIN_ALPHA: f32 = 2.0e-3;

fn roughness_to_alpha(roughness: f32) -> f32 {
  return max(roughness * roughness, MIN_ALPHA);
}

// GGX / Trowbridge-Reitz normal distribution.
//
// The textbook denominator `(n.m)^2 (alpha^2 - 1) + 1` catastrophically cancels
// at low roughness. The algebraically identical `alpha^2 c^2 + sin^2(theta)`
// used here does not, because both terms are small and positive.
fn ggx_d(alpha: f32, cos_theta_m: f32) -> f32 {
  if (cos_theta_m <= 0.0) {
    return 0.0;
  }
  let a2 = alpha * alpha;
  let c2 = cos_theta_m * cos_theta_m;
  let sin2 = max(0.0, 1.0 - c2);
  let denom = a2 * c2 + sin2;
  return a2 / (PI * denom * denom);
}

// Smith's lambda: the ratio of masked to visible microsurface area seen from v.
fn smith_lambda(alpha: f32, cos_theta: f32) -> f32 {
  let c = abs(cos_theta);
  if (c >= 1.0) {
    return 0.0;
  }
  let c2 = c * c;
  let tan2 = (1.0 - c2) / max(c2, 1.0e-12);
  return 0.5 * (-1.0 + sqrt(1.0 + alpha * alpha * tan2));
}

fn smith_g1(alpha: f32, cos_theta: f32) -> f32 {
  return 1.0 / (1.0 + smith_lambda(alpha, cos_theta));
}

// Height-correlated Smith masking-shadowing.
//
// Not G1(wo) * G1(wi). The separable form assumes masking and shadowing are
// independent, which is wrong: a microfacet high on the surface is more likely
// visible from *both* directions and one low down is likely hidden from both.
// Treating them as independent under-counts joint visibility and darkens rough
// surfaces, most visibly at grazing angles.
fn smith_g2(alpha: f32, cos_theta_o: f32, cos_theta_i: f32) -> f32 {
  return 1.0 / (1.0 + smith_lambda(alpha, cos_theta_o) + smith_lambda(alpha, cos_theta_i));
}

// Sample the distribution of visible normals (Heitz 2018).
//
// Sampling D directly proposes microfacets facing away from wo, which are
// masked and contribute nothing — at high roughness most samples are wasted
// that way. VNDF samples only what wo can see, and the estimator then collapses
// to F * G2 / G1(wo) with D cancelling entirely, so there is no division by a
// near-zero density and no fireflies from the pdf.
fn sample_ggx_vndf(wo: vec3<f32>, alpha: f32, u: vec2<f32>) -> vec3<f32> {
  // Stretch so the ellipsoid becomes a hemisphere.
  let v = normalize(vec3<f32>(alpha * wo.x, alpha * wo.y, wo.z));

  let lensq = v.x * v.x + v.y * v.y;
  var t1: vec3<f32>;
  if (lensq > 0.0) {
    t1 = vec3<f32>(-v.y, v.x, 0.0) / sqrt(lensq);
  } else {
    t1 = vec3<f32>(1.0, 0.0, 0.0);
  }
  let t2 = cross(v, t1);

  // Uniform on the disk, squashed toward the silhouette by how oblique the view
  // is — this is what matches the density to *projected* rather than raw area.
  let r = sqrt(u.x);
  let phi = 2.0 * PI * u.y;
  let p1 = r * cos(phi);
  var p2 = r * sin(phi);
  let s = 0.5 * (1.0 + v.z);
  p2 = (1.0 - s) * sqrt(max(0.0, 1.0 - p1 * p1)) + s * p2;

  let nh = p1 * t1 + p2 * t2 + sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2)) * v;

  // Unstretch.
  return normalize(vec3<f32>(alpha * nh.x, alpha * nh.y, max(0.0, nh.z)));
}

// pdf of the VNDF sample with respect to the reflected direction wi.
//
//   pdf(wi) = D_v(m) / (4 (wo.m)) = G1(wo) D(m) / (4 (n.wo))
//
// The (wo.m) cancels against the reflection Jacobian — and that is exactly the
// term that would otherwise go to zero at grazing angles.
fn ggx_vndf_pdf(alpha: f32, wo: vec3<f32>, m: vec3<f32>) -> f32 {
  let cos_o = wo.z;
  if (cos_o <= 0.0) {
    return 0.0;
  }
  return smith_g1(alpha, cos_o) * ggx_d(alpha, m.z) / (4.0 * cos_o);
}

// Density of the half-vector m itself, before any Jacobian.
//
//   D_vis(m) = G1(wo) * D(m) * max(0, wo.m) / cos_theta_o
//
// Distinct from `ggx_vndf_pdf`, which despite its name is the density of the
// *reflected direction*: that one has already been through the reflection
// Jacobian 1/(4|wo.m|), and the |wo.m| cancelled on the way. Reusing it for
// transmission, whose Jacobian is entirely different, is wrong by 4|wo.m|.
//
// `wo.m <= 0` returns zero by definition rather than as a guard: the visible
// normal distribution contains only microfacets facing the viewer, and using
// |wo.m| assigns density to ones the sampler can never produce.
fn ggx_vndf_half_pdf(alpha: f32, wo: vec3<f32>, m: vec3<f32>) -> f32 {
  let cos_o = wo.z;
  let cos_om = dot(wo, m);
  if (cos_o <= 0.0 || cos_om <= 0.0) {
    return 0.0;
  }
  return smith_g1(alpha, cos_o) * ggx_d(alpha, abs(m.z)) * cos_om / cos_o;
}

// Exact unpolarised Fresnel reflectance for a dielectric interface.
// `eta` is the relative index, n_transmitted / n_incident. Returns 1 under
// total internal reflection, where no transmitted component exists.
fn fresnel_dielectric(cos_theta_i: f32, eta: f32) -> f32 {
  let c = abs(clamp(cos_theta_i, -1.0, 1.0));
  let sin2_t = (1.0 - c * c) / (eta * eta);
  if (sin2_t >= 1.0) {
    return 1.0;
  }
  let cos_t = sqrt(max(1.0 - sin2_t, 0.0));
  let r_parallel = (eta * c - cos_t) / (eta * c + cos_t);
  let r_perpendicular = (c - eta * cos_t) / (c + eta * cos_t);
  return 0.5 * (r_parallel * r_parallel + r_perpendicular * r_perpendicular);
}

struct Refraction {
  wt: vec3<f32>,
  valid: bool,   // false under total internal reflection
};

// Refract `wo` about microfacet normal `m` with relative index eta = n_t / n_i.
// Both point away from the surface; the result points into it.
//
// Snell in vector form:  wt = -wo/eta + (cos_i/eta - cos_t) * m
//
// Invalid under total internal reflection, which is not an error to clamp away:
// past the critical angle there is genuinely no transmitted direction, and
// inventing a grazing one puts energy where the pdf does not describe it.
fn refract_about(wo: vec3<f32>, m: vec3<f32>, eta: f32) -> Refraction {
  var out: Refraction;
  out.wt = vec3<f32>(0.0);
  out.valid = false;
  let cos_i = dot(wo, m);
  let sin2_i = max(1.0 - cos_i * cos_i, 0.0);
  let sin2_t = sin2_i / (eta * eta);
  if (sin2_t >= 1.0) {
    return out;
  }
  let cos_t = sqrt(1.0 - sin2_t);
  let signed_cos_t = select(-cos_t, cos_t, cos_i >= 0.0);
  let wt = -wo / eta + (cos_i / eta - signed_cos_t) * m;
  let len2 = dot(wt, wt);
  if (len2 <= 1.0e-20) {
    return out;
  }
  out.wt = wt * inverseSqrt(len2);
  out.valid = true;
  return out;
}

// Half-vector connecting wo and a transmitted wi, flipped into z > 0.
//
// Reflection's half-vector bisects the two directions. Refraction's does not:
// they are in different media, so the microfacet that connects them is weighted
// by the indices (Walter 2007 eq. 16), h = normalize(wo + eta * wi).
fn transmission_half_vector(wo: vec3<f32>, wi: vec3<f32>, eta: f32) -> vec3<f32> {
  let h_raw = wo + eta * wi;
  if (dot(h_raw, h_raw) <= 1.0e-12) {
    return vec3<f32>(0.0);
  }
  let h = normalize(h_raw);
  return select(h, -h, h.z < 0.0);
}

// The BTDF (Walter 2007 eq. 21), with the radiance-transport correction folded
// in.
//
// A BTDF is not symmetric: radiance is compressed by eta^2 entering a denser
// medium, so light transport and importance transport differ by that factor. A
// path tracer starting at the camera transports importance and carries
// factor = 1/eta, and eta^2 * (1/eta)^2 = 1 — the two cancel exactly, which is
// why no eta^2 appears below. An implementation that keeps the eta^2 and forgets
// the correction is too bright by 2.25x for glass and merely looks "glowy".
fn dielectric_eval(alpha: f32, eta: f32, wo: vec3<f32>, wi: vec3<f32>) -> f32 {
  if (wo.z * wi.z >= 0.0) {
    return 0.0;
  }
  let h = transmission_half_vector(wo, wi, eta);
  if (dot(h, h) <= 0.0) {
    return 0.0;
  }
  let wo_h = dot(wo, h);
  let wi_h = dot(wi, h);
  // The microfacet must face the viewer and the two directions must be on
  // opposite sides of it. Testing only the product would admit the back-facing
  // case, which the sampler never produces.
  if (wo_h <= 0.0 || wi_h >= 0.0) {
    return 0.0;
  }
  let denom = wo_h + eta * wi_h;
  if (abs(denom) < 1.0e-9) {
    return 0.0;
  }
  let f = fresnel_dielectric(wo_h, eta);
  let d = ggx_d(alpha, abs(h.z));
  let g2 = smith_g2(alpha, abs(wo.z), abs(wi.z));
  return ((1.0 - f) * d * g2 * abs(wi_h * wo_h)) / (abs(wo.z) * abs(wi.z) * denom * denom);
}

// Solid-angle density of a transmitted direction.
// Jacobian of h -> wi (Walter eq. 17): eta^2 |wi.h| / (wo.h + eta wi.h)^2.
fn dielectric_pdf(alpha: f32, eta: f32, wo: vec3<f32>, wi: vec3<f32>) -> f32 {
  if (wo.z * wi.z >= 0.0) {
    return 0.0;
  }
  let h = transmission_half_vector(wo, wi, eta);
  if (dot(h, h) <= 0.0) {
    return 0.0;
  }
  let wo_h = dot(wo, h);
  let wi_h = dot(wi, h);
  if (wo_h <= 0.0 || wi_h >= 0.0) {
    return 0.0;
  }
  let denom = wo_h + eta * wi_h;
  if (abs(denom) < 1.0e-9) {
    return 0.0;
  }
  let jacobian = abs(eta * eta * wi_h) / (denom * denom);
  return ggx_vndf_half_pdf(alpha, wo, h) * jacobian;
}

fn fresnel_schlick(f0: vec3<f32>, cos_theta: f32) -> vec3<f32> {
  let m = clamp(1.0 - cos_theta, 0.0, 1.0);
  let m2 = m * m;
  return f0 + (vec3<f32>(1.0) - f0) * (m2 * m2 * m);
}

// Exact unpolarised Fresnel for a conductor, from the complex IOR n - ik.
//
// Metals do not follow the dielectric curve Schlick approximates: copper and
// gold dip in the middle of their angular response before climbing to white at
// grazing, and Schlick can only interpolate monotonically from F0 to 1.
fn fresnel_conductor(cos_theta_i: f32, eta: vec3<f32>, k: vec3<f32>) -> vec3<f32> {
  let c = clamp(cos_theta_i, 0.0, 1.0);
  let c2 = c * c;
  let sin2 = 1.0 - c2;

  let eta2 = eta * eta;
  let k2 = k * k;

  let t0 = eta2 - k2 - vec3<f32>(sin2);
  let a2_plus_b2 = sqrt(max(t0 * t0 + 4.0 * eta2 * k2, vec3<f32>(0.0)));
  let t1 = a2_plus_b2 + vec3<f32>(c2);
  let a = sqrt(max(0.5 * (a2_plus_b2 + t0), vec3<f32>(0.0)));
  let t2 = 2.0 * c * a;
  let rs = (t1 - t2) / (t1 + t2);

  let t3 = c2 * a2_plus_b2 + vec3<f32>(sin2 * sin2);
  let t4 = t2 * sin2;
  let rp = rs * (t3 - t4) / (t3 + t4);

  return 0.5 * (rs + rp);
}

fn f0_from_ior(ior: f32) -> f32 {
  let r = (ior - 1.0) / (ior + 1.0);
  return r * r;
}

// Bilinear lookup into the baked directional-albedo table. Must interpolate the
// same way `energy::lookup` does in Rust.
fn ggx_directional_albedo(roughness: f32, cos_theta_o: f32) -> f32 {
  let n = GGX_E_SIZE;
  let fx = clamp(roughness, 0.0, 1.0) * f32(n - 1u);
  // sqrt(cos), matching the parameterisation the table was built with.
  let fy = sqrt(clamp(cos_theta_o, 0.0, 1.0)) * f32(n - 1u);
  let x0 = min(u32(fx), n - 1u);
  let y0 = min(u32(fy), n - 1u);
  let x1 = min(x0 + 1u, n - 1u);
  let y1 = min(y0 + 1u, n - 1u);
  let tx = fx - f32(x0);
  let ty = fy - f32(y0);

  let a = GGX_E[x0 * n + y0];
  let b = GGX_E[x0 * n + y1];
  let c = GGX_E[x1 * n + y0];
  let d = GGX_E[x1 * n + y1];
  let top = a + (b - a) * ty;
  let bot = c + (d - c) * ty;
  return clamp(top + (bot - top) * tx, 1.0e-3, 1.0);
}

// Average Fresnel over the hemisphere, Schlick's closed form.
fn fresnel_average(f0: vec3<f32>) -> vec3<f32> {
  return f0 + (vec3<f32>(1.0) - f0) * (1.0 / 21.0);
}

// Multiple-scattering compensation (Turquin 2019).
//
// A single-scattering G2 discards light that bounces *between* microfacets:
// masked, dropped. On a rough surface most of it is — measured, 68% at
// roughness 1, which the white furnace test shows as an obviously dark object.
// Scaling the lobe by the fraction that went missing restores it, and with
// white Fresnel the directional albedo becomes exactly 1 for every roughness
// and angle.
fn ggx_compensation(roughness: f32, cos_theta_o: f32, f0: vec3<f32>) -> vec3<f32> {
  let e = ggx_directional_albedo(roughness, cos_theta_o);
  return vec3<f32>(1.0) + fresnel_average(f0) * ((1.0 - e) / e);
}

// ---------------------------------------------------------------------------
// The combined surface BSDF
// ---------------------------------------------------------------------------

struct Surface {
  diffuse_albedo: vec3<f32>,
  roughness: f32,
  f0: vec3<f32>,
  eta: vec3<f32>,
  k: vec3<f32>,
  conductor: bool,
  // Fraction of the non-specularly-reflected energy that refracts through.
  transmission: f32,
  // Colour applied to transmitted light. Belongs in Beer-Lambert absorption
  // along the path inside the medium, which needs volumetric tracking; tinting
  // at the interface is exact for a thin surface.
  transmission_tint: vec3<f32>,
  // Relative index n_transmitted / n_incident for the side the ray is on.
  // Resolved here because the shading frame has already been flipped to face
  // the viewer, so wo.z > 0 on both sides and the frame alone cannot say which
  // medium the ray is in.
  relative_ior: f32,
};

fn surface_from_material(m: Material, front_face: bool) -> Surface {
  var s: Surface;
  let base = m.base_color;
  let metallic = clamp(m.metallic, 0.0, 1.0);
  // A metal has no diffuse lobe: the free electrons that make it reflective
  // also absorb whatever enters, so there is no subsurface scattering.
  s.diffuse_albedo = base * (1.0 - metallic);
  s.roughness = clamp(m.roughness, 0.0, 1.0);
  let dielectric_f0 = vec3<f32>(f0_from_ior(max(m.ior, 1.0)));
  s.f0 = mix(dielectric_f0, base, metallic);
  s.eta = m.eta;
  s.k = m.k;
  s.conductor = m.k.x > 0.0 || m.k.y > 0.0 || m.k.z > 0.0;
  // A metal cannot transmit whatever the material says: the same free electrons
  // that make it reflective absorb anything that gets in.
  s.transmission = clamp(m.transmission, 0.0, 1.0) * (1.0 - metallic);
  // White glass is the common case and base_color defaults to black, so an
  // unset colour must not make glass opaque.
  let is_black = base.x <= 0.0 && base.y <= 0.0 && base.z <= 0.0;
  s.transmission_tint = select(base, vec3<f32>(1.0), is_black);
  let ior = max(m.ior, 1.0);
  s.relative_ior = select(1.0 / ior, ior, front_face);
  return s;
}

// Fresnel for the reflection lobe.
//
// A transmissive dielectric uses the exact equations rather than Schlick's fit,
// because its reflection and transmission lobes have to sum to one. Schlick is
// off by up to 0.036, which is invisible on an opaque surface and is energy
// created or destroyed on a transmissive one.
fn surface_reflection_fresnel(s: Surface, cos_theta: f32) -> vec3<f32> {
  if (s.transmission > 0.0 && !s.conductor) {
    return vec3<f32>(fresnel_dielectric(cos_theta, s.relative_ior));
  }
  return surface_fresnel(s, cos_theta);
}

fn surface_fresnel(s: Surface, cos_theta: f32) -> vec3<f32> {
  if (s.conductor) {
    return fresnel_conductor(cos_theta, s.eta, s.k);
  }
  return fresnel_schlick(s.f0, cos_theta);
}

fn surface_normal_reflectance(s: Surface) -> vec3<f32> {
  if (s.conductor) {
    return fresnel_conductor(1.0, s.eta, s.k);
  }
  return s.f0;
}

// Fraction of incoming energy the specular layer reflects, seen from wo.
//
// The lobes are not independent: the specular layer sits over the diffuse base,
// so light it reflects never reaches the base. Adding them without this lets a
// surface reflect more than arrived — measured, a smooth blue plastic at
// grazing returned 1.40 in blue. Singly-scattered energy is weighted by the
// Fresnel at this angle, the multiply-scattered remainder by the hemispherical
// average, because it was filtered by Fresnel several times on the way out.
fn surface_specular_albedo(s: Surface, cos_theta_o: f32) -> vec3<f32> {
  let mu = abs(cos_theta_o);
  let e_ss = ggx_directional_albedo(s.roughness, mu);
  let f = surface_fresnel(s, mu);
  let f_avg = fresnel_average(surface_normal_reflectance(s));
  return clamp(f * e_ss + f_avg * (1.0 - e_ss), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Probability of choosing the specular lobe. Fresnel-weighted, so the split
// tracks where the energy is; clamped so a lobe that can still contribute keeps
// a non-zero chance of being sampled.
fn surface_specular_probability(s: Surface, cos_theta_o: f32) -> f32 {
  if (s.transmission > 0.0) {
    // For a transmissive dielectric the split is exactly Fresnel: what is not
    // reflected goes through. The opaque path's albedo-ratio heuristic would
    // put most samples in the reflection lobe of a nearly transparent material.
    let f = fresnel_dielectric(abs(cos_theta_o), s.relative_ior);
    // Past the critical angle everything reflects, and the clamp must not
    // apply: reserving 5% of samples for a lobe that cannot produce a direction
    // leaves them to the TIR fallback, which the pdf does not describe.
    if (f >= 1.0) {
      return 1.0;
    }
    return clamp(f, 0.05, 0.95);
  }
  let spec_albedo = surface_specular_albedo(s, cos_theta_o);
  let spec = luminance(spec_albedo);
  let diff = luminance(s.diffuse_albedo * (vec3<f32>(1.0) - spec_albedo));
  if (spec + diff <= 0.0) {
    return 1.0;
  }
  return clamp(spec / (spec + diff), 0.1, 0.9);
}

// Evaluate the full BSDF. Returns f, not f * cos.
fn surface_eval(s: Surface, wo: vec3<f32>, wi: vec3<f32>) -> vec3<f32> {
  if (wo.z <= 0.0) {
    return vec3<f32>(0.0);
  }
  let alpha = roughness_to_alpha(s.roughness);

  // wi.z < 0 is the transmitted hemisphere, which only a material with non-zero
  // transmission can reach.
  if (wi.z < 0.0) {
    if (s.transmission <= 0.0) {
      return vec3<f32>(0.0);
    }
    let t = dielectric_eval(alpha, s.relative_ior, wo, wi);
    return vec3<f32>(t) * s.transmission * s.transmission_tint;
  }
  if (wi.z <= 0.0) {
    return vec3<f32>(0.0);
  }

  // The diffuse base only sees what the specular layer let through, and a
  // transmissive material has no diffuse base to speak of.
  let transmitted = vec3<f32>(1.0) - surface_specular_albedo(s, wo.z);
  var f = s.diffuse_albedo * INV_PI * transmitted * (1.0 - s.transmission);

  let h_raw = wo + wi;
  if (dot(h_raw, h_raw) > 1.0e-12) {
    let h = normalize(h_raw);
    let d = ggx_d(alpha, h.z);
    let g2 = smith_g2(alpha, wo.z, wi.z);
    let fr = surface_reflection_fresnel(s, max(dot(wo, h), 0.0));
    let single = fr * (d * g2 / (4.0 * wo.z * wi.z));
    // Energy compensation is a reflection-only correction: it adds back the
    // multiple scattering between microfacets that single-scattering Smith
    // drops. A transmissive surface loses that energy to the transmission lobe
    // instead, so applying it here would create light.
    if (s.transmission > 0.0) {
      f = f + single;
    } else {
      f = f + single * ggx_compensation(s.roughness, wo.z, surface_normal_reflectance(s));
    }
  }
  return f;
}

// Combined solid-angle density. Must be the *sum* over lobes, weighted by
// selection probability — not the density of whichever lobe was chosen. A
// direction reachable by either lobe really is generated with the combined
// probability, so using only one biases the estimator, and MIS at build step 9
// needs the true density of the strategy.
fn surface_pdf(s: Surface, wo: vec3<f32>, wi: vec3<f32>) -> f32 {
  if (wo.z <= 0.0) {
    return 0.0;
  }
  let p_spec = surface_specular_probability(s, wo.z);
  let alpha = roughness_to_alpha(s.roughness);
  let rest = 1.0 - p_spec;

  if (wi.z < 0.0) {
    if (s.transmission <= 0.0) {
      return 0.0;
    }
    return rest * s.transmission * dielectric_pdf(alpha, s.relative_ior, wo, wi);
  }
  if (wi.z <= 0.0) {
    return 0.0;
  }

  var spec_pdf = 0.0;
  let h_raw = wo + wi;
  if (dot(h_raw, h_raw) > 1.0e-12) {
    spec_pdf = ggx_vndf_pdf(alpha, wo, normalize(h_raw));
  }
  let diff_pdf = wi.z * INV_PI;
  return p_spec * spec_pdf + rest * (1.0 - s.transmission) * diff_pdf;
}

struct BsdfSample {
  wi: vec3<f32>,
  weight: vec3<f32>,
  pdf: f32,
  valid: bool,
};

fn reflect_about(v: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
  return -v + 2.0 * dot(v, n) * n;
}

fn surface_sample(s: Surface, wo: vec3<f32>, u_lobe: f32, u: vec2<f32>) -> BsdfSample {
  var out: BsdfSample;
  out.valid = false;
  out.pdf = 0.0;
  out.weight = vec3<f32>(0.0);
  out.wi = vec3<f32>(0.0, 0.0, 1.0);

  if (wo.z <= 0.0) {
    return out;
  }
  let p_spec = surface_specular_probability(s, wo.z);
  let alpha = roughness_to_alpha(s.roughness);

  // A sample that misses the hemisphere its lobe implies is dropped rather than
  // reinterpreted: on a rough surface at a grazing microfacet, `reflect_about`
  // can return a direction below the surface and a refraction one above it, and
  // letting those through asks the other lobe's pdf to describe a sample it
  // could never have produced.
  var wi: vec3<f32>;
  if (u_lobe < p_spec) {
    let m = sample_ggx_vndf(wo, alpha, u);
    wi = reflect_about(wo, m);
    if (wi.z <= 0.0) {
      return out;
    }
  } else if (u_lobe < p_spec + (1.0 - p_spec) * s.transmission) {
    // Refract about the *same* visible microfacet distribution the reflection
    // lobe samples, which is what makes the two lobes two halves of one
    // interface rather than two unrelated surfaces.
    let m = sample_ggx_vndf(wo, alpha, u);
    let r = refract_about(wo, m, s.relative_ior);
    // Total internal reflection at this microfacet. Dropped rather than
    // redirected into the reflection lobe: redirected samples would land in the
    // reflected hemisphere with a density `surface_pdf` has no term for. Past
    // the critical angle this is unreachable anyway, because
    // `surface_specular_probability` returns 1 there.
    if (!r.valid || r.wt.z >= 0.0) {
      return out;
    }
    wi = r.wt;
  } else {
    wi = sample_cosine_hemisphere(u);
    if (wi.z <= 0.0) {
      return out;
    }
  }

  // Evaluate the whole BSDF and the combined pdf regardless of which lobe
  // produced the direction — that is what makes the estimator unbiased when the
  // lobes overlap.
  let f = surface_eval(s, wo, wi);
  let pdf = surface_pdf(s, wo, wi);
  if (pdf <= 0.0) {
    return out;
  }
  out.wi = wi;
  out.pdf = pdf;
  // |cos|, not cos: the projected-solid-angle factor is positive on both sides.
  // A transmitted direction has wi.z < 0 and would otherwise carry a negative
  // weight, subtracting light from the image.
  out.weight = f * (abs(wi.z) / pdf);
  out.valid = true;
  return out;
}
