// Area lights and next event estimation. Mirrors `crates/core/src/light.rs`.
//
// # The measure conversion
//
// Light sampling picks a **point** on an emitter, so its density is naturally
// with respect to **area**: pick a light with probability p_select, then a point
// uniformly over its surface with density 1/area.
//
// But the transport integral is over **solid angle**, and so is the BSDF's pdf.
// Adding an area-measure density to a solid-angle one produces an image that
// looks entirely plausible and is wrong by a factor that varies with distance.
//
// A patch of area projects to solid angle as
//
//   dw = dA * |cos(theta')| / d^2
//
// where theta' is the angle at the *light* between its normal and the direction
// back to the shading point. Densities transform by the inverse Jacobian:
//
//   p_w(w) = p_A(x') * d^2 / |cos(theta')|
//
// The pdf therefore grows with distance, correctly down-weighting distant
// lights, and diverges as the emitter is seen edge-on — the classic NEE firefly,
// handled by rejecting those samples rather than dividing by a near-zero.
//
// Full derivation in crates/core/src/light.rs.

// Below this the emitter is edge-on and the pdf diverges. Such samples carry
// essentially no energy, so discarding them costs nothing.
const MIN_COS_LIGHT: f32 = 1.0e-6;


const SAMPLING_BSDF_ONLY: u32 = 0u;
const SAMPLING_NEE_ONLY: u32 = 1u;
const SAMPLING_MIS: u32 = 2u;

struct LightSample {
  wi: vec3<f32>,
  distance: f32,
  radiance: vec3<f32>,
  pdf: f32,          // solid-angle density at the shading point
  valid: bool,
};

// Uniformly sample a point on a light's surface.
fn sample_light_point(light: Light, u: vec2<f32>) -> vec3<f32> {
  if (light.kind == LIGHT_KIND_TRIANGLE) {
    // The square root warps the unit square onto the triangle with constant
    // Jacobian. Without it samples pile up toward one corner, biasing the
    // estimate toward whatever that corner happens to illuminate.
    let su = sqrt(u.x);
    let b1 = 1.0 - su;
    let b2 = u.y * su;
    return light.origin + b1 * light.edge_u + b2 * light.edge_v;
  }
  // Uniform on a parallelogram is the unit square, unwarped.
  return light.origin + u.x * light.edge_u + u.y * light.edge_v;
}

// Select a light uniformly and sample a point on it.
fn sample_lights(shading_point: vec3<f32>, u_select: f32, u_area: vec2<f32>) -> LightSample {
  var out: LightSample;
  out.valid = false;
  out.pdf = 0.0;

  if (U.num_lights == 0u) {
    return out;
  }
  let n = U.num_lights;
  let index = min(u32(u_select * f32(n)), n - 1u);
  let light = lights[index];

  let point = sample_light_point(light, u_area);
  let to_light = point - shading_point;
  let dist_sq = dot(to_light, to_light);
  if (dist_sq <= 0.0) {
    return out;
  }
  let distance = sqrt(dist_sq);
  let wi = to_light / distance;

  // Emission is one-sided: the emitter radiates from the face its normal points
  // from.
  let cos_light = dot(light.normal, -wi);
  if (cos_light <= MIN_COS_LIGHT) {
    return out;
  }

  // Area measure -> solid angle. This is the line where energy errors live.
  let pdf_area = 1.0 / (f32(n) * light.area);
  let pdf = pdf_area * dist_sq / cos_light;
  if (pdf <= 0.0) {
    return out;
  }

  out.wi = wi;
  out.distance = distance;
  out.radiance = materials[light.material].emissive;
  out.pdf = pdf;
  out.valid = true;
  return out;
}

// Solid-angle density the light sampler *would have* assigned to a direction
// that landed on an emitter, from the geometry at the hit alone.
//
// This is what MIS needs from the BSDF side: having walked into a light, how
// likely was the other strategy to have found the same direction?
//
// `area == 0` means "not sampleable as a light" and returns zero — an emissive
// sphere is the case, since build_lights does not include spheres. Returning a
// spurious density there would silently darken them.
fn light_pdf_from_geometry(
  area: f32,
  light_normal: vec3<f32>,
  shading_point: vec3<f32>,
  hit_point: vec3<f32>,
) -> f32 {
  if (U.num_lights == 0u || area <= 0.0) {
    return 0.0;
  }
  let to_light = hit_point - shading_point;
  let dist_sq = dot(to_light, to_light);
  if (dist_sq <= 0.0) {
    return 0.0;
  }
  let wi = to_light / sqrt(dist_sq);
  let cos_light = dot(light_normal, -wi);
  if (cos_light <= MIN_COS_LIGHT) {
    return 0.0;
  }
  let pdf_area = 1.0 / (f32(U.num_lights) * area);
  return pdf_area * dist_sq / cos_light;
}

// The power heuristic with beta = 2 (Veach 1995).
//
//   w_a = p_a^2 / (p_a^2 + p_b^2)
//
// Decides how much credit each strategy gets for a direction both could have
// produced. The weights sum to 1 for any pair, which is what keeps the combined
// estimator unbiased — no path counted twice, none dropped.
//
// Squaring is what makes it better than the balance heuristic: it pushes weight
// harder toward whichever strategy sampled the direction densely, suppressing
// the low-probability samples that become fireflies.
//
// Written as 1 / (1 + (p_b/p_a)^2) rather than squaring both: algebraically
// identical, and it degrades to exactly 0 or 1 rather than to NaN however
// extreme the pair becomes.
fn power_heuristic(pdf_a: f32, pdf_b: f32) -> f32 {
  if (pdf_a <= 0.0) {
    return 0.0;
  }
  let r = pdf_b / pdf_a;
  return 1.0 / (1.0 + r * r);
}

// Any-hit along a shadow ray. Returns true when something blocks the path.
//
// Shortened relative to the distance, not by a fixed epsilon, so the emitter
// does not count as its own occluder at any scene scale — float spacing grows
// with magnitude, so a fixed epsilon is both too small far away and too large
// up close.
fn shadow_ray_blocked(origin: vec3<f32>, wi: vec3<f32>, distance: f32) -> bool {
  let t_max = distance * (1.0 - 1.0e-3);

  for (var i: u32 = 0u; i < U.num_primitives; i = i + 1u) {
    let prim = primitives[i];
    var t: f32;
    if (prim.kind == PRIM_KIND_QUAD) {
      t = intersect_quad(prim, Ray(origin, wi), T_MIN, t_max);
    } else {
      t = intersect_sphere(prim, Ray(origin, wi), T_MIN, t_max);
    }
    if (t > 0.0) {
      return true;
    }
  }

  // Any-hit through the BVH: return on the first blocker rather than finding the
  // closest, and never narrow t_max — a shadow ray only needs to know *whether*
  // something is in the way.
  let tri = intersect_triangles(origin, wi, T_MIN, t_max);
  return tri.valid;
}

// One light-sampling connection from a shading point.
//
//   L = f(wo, wi) * cos(theta_i) * L_e * V / p_w(wi)
//
// Random numbers are drawn unconditionally, even when the sample turns out to
// contribute nothing, so the stream stays aligned with the CPU reference.
fn direct_light(
  surf: Surface,
  position: vec3<f32>,
  normal: vec3<f32>,
  geometric_normal: vec3<f32>,
  wo: vec3<f32>,
  rng: ptr<function, Rng>,
  mode: u32,
) -> vec3<f32> {
  let u_select = rng_next_f32(rng);
  let u_area = rng_next_vec2(rng);

  // The environment map is one strategy among the area lights, chosen with the
  // same uniform draw, so every density below carries a 1/total selection
  // factor — including the one the BSDF side uses for its MIS weight.
  let n_lights = U.num_lights;
  let total = light_strategy_count();
  if (total == 0u) {
    return vec3<f32>(0.0);
  }
  let pick = min(u32(u_select * f32(total)), total - 1u);

  var wi_world: vec3<f32>;
  var radiance_in: vec3<f32>;
  var distance: f32;
  var pdf: f32;
  if (pick < n_lights) {
    // Select this light exactly. The fraction within the cell is not needed for
    // the choice — the point on the light comes from u_area — so handing
    // sample_lights the cell centre wastes no randomness.
    let u_light = (f32(pick) + 0.5) / f32(n_lights);
    let ls = sample_lights(position, u_light, u_area);
    if (!ls.valid) {
      return vec3<f32>(0.0);
    }
    wi_world = ls.wi;
    radiance_in = ls.radiance;
    distance = ls.distance;
    // sample_lights already divided by the light count; rescale to the full
    // strategy count.
    pdf = ls.pdf * (f32(n_lights) / f32(total));
  } else {
    let es = env_sample(u_area);
    if (!es.valid) {
      return vec3<f32>(0.0);
    }
    wi_world = es.direction;
    radiance_in = es.radiance;
    // Nothing occludes the sky from beyond itself, so the shadow ray runs to
    // infinity and any hit at all blocks it.
    distance = ENV_DISTANCE;
    pdf = es.pdf / f32(total);
  }
  if (pdf <= 0.0) {
    return vec3<f32>(0.0);
  }

  let wi = to_local(wi_world, normal);
  // Below the shading hemisphere is not automatically a rejection: a
  // transmissive surface can be lit from behind, and that is most of what
  // makes glass look like glass. It also has to be allowed for MIS to stay
  // unbiased, because the BSDF strategy can reach a light through the surface
  // and discounts itself assuming light sampling could find the same
  // direction.
  if (wi.z == 0.0 || (wi.z < 0.0 && surf.transmission <= 0.0)) {
    return vec3<f32>(0.0);
  }
  let f = surface_eval(surf, wo, wi);
  if (max(f.x, max(f.y, f.z)) <= 0.0) {
    return vec3<f32>(0.0);
  }

  // Under MIS the BSDF strategy could also have produced this direction, so the
  // two share the credit. Both densities are in solid angle at this shading
  // point — mixing measures here is the classic way to get an image that is
  // subtly wrong everywhere and obviously wrong nowhere.
  var mis_weight = 1.0;
  if (mode == SAMPLING_MIS) {
    mis_weight = power_heuristic(pdf, surface_pdf(surf, wo, wi));
  }
  if (mis_weight <= 0.0) {
    return vec3<f32>(0.0);
  }

  // The shadow ray goes last: it is by far the most expensive part, and
  // everything above can reject the sample for free.
  // Toward whichever side the shadow ray leaves on, or a transmissive surface
  // becomes its own occluder.
  let shadow_normal = select(geometric_normal, -geometric_normal, wi.z < 0.0);
  let origin = offset_ray_origin(position, shadow_normal);
  if (shadow_ray_blocked(origin, wi_world, distance)) {
    return vec3<f32>(0.0);
  }

  // abs(cos): the projected-solid-angle factor is positive on both sides.
  return f * abs(wi.z) * radiance_in * (mis_weight / pdf);
}
