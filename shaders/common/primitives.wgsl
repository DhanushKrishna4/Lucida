// Ray/primitive intersection and the Hit record. Mirrors
// `crates/core/src/scene.rs`.
//
// Split out from scene.wgsl so a stage that only needs occlusion against
// analytic primitives can include it without also pulling in the vertex
// attribute binding that triangle *shading* requires. WebGPU guarantees only
// eight storage buffers per shader stage, so a binding a kernel never reads
// still costs it one of the eight.
//
// Brute force over every primitive: no acceleration structure yet (that arrives
// at build step 5, validated against exactly this code).
//
// Requires: generated.wgsl (struct declarations), math.wgsl (T_MIN / T_MAX),
// and the bindings declared by the including shader.

struct Hit {
  t: f32,
  // BVH nodes visited finding this hit, for the traversal heatmap.
  steps: u32,
  position: vec3<f32>,
  // Shading normal, oriented against the incoming ray. For a smooth mesh this
  // is the barycentric blend of the vertex normals.
  normal: vec3<f32>,
  // Facet normal, oriented against the ray. Ray offsetting must use this one:
  // offsetting along an interpolated normal can push the new origin *below* the
  // actual surface at a grazing angle, reintroducing the self-intersection the
  // offset exists to prevent.
  geometric_normal: vec3<f32>,
  front_face: bool,
  material: u32,
  // Surface area of the primitive hit, when the light sampler could have chosen
  // it; **zero otherwise**. Multiple importance sampling needs it to ask how
  // likely light sampling was to find the same point. Zero for spheres, which
  // are not in the light list, so BSDF sampling correctly takes full credit.
  light_area: f32,
  valid: bool,
};

// Solves |o + t*d - c|^2 = r^2 using the half-b form: `h = dot(oc, d)` and
// discriminant `h*h - a*c`. One multiply fewer than the textbook quadratic, and
// it avoids the factor-of-4 growth in the discriminant.
// Returns t, or -1.0 for a miss.
fn intersect_sphere(s: Primitive, ray: Ray, t_min: f32, t_max: f32) -> f32 {
  let oc = ray.origin - s.position;
  let a = dot(ray.dir, ray.dir);
  let h = dot(oc, ray.dir);
  let c = dot(oc, oc) - s.radius * s.radius;
  let disc = h * h - a * c;
  if (disc < 0.0) {
    return -1.0;
  }
  let sqrt_d = sqrt(disc);
  // Near root first; fall back to the far root when the origin is inside.
  var t = (-h - sqrt_d) / a;
  if (t < t_min || t > t_max) {
    t = (-h + sqrt_d) / a;
    if (t < t_min || t > t_max) {
      return -1.0;
    }
  }
  return t;
}

// Ray/parallelogram. Plane test, then an inside test in the (edge_u, edge_v)
// parameterisation using the reciprocal-basis trick; see the derivation in
// crates/core/src/scene.rs.
// Returns t, or -1.0 for a miss.
fn intersect_quad(q: Primitive, ray: Ray, t_min: f32, t_max: f32) -> f32 {
  let denom = dot(q.normal, ray.dir);
  if (abs(denom) < 1.0e-8) {
    return -1.0;
  }
  let t = (dot(q.normal, q.position) - dot(q.normal, ray.origin)) / denom;
  if (t < t_min || t > t_max) {
    return -1.0;
  }
  let p = ray.origin + t * ray.dir;
  let d = p - q.position;
  let n_raw = cross(q.edge_u, q.edge_v);
  let w = n_raw / dot(n_raw, n_raw);
  let a = dot(w, cross(d, q.edge_v));
  let b = dot(w, cross(q.edge_u, d));
  if (a < 0.0 || a > 1.0 || b < 0.0 || b > 1.0) {
    return -1.0;
  }
  return t;
}

