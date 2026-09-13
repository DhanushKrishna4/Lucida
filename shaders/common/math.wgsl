// Shared math. Every function here mirrors `crates/core/src/math.rs` exactly.
// Where the two could diverge on an edge case, the WGSL is written to match the
// Rust rather than to be idiomatic — see `onb_sign` below.

const PI: f32 = 3.14159265358979323846;
const INV_PI: f32 = 0.31830988618379067154;

const T_MIN: f32 = 1.0e-4;
const T_MAX: f32 = 1.0e30;

// Sign of a float *by its sign bit*, matching Rust's `f32::copysign`.
//
// The obvious `select(-1.0, 1.0, z >= 0.0)` disagrees with Rust for negative
// zero: IEEE says `-0.0 >= 0.0` is true, so that form returns +1, while
// `copysign(1.0, -0.0)` returns -1. Negative zero is not hypothetical here —
// flipping an axis-aligned normal to face the ray produces components of exactly
// -0.0, and a mismatched basis sign would rotate the sampled hemisphere
// differently on the two devices and break the CPU/GPU comparison for every
// axis-aligned surface in the scene (that is, every wall of the Cornell box).
fn onb_sign(z: f32) -> f32 {
  return select(-1.0, 1.0, (bitcast<u32>(z) & 0x80000000u) == 0u);
}

struct Onb {
  t: vec3<f32>,
  b: vec3<f32>,
};

// Duff et al. 2017, "Building an Orthonormal Basis, Revisited".
fn onb(n: vec3<f32>) -> Onb {
  let sg = onb_sign(n.z);
  let a = -1.0 / (sg + n.z);
  let b = n.x * n.y * a;
  var o: Onb;
  o.t = vec3<f32>(1.0 + sg * n.x * n.x * a, sg * b, -sg * n.x);
  o.b = vec3<f32>(b, sg + n.y * n.y * a, -n.y);
  return o;
}

fn to_world(local: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
  let o = onb(n);
  return o.t * local.x + o.b * local.y + n * local.z;
}

// Inverse of to_world. The basis is orthonormal, so the inverse is the
// transpose: three dot products, no matrix inverse.
fn to_local(world: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
  let o = onb(n);
  return vec3<f32>(dot(world, o.t), dot(world, o.b), dot(world, n));
}

// Shirley-Chiu concentric mapping from the unit square to the unit disk.
fn concentric_sample_disk(u: vec2<f32>) -> vec2<f32> {
  let o = 2.0 * u - vec2<f32>(1.0, 1.0);
  if (o.x == 0.0 && o.y == 0.0) {
    return vec2<f32>(0.0, 0.0);
  }
  var r: f32;
  var theta: f32;
  if (abs(o.x) > abs(o.y)) {
    r = o.x;
    theta = (PI / 4.0) * (o.y / o.x);
  } else {
    r = o.y;
    theta = PI / 2.0 - (PI / 4.0) * (o.x / o.y);
  }
  return r * vec2<f32>(cos(theta), sin(theta));
}

// Cosine-weighted hemisphere sample around +Z, via Malley's method.
// pdf(w) = cos(theta) / pi, with respect to solid angle. See the derivation in
// crates/core/src/math.rs.
fn sample_cosine_hemisphere(u: vec2<f32>) -> vec3<f32> {
  let d = concentric_sample_disk(u);
  let z = sqrt(max(0.0, 1.0 - d.x * d.x - d.y * d.y));
  return vec3<f32>(d.x, d.y, z);
}

fn cosine_hemisphere_pdf(cos_theta: f32) -> f32 {
  return max(0.0, cos_theta * INV_PI);
}

// Wachter & Binder robust ray-origin offset (Ray Tracing Gems ch. 6).
// Offsets by a fixed number of ULP rather than a fixed distance, so it holds up
// across the whole range of scene scales. See crates/core/src/math.rs.
const OFFSET_ORIGIN: f32 = 0.03125;      // 1/32
const OFFSET_FLOAT_SCALE: f32 = 0.0000152587890625; // 1/65536
const OFFSET_INT_SCALE: f32 = 256.0;

fn offset_ray_origin(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
  let of_i = vec3<i32>(
    i32(OFFSET_INT_SCALE * n.x),
    i32(OFFSET_INT_SCALE * n.y),
    i32(OFFSET_INT_SCALE * n.z),
  );
  let p_i = vec3<f32>(
    bitcast<f32>(bitcast<i32>(p.x) + select(of_i.x, -of_i.x, p.x < 0.0)),
    bitcast<f32>(bitcast<i32>(p.y) + select(of_i.y, -of_i.y, p.y < 0.0)),
    bitcast<f32>(bitcast<i32>(p.z) + select(of_i.z, -of_i.z, p.z < 0.0)),
  );
  return vec3<f32>(
    select(p_i.x, p.x + OFFSET_FLOAT_SCALE * n.x, abs(p.x) < OFFSET_ORIGIN),
    select(p_i.y, p.y + OFFSET_FLOAT_SCALE * n.y, abs(p.y) < OFFSET_ORIGIN),
    select(p_i.z, p.z + OFFSET_FLOAT_SCALE * n.z, abs(p.z) < OFFSET_ORIGIN),
  );
}

fn luminance(c: vec3<f32>) -> f32 {
  return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}
