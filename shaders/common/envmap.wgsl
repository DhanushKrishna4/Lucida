// Environment lighting. Mirrors `crates/core/src/envmap.rs`.
//
// # Why textures and not storage buffers
//
// Forced, not preferred. The wavefront's SHADE stage already binds exactly eight
// storage buffers, which is WebGPU's *guaranteed* per-stage limit counted across
// every bind group — so there is no room for the map, the conditional CDF or the
// marginal. Sampled textures come from a separate budget (sixteen guaranteed),
// and two of them carry everything this needs.
//
// # Why textureLoad and never textureSample
//
// `textureLoad` reads one texel with integer coordinates and no filtering. That
// is the correct lookup here, not merely the convenient one: the CDF describes a
// **piecewise-constant** image, so a bilinearly filtered radiance would not be
// the quantity the density is proportional to, and the estimator would pick up a
// bias no energy test would catch. It also keeps the GPU bit-identical to the
// CPU, since filtering is implementation-defined and `textureLoad` is not.
//
// The `sin(theta)` weight, the two-dimensional inversion and the solid-angle
// Jacobian are all explained in the Rust module; only the parts that differ are
// commented again here.

// Radiance per texel, rgba32float. Bound even when the scene has no environment
// map — a 1x1 black texture stands in, because WebGPU has no optional bindings
// and a branch is cheaper than a second pipeline.
@group(0) @binding(90) var env_radiance_tex: texture_2d<f32>;
// (width + 1) x (height + 1), r32float. Rows 0..height-1 hold each row's
// conditional CDF over columns; row `height` holds the marginal CDF over rows.
// Packed into one texture rather than two because a 1-D CDF would need either a
// storage buffer (none available) or a 1-D texture binding of its own.
@group(0) @binding(91) var env_cdf_tex: texture_2d<f32>;

// math.wgsl defines PI and INV_PI but not this one.
const TAU: f32 = 6.28318530717958647692;

// Shadow-ray length for an environment connection. Large enough that nothing in
// any scene lies beyond it, finite because the shadow test shortens the distance
// by a relative epsilon and `inf * (1 - eps)` is fine but `inf - inf` in a slab
// test is not.
//
// Defined here rather than in light.wgsl because the wavefront's SHADE stage
// reimplements the connection inline and does not include that file.
const ENV_DISTANCE: f32 = 1.0e30;

struct EnvSample {
  direction: vec3<f32>,
  radiance: vec3<f32>,
  pdf: f32,
  valid: bool,
};

fn env_present() -> bool {
  return U.env_width > 0u && U.env_total_weight > 0.0;
}

fn env_direction_from_uv(u: f32, v: f32) -> vec3<f32> {
  let theta = v * PI;
  let phi = u * TAU;
  let sin_t = sin(theta);
  return vec3<f32>(sin_t * cos(phi), cos(theta), sin_t * sin(phi));
}

fn env_uv_from_direction(d: vec3<f32>) -> vec2<f32> {
  let theta = acos(clamp(d.y, -1.0, 1.0));
  // atan2 returns (-pi, pi]; shift into [0, tau) so it maps onto [0, 1).
  var phi = atan2(d.z, d.x);
  if (phi < 0.0) {
    phi = phi + TAU;
  }
  return vec2<f32>(phi / TAU, theta / PI);
}

fn env_texel_of(uv: vec2<f32>) -> vec2<u32> {
  let w = i32(U.env_width);
  let h = i32(U.env_height);
  let i = clamp(i32(uv.x * f32(U.env_width)), 0, w - 1);
  let j = clamp(i32(uv.y * f32(U.env_height)), 0, h - 1);
  return vec2<u32>(u32(i), u32(j));
}

fn env_radiance_at(direction: vec3<f32>) -> vec3<f32> {
  if (!env_present()) {
    return vec3<f32>(0.0);
  }
  let t = env_texel_of(env_uv_from_direction(direction));
  return textureLoad(env_radiance_tex, vec2<i32>(t), 0).rgb;
}

// Sine at the row's *centre*, matching the weight the CDF was built from. Using
// the row's upper edge would put a zero at the pole row and make the topmost
// band of sky unsampleable.
fn env_row_sin_theta(j: u32) -> f32 {
  return sin((f32(j) + 0.5) / f32(U.env_height) * PI);
}

fn env_weight_at(i: u32, j: u32) -> f32 {
  let rgb = textureLoad(env_radiance_tex, vec2<i32>(i32(i), i32(j)), 0).rgb;
  return max(luminance(rgb), 0.0) * env_row_sin_theta(j);
}

fn env_pdf(direction: vec3<f32>) -> f32 {
  if (!env_present()) {
    return 0.0;
  }
  let t = env_texel_of(env_uv_from_direction(direction));
  let sin_theta = env_row_sin_theta(t.y);
  if (sin_theta <= 0.0) {
    return 0.0;
  }
  let weight = env_weight_at(t.x, t.y);
  let texels = f32(U.env_width * U.env_height);
  // Density over the unit square, then into solid angle: dw = 2 pi^2 sin(theta) du dv.
  let pdf_uv = weight / U.env_total_weight * texels;
  return pdf_uv / (2.0 * PI * PI * sin_theta);
}

// One CDF cell lookup. `row` selects the texture row; the CDF runs along x.
fn env_cdf_at(row: u32, k: u32) -> f32 {
  return textureLoad(env_cdf_tex, vec2<i32>(i32(k), i32(row)), 0).r;
}

struct CdfHit {
  cell: u32,
  frac: f32,
};

// Invert one CDF: find the cell containing `x` and where inside it.
//
// Binary search rather than a linear scan: 11 iterations for a 2048-wide map
// against up to 2048. Written so both branches do the same amount of work, which
// is what keeps a warp from serialising — the loop trip count depends only on
// `n`, which is uniform across the workgroup.
fn env_sample_cdf(row: u32, n: u32, x: f32) -> CdfHit {
  var lo = 0u;
  var len = n;
  loop {
    if (len == 0u) {
      break;
    }
    let half = len / 2u;
    let mid = lo + half;
    if (env_cdf_at(row, mid) <= x) {
      lo = mid + 1u;
      len = len - (half + 1u);
    } else {
      len = half;
    }
  }
  var out: CdfHit;
  out.cell = min(select(lo - 1u, 0u, lo == 0u), n - 1u);

  let c0 = env_cdf_at(row, out.cell);
  let c1 = env_cdf_at(row, out.cell + 1u);
  let span = c1 - c0;
  // A zero-width cell carries no probability and can only be reached through
  // rounding; placing the sample at its start is as good as anywhere.
  out.frac = select(0.0, clamp((x - c0) / span, 0.0, 1.0), span > 0.0);
  return out;
}

fn env_sample(u: vec2<f32>) -> EnvSample {
  var out: EnvSample;
  out.direction = vec3<f32>(0.0, 1.0, 0.0);
  out.radiance = vec3<f32>(0.0);
  out.pdf = 0.0;
  out.valid = false;
  if (!env_present()) {
    return out;
  }

  let w = U.env_width;
  let h = U.env_height;

  // Row from the marginal (stored in row `h`), then column from that row's
  // conditional.
  let row_hit = env_sample_cdf(h, h, u.y);
  let j = row_hit.cell;
  let col_hit = env_sample_cdf(j, w, u.x);
  let i = col_hit.cell;

  let uv = vec2<f32>(
    (f32(i) + col_hit.frac) / f32(w),
    (f32(j) + row_hit.frac) / f32(h),
  );

  let sin_theta = env_row_sin_theta(j);
  if (sin_theta <= 0.0) {
    return out;
  }
  let weight = env_weight_at(i, j);
  let pdf_uv = weight / U.env_total_weight * f32(w * h);
  let pdf = pdf_uv / (2.0 * PI * PI * sin_theta);
  if (pdf <= 0.0) {
    return out;
  }

  out.direction = env_direction_from_uv(uv.x, uv.y);
  out.radiance = textureLoad(env_radiance_tex, vec2<i32>(i32(i), i32(j)), 0).rgb;
  out.pdf = pdf;
  out.valid = true;
  return out;
}

// How many strategies the light sampler chooses between: the area lights plus
// the environment map, when there is one.
//
// Every light-sampling density is divided by this, and so is the density the
// BSDF side uses for its MIS weight. If the two disagree the weights stop
// summing to one and the image is uniformly wrong by a factor nobody can see.
fn light_strategy_count() -> u32 {
  return U.num_lights + select(0u, 1u, env_present());
}
