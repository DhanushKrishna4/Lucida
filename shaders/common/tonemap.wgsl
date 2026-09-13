// Display transforms. Mirrors `crates/core/src/tonemap.rs` exactly.
//
// Contract: every operator takes scene-linear HDR and returns **display-linear**
// values in [0, 1]. The sRGB transfer function is applied once, afterwards, by
// the caller. Several published ACES/AgX snippets end in display-*encoded* space
// instead; applying sRGB on top of those double-encodes and washes out shadows.
// Where a fit ends encoded (AgX does), it is linearised here.
//
// `crates/gpu/tests/tonemap_agreement.rs` runs these on the GPU and compares
// against the Rust implementations value for value.

const TONEMAP_CLAMP: u32 = 0u;
const TONEMAP_REINHARD: u32 = 1u;
const TONEMAP_ACES: u32 = 2u;
const TONEMAP_AGX: u32 = 3u;

fn tonemap_reinhard(c: vec3<f32>) -> vec3<f32> {
  return max(c / (vec3<f32>(1.0) + c), vec3<f32>(0.0));
}

// ACES, via Stephen Hill's fit of the RRT + sRGB ODT. Chosen over Narkowicz's
// one-liner because that fit targets already-encoded output, making it ambiguous
// whether a gamma step belongs after it. Matrices are column-major, matching
// glam's `from_cols_array`.
const ACES_INPUT = mat3x3<f32>(
  vec3<f32>(0.59719, 0.07600, 0.02840),
  vec3<f32>(0.35458, 0.90834, 0.13383),
  vec3<f32>(0.04823, 0.01566, 0.83777),
);
const ACES_OUTPUT = mat3x3<f32>(
  vec3<f32>( 1.60475, -0.10208, -0.00327),
  vec3<f32>(-0.53108,  1.10813, -0.07276),
  vec3<f32>(-0.07367, -0.00605,  1.07602),
);

fn rrt_and_odt_fit(v: vec3<f32>) -> vec3<f32> {
  let a = v * (v + 0.0245786) - vec3<f32>(0.000090537);
  let b = v * (0.983729 * v + vec3<f32>(0.4329510)) + vec3<f32>(0.238081);
  return a / b;
}

fn tonemap_aces(c: vec3<f32>) -> vec3<f32> {
  let v = rrt_and_odt_fit(ACES_INPUT * max(c, vec3<f32>(0.0)));
  return clamp(ACES_OUTPUT * v, vec3<f32>(0.0), vec3<f32>(1.0));
}

// AgX (Troy Sobotka), minimal approximation. Works in log2 exposure, applies a
// sigmoid that desaturates along a film-like path rather than straight to white,
// then returns. Bright saturated colours keep their hue instead of rotating the
// way ACES rotates them.
const AGX_INSET = mat3x3<f32>(
  vec3<f32>(0.8566272,  0.13731897, 0.11189821),
  vec3<f32>(0.09512124, 0.761242,   0.07679942),
  vec3<f32>(0.04825161, 0.10143904, 0.8113024),
);
const AGX_OUTSET = mat3x3<f32>(
  vec3<f32>( 1.1271006,   -0.14132976, -0.14132976),
  vec3<f32>(-0.11060664,   1.1578237,  -0.11060664),
  vec3<f32>(-0.016493939, -0.016493939, 1.2519364),
);

const AGX_MIN_EV: f32 = -12.47393;
const AGX_MAX_EV: f32 = 4.026069;

fn agx_contrast(x: vec3<f32>) -> vec3<f32> {
  let x2 = x * x;
  let x4 = x2 * x2;
  return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4
       - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - vec3<f32>(0.00232);
}

fn tonemap_agx(c: vec3<f32>) -> vec3<f32> {
  var v = AGX_INSET * max(c, vec3<f32>(0.0));
  // Floor before log2: pure black is legitimate in a render, and log2(0) is
  // -inf, which would propagate NaN through the polynomial.
  v = max(v, vec3<f32>(1e-10));
  v = log2(v);
  v = clamp((v - vec3<f32>(AGX_MIN_EV)) / (AGX_MAX_EV - AGX_MIN_EV), vec3<f32>(0.0), vec3<f32>(1.0));
  v = agx_contrast(v);
  v = max(AGX_OUTSET * v, vec3<f32>(0.0));
  // Back to display-linear; see the header.
  return clamp(pow(v, vec3<f32>(2.2)), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn tonemap(c: vec3<f32>, mode: u32) -> vec3<f32> {
  switch (mode) {
    case 1u: { return tonemap_reinhard(c); }
    case 2u: { return tonemap_aces(c); }
    case 3u: { return tonemap_agx(c); }
    default: { return clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)); }
  }
}

// The sRGB opto-electronic transfer function. Not pow(1/2.2): sRGB has a linear
// segment near black and an exponent of 2.4 on the curved segment. The gamma
// approximation shifts shadows by a couple of percent, which is the same
// magnitude as the differences a CPU/GPU comparison is looking for.
fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
  let lo = c * 12.92;
  let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
  return select(hi, lo, c <= vec3<f32>(0.0031308));
}
