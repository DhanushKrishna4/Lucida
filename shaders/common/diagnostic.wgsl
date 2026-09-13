// Diagnostic render modes. Mirrors crates/core/src/diagnostic.rs.
//
// Split out of display.wgsl so the native harness can evaluate the ramp against
// its Rust twin in `cargo test`. A colour ramp is exactly the kind of thing that
// looks fine while being wrong — nobody can tell a slightly-off gradient by eye.

const MODE_BEAUTY: u32 = 0u;
const MODE_NORMAL: u32 = 1u;
const MODE_ALBEDO: u32 = 2u;
const MODE_DEPTH: u32 = 3u;
const MODE_HEAT: u32 = 4u;

// Node visits per ray that map to the top of the ramp. Fixed rather than scaled
// to the image's own maximum: auto-scaling makes every heatmap look the same, so
// it can never say whether a change helped, which is the only question anyone
// asks of one. Mirrors `HEAT_SCALE` in crates/core/src/diagnostic.rs.
const HEAT_SCALE: f32 = 128.0;

// A perceptually-ordered ramp: black, blue, green, yellow, red, white. Ordered
// by lightness as well as hue, so it reads in greyscale and for a red-green
// colour-blind viewer — a plain blue-to-red ramp fails both, its middle being a
// lightness plateau exactly where the interesting transition is.
fn heat_ramp(t_in: f32) -> vec3<f32> {
  let t = clamp(t_in, 0.0, 1.0);
  if (t <= 0.20) {
    return mix(vec3<f32>(0.0, 0.0, 0.05), vec3<f32>(0.10, 0.15, 0.60), t / 0.20);
  }
  if (t <= 0.45) {
    return mix(vec3<f32>(0.10, 0.15, 0.60), vec3<f32>(0.05, 0.65, 0.35), (t - 0.20) / 0.25);
  }
  if (t <= 0.68) {
    return mix(vec3<f32>(0.05, 0.65, 0.35), vec3<f32>(0.95, 0.85, 0.10), (t - 0.45) / 0.23);
  }
  if (t <= 0.87) {
    return mix(vec3<f32>(0.95, 0.85, 0.10), vec3<f32>(0.90, 0.20, 0.10), (t - 0.68) / 0.19);
  }
  return mix(vec3<f32>(0.90, 0.20, 0.10), vec3<f32>(1.0, 1.0, 1.0), (t - 0.87) / 0.13);
}
