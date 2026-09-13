// Edge-avoiding à-trous wavelet denoising. Mirrors crates/core/src/denoise.rs.
//
// Three entry points, run as a chain:
//
//   demodulate  accum -> filter buffer, dividing out the albedo, and measure
//               the local luminance deviation once
//   filter      one à-trous pass, ping-ponging between two buffers
//   modulate    multiply the albedo back in, into the output the display reads
//
// The accumulation buffer is never written. Denoising is a display-time product:
// one more sample still converges to the unbiased answer.

//!include "common/generated.wgsl"
//!include "common/math.wgsl"

struct DenoiseParams {
  width          : u32,
  height         : u32,
  stride         : u32,
  sigma_normal   : f32,
  sigma_depth    : f32,
  sigma_luminance: f32,
  sigma_albedo   : f32,
  _pad0          : u32,
};

@group(0) @binding(0) var<uniform> D: DenoiseParams;
@group(0) @binding(1) var<storage, read>       accum     : array<Accum>;
@group(0) @binding(2) var<storage, read_write> buf_a     : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> buf_b     : array<vec4<f32>>;
// Per-pixel luminance deviation, measured once on the demodulated signal.
@group(0) @binding(4) var<storage, read_write> deviation : array<f32>;
// What the display pass reads.
//
// Shaped like the accumulation buffer rather than as plain RGBA, so the display
// pass reads it with no change at all — switching the denoiser on swaps which
// buffer is bound and nothing else. `samples` is written as 1 because the
// radiance here is already averaged.
@group(0) @binding(5) var<storage, read_write> output    : array<Accum>;

// The 5-tap B-spline kernel, [1, 4, 6, 4, 1] / 16. Repeated convolution of it
// converges to a Gaussian quickly, which is what makes an à-trous stack a good
// approximation of a wide blur after only a few passes.
const KERNEL: array<f32, 5> = array<f32, 5>(0.0625, 0.25, 0.375, 0.25, 0.0625);

fn pixel_count() -> u32 {
  return D.width * D.height;
}

// Averaged guide channels for a pixel. `samples` divides all of them.
struct Guide {
  albedo : vec3<f32>,
  normal : vec3<f32>,
  depth  : f32,
};

fn read_guide(i: u32) -> Guide {
  let a = accum[i];
  let inv = 1.0 / max(a.samples, 1.0);
  var g: Guide;
  g.albedo = max(a.albedo * inv, vec3<f32>(1.0e-3));
  g.normal = a.normal * inv;
  g.depth = a.depth * inv;
  return g;
}

// accum -> buf_a, divided by albedo.
//
// Filtering `radiance / albedo` and multiplying back afterwards keeps texture
// out of the filter's way: the denoiser sees only the lighting, which is smooth,
// instead of lighting times a pattern its edge-stopping weights would spend
// themselves preserving.
@compute @workgroup_size(8, 8, 1)
fn demodulate(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= D.width || gid.y >= D.height) {
    return;
  }
  let i = gid.y * D.width + gid.x;
  let a = accum[i];
  let inv = 1.0 / max(a.samples, 1.0);
  let g = read_guide(i);
  buf_a[i] = vec4<f32>(a.radiance * inv / g.albedo, 0.0);
}

// Measure the local luminance deviation, on the **demodulated** signal.
//
// That is the quantity the luminance weight compares, and measuring it on the
// modulated image instead is wrong exactly where it matters most: an area light
// sits at radiance 16 beside a ceiling at 0.2, so its local deviation is
// enormous, the tolerance derived from it admits every neighbour, and the filter
// smears the light across the ceiling. Measured on the CPU before this was
// fixed: the Cornell box lost 39% of its energy.
//
// Measured once rather than per pass: recomputing it would let it shrink as the
// filter smooths, and the filter would keep finding its own output clean enough
// and blur without limit.
@compute @workgroup_size(8, 8, 1)
fn measure(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= D.width || gid.y >= D.height) {
    return;
  }
  let i = gid.y * D.width + gid.x;
  var sum = 0.0;
  var sum_sq = 0.0;
  var n = 0.0;
  for (var dy = -1; dy <= 1; dy = dy + 1) {
    for (var dx = -1; dx <= 1; dx = dx + 1) {
      let px = i32(gid.x) + dx;
      let py = i32(gid.y) + dy;
      if (px < 0 || py < 0 || px >= i32(D.width) || py >= i32(D.height)) {
        continue;
      }
      let l = luminance(buf_a[u32(py) * D.width + u32(px)].rgb);
      sum = sum + l;
      sum_sq = sum_sq + l * l;
      n = n + 1.0;
    }
  }
  let mean = sum / n;
  deviation[i] = sqrt(max(sum_sq / n - mean * mean, 0.0));
}

// One à-trous pass. `D.stride` doubles each time, so a 5x5 kernel applied at
// 1, 2, 4, 8 touches what a 65x65 would at a sixteenth of the cost.
//
// Reads buf_a and writes buf_b; the host swaps the bindings between passes.
@compute @workgroup_size(8, 8, 1)
// Named `atrous_pass` and not `filter`: that is a WGSL reserved keyword.
fn atrous_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= D.width || gid.y >= D.height) {
    return;
  }
  let ci = gid.y * D.width + gid.x;
  let cg = read_guide(ci);
  let cl = luminance(buf_a[ci].rgb);
  let ca = max(luminance(cg.albedo), 1.0e-3);
  // The floor keeps a perfectly flat region from dividing by zero and rejecting
  // every tap.
  let lum_scale = D.sigma_luminance * max(deviation[ci], 1.0e-4);

  var sum = vec3<f32>(0.0);
  var weight_sum = 0.0;
  for (var ky = 0; ky < 5; ky = ky + 1) {
    for (var kx = 0; kx < 5; kx = kx + 1) {
      let px = i32(gid.x) + (kx - 2) * i32(D.stride);
      let py = i32(gid.y) + (ky - 2) * i32(D.stride);
      if (px < 0 || py < 0 || px >= i32(D.width) || py >= i32(D.height)) {
        continue;
      }
      let qi = u32(py) * D.width + u32(px);
      let qg = read_guide(qi);

      // Normal: a dot product raised to a power. Sharp falloff, so adjacent
      // faces of a box do not bleed together.
      let w_n = pow(max(dot(cg.normal, qg.normal), 0.0), D.sigma_normal);

      // Depth: relative to the centre, so one tolerance works at any scene
      // scale. A background pixel has depth 0 and is excluded from a foreground
      // pixel's neighbourhood by this alone.
      let dd = abs(cg.depth - qg.depth);
      let w_d = exp(-dd / (D.sigma_depth * max(abs(cg.depth), 1.0e-3)));

      // Luminance, against the noise actually present.
      let dl = abs(cl - luminance(buf_a[qi].rgb));
      let w_l = exp(-dl / lum_scale);

      // Albedo: different materials do not blend, however similar their
      // geometry. Demodulation removes *texture* from the filter's view; this
      // removes *material boundaries*, which is a different problem — an area
      // light and the ceiling it is set into share a normal and a depth, so
      // nothing else separates them.
      let da = abs(ca - luminance(qg.albedo));
      let w_a = exp(-da / (D.sigma_albedo * ca));

      let w = KERNEL[kx] * KERNEL[ky] * w_n * w_d * w_l * w_a;
      sum = sum + buf_a[qi].rgb * w;
      weight_sum = weight_sum + w;
    }
  }
  buf_b[ci] = select(buf_a[ci], vec4<f32>(sum / weight_sum, 0.0), weight_sum > 0.0);
}

// Multiply the albedo back in.
@compute @workgroup_size(8, 8, 1)
fn modulate(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= D.width || gid.y >= D.height) {
    return;
  }
  let i = gid.y * D.width + gid.x;
  let g = read_guide(i);
  var o: Accum;
  o.radiance = buf_a[i].rgb * g.albedo;
  o.samples = 1.0;
  o.albedo = g.albedo;
  o.normal = g.normal;
  o.depth = g.depth;
  output[i] = o;
}
