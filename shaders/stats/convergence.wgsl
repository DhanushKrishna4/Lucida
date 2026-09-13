// Reduce the accumulation buffer to one number: how converged the image is.
//
// # What it reports
//
// With the sum and the sum of squares of the per-sample estimates, the variance
// of the estimator at a pixel is `sum_sq/N - mean^2`, and the standard error of
// its mean is `sqrt(var/N)`. Dividing by the mean gives a **relative** standard
// error, which is comparable across pixels of wildly different brightness and
// across scenes.
//
// The reported figure is the mean of that over the pixels bright enough to be
// meaningful. It is a *measurement*, not the `1/sqrt(N)` a model would predict —
// and this renderer has spent a while establishing that the two part company
// whenever the integrand is heavy-tailed, which is exactly when someone wants to
// know how converged things are.
//
// # It assumes independent samples, and says so
//
// `Var[mean] = Var[sample] / N` holds when the samples are independent. Sobol's
// samples are deliberately *not*: stratification correlates them so they cover
// the domain more evenly than chance would, and the true spread of the mean
// falls below what this computes. Measured on the Cornell box at 64 spp: the
// meter reports 0.115 where sixteen independent renders actually spread by
// 0.091, so the figure is 1.27x conservative.
//
// Left conservative on purpose. Correcting it would mean estimating the
// stratification gain, which is scene-dependent and is exactly the thing nobody
// can do cheaply; and of the two ways to be wrong, over-reporting noise is much
// the better one — the alternative tells someone an image is finished when it is
// not. The UI labels it accordingly.
//
// # Why two passes
//
// A single workgroup reducing a megapixel serially would take longer than the
// frame it is reporting on. Instead each workgroup reduces a strided slice into
// one partial, and a second pass folds the partials — the standard shape, and
// the strided access is what makes the loads coalesce.

//!include "common/generated.wgsl"
//!include "common/math.wgsl"

struct StatsParams {
  pixels   : u32,
  partials : u32,
  _pad0    : u32,
  _pad1    : u32,
};

@group(0) @binding(0) var<uniform> S: StatsParams;
@group(0) @binding(1) var<storage, read>       accum    : array<Accum>;
// x = summed relative standard error, y = how many pixels contributed.
@group(0) @binding(2) var<storage, read_write> partials : array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> result   : array<f32>;

const WG: u32 = 256u;

// Below this a pixel is essentially black, and a relative error against it is a
// division by noise. Excluded rather than clamped: including them would make a
// mostly-dark scene report a huge error that never improves, which is worse than
// saying nothing.
const MIN_LUMINANCE: f32 = 1.0e-3;

var<workgroup> red_err: array<f32, WG>;
var<workgroup> red_cnt: array<f32, WG>;

fn fold(lid: u32) {
  for (var s = WG / 2u; s > 0u; s = s / 2u) {
    if (lid < s) {
      red_err[lid] = red_err[lid] + red_err[lid + s];
      red_cnt[lid] = red_cnt[lid] + red_cnt[lid + s];
    }
    workgroupBarrier();
  }
}

@compute @workgroup_size(256, 1, 1)
fn reduce(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(num_workgroups) groups: vec3<u32>,
) {
  var err = 0.0;
  var cnt = 0.0;
  // Strided so consecutive threads read consecutive pixels.
  let stride = groups.x * WG;
  var i = wid.x * WG + lid.x;
  loop {
    if (i >= S.pixels) {
      break;
    }
    let a = accum[i];
    let n = a.samples;
    // Two samples minimum: a variance from one is not a variance.
    if (n >= 2.0) {
      let mean = a.radiance / n;
      let mean_l = luminance(mean);
      if (mean_l > MIN_LUMINANCE) {
        // Variance of the per-sample estimates, then of their mean.
        let var_rgb = max(a.radiance_sq / n - mean * mean, vec3<f32>(0.0));
        let se = sqrt(luminance(var_rgb) / n);
        err = err + se / mean_l;
        cnt = cnt + 1.0;
      }
    }
    i = i + stride;
  }

  red_err[lid.x] = err;
  red_cnt[lid.x] = cnt;
  workgroupBarrier();
  fold(lid.x);
  if (lid.x == 0u) {
    partials[wid.x] = vec2<f32>(red_err[0], red_cnt[0]);
  }
}

// Fold the partials. One workgroup; there are only as many partials as there
// were workgroups above.
@compute @workgroup_size(256, 1, 1)
fn finish(@builtin(local_invocation_id) lid: vec3<u32>) {
  var err = 0.0;
  var cnt = 0.0;
  var i = lid.x;
  loop {
    if (i >= S.partials) {
      break;
    }
    err = err + partials[i].x;
    cnt = cnt + partials[i].y;
    i = i + WG;
  }
  red_err[lid.x] = err;
  red_cnt[lid.x] = cnt;
  workgroupBarrier();
  fold(lid.x);
  if (lid.x == 0u) {
    // Negative means "nothing measurable yet", which the reader shows as a dash
    // rather than as a confident zero.
    result[0] = select(-1.0, red_err[0] / red_cnt[0], red_cnt[0] > 0.0);
  }
}
