//! Environment lighting: an equirectangular radiance map, importance-sampled
//! with a 2D cumulative distribution.
//!
//! # Why a CDF and not uniform sampling
//!
//! A real sky is mostly dim with a very small, very bright sun — four or five
//! orders of magnitude between them. Sampling directions uniformly puts almost
//! every sample in the dim part and finds the sun roughly as often as the sun
//! covers the sphere, which for a half-degree disc is about one sample in
//! 100 000. Each one that *does* hit carries a proportionally enormous value, so
//! the estimator is unbiased and its variance is appalling: the image is black
//! with white sparks.
//!
//! Importance sampling inverts that. Build a distribution proportional to the
//! map's own brightness and the sun gets sampled in proportion to how much light
//! it actually delivers.
//!
//! # The two-dimensional inversion
//!
//! The map is a function on the unit square, and sampling it is done by
//! conditioning:
//!
//! ```text
//!   p(u, v) = p(v) * p(u | v)
//! ```
//!
//! So first pick a **row** from the marginal distribution over rows, then a
//! **column** from that row's own conditional distribution. Each step is a
//! one-dimensional inversion — a binary search through a CDF — which is exactly
//! the kind of thing a GPU can do without divergence.
//!
//! # The `sin(theta)` weight, which is the whole trick
//!
//! An equirectangular map is a rectangle, and the sphere is not. Rows near the
//! poles are stretched: a row of pixels at the zenith covers a vanishing sliver
//! of solid angle while a row at the horizon covers a full band. The Jacobian is
//!
//! ```text
//!   dw = sin(theta) dtheta dphi = 2 pi^2 sin(theta) du dv
//! ```
//!
//! For the density in **solid angle** to be proportional to radiance, the
//! density in `(u, v)` has to be proportional to radiance times `sin(theta)` —
//! so the CDF is built on that product, not on brightness alone. Leaving the
//! weight out builds a sampler that oversamples the poles by a factor that grows
//! without bound as `theta -> 0`, and the resulting image is wrong in a way that
//! looks like nothing more than a slightly odd sky.

use crate::math::luminance;
use glam::{Vec2, Vec3};

/// A sampled environment direction.
#[derive(Clone, Copy, Debug)]
pub struct EnvSample {
    pub direction: Vec3,
    pub radiance: Vec3,
    /// Solid-angle density.
    pub pdf: f32,
}

/// An equirectangular environment map with its sampling distribution.
#[derive(Clone, Debug, Default)]
pub struct EnvMap {
    pub width: u32,
    pub height: u32,
    /// Radiance per texel, row-major from `v = 0` (the zenith). Four floats per
    /// texel because that is what a GPU texture wants; the fourth is unused.
    pub pixels: Vec<[f32; 4]>,
    /// Per-row cumulative distribution over columns, `width + 1` entries per
    /// row, each row normalised to end at 1.
    pub conditional: Vec<f32>,
    /// Cumulative distribution over rows, `height + 1` entries, ending at 1.
    pub marginal: Vec<f32>,
    /// Sum of `luminance * sin(theta)` over every texel.
    ///
    /// The normalising constant. Kept rather than folded into the CDFs because
    /// the pdf needs it directly, and recomputing it in a shader would mean
    /// summing the whole map per lookup.
    pub total_weight: f32,
}

impl EnvMap {
    pub fn is_empty(&self) -> bool {
        self.pixels.is_empty() || self.total_weight <= 0.0
    }

    /// Direction for a point on the unit square.
    ///
    /// `v = 0` is the zenith (+Y) and `v = 1` the nadir, which matches how
    /// equirectangular images are stored — row 0 at the top.
    #[inline]
    pub fn direction_from_uv(u: f32, v: f32) -> Vec3 {
        let theta = v * std::f32::consts::PI;
        let phi = u * std::f32::consts::TAU;
        let (sin_t, cos_t) = theta.sin_cos();
        Vec3::new(sin_t * phi.cos(), cos_t, sin_t * phi.sin())
    }

    /// Inverse of [`EnvMap::direction_from_uv`].
    #[inline]
    pub fn uv_from_direction(d: Vec3) -> Vec2 {
        let theta = d.y.clamp(-1.0, 1.0).acos();
        // atan2 returns (-pi, pi]; shift into [0, tau) so it maps onto [0, 1).
        let phi = d.z.atan2(d.x).rem_euclid(std::f32::consts::TAU);
        Vec2::new(
            phi / std::f32::consts::TAU,
            theta / std::f32::consts::PI,
        )
    }

    /// Build the sampling distribution over `pixels`.
    pub fn new(width: u32, height: u32, pixels: Vec<[f32; 4]>) -> EnvMap {
        assert_eq!(
            pixels.len(),
            (width as usize) * (height as usize),
            "pixel count must match the dimensions"
        );
        let (w, h) = (width as usize, height as usize);

        let mut conditional = vec![0.0f32; (w + 1) * h];
        let mut marginal = vec![0.0f32; h + 1];
        let mut row_weight = vec![0.0f32; h];

        for j in 0..h {
            // Sine at the row's **centre**, which is what makes the weight match
            // the solid angle the row actually covers. Using the row's upper
            // edge instead puts a zero at the pole row and makes it
            // unsampleable, so the topmost band of the sky never contributes.
            let theta = (j as f32 + 0.5) / h as f32 * std::f32::consts::PI;
            let sin_theta = theta.sin();

            let base = j * (w + 1);
            let mut running = 0.0f32;
            for i in 0..w {
                let p = pixels[j * w + i];
                let weight = luminance(Vec3::new(p[0], p[1], p[2])).max(0.0) * sin_theta;
                running += weight;
                conditional[base + i + 1] = running;
            }
            row_weight[j] = running;

            // Normalise the row. A row that is entirely black has no
            // distribution of its own; it is left as zeros and the marginal
            // simply never selects it.
            if running > 0.0 {
                let inv = 1.0 / running;
                for k in 1..=w {
                    conditional[base + k] *= inv;
                }
                // Exactly 1.0 at the end, so the search can never fall off it
                // because of accumulated rounding.
                conditional[base + w] = 1.0;
            }
        }

        let mut running = 0.0f32;
        for j in 0..h {
            running += row_weight[j];
            marginal[j + 1] = running;
        }
        let total_weight = running;
        if total_weight > 0.0 {
            let inv = 1.0 / total_weight;
            for m in marginal.iter_mut().take(h + 1).skip(1) {
                *m *= inv;
            }
            marginal[h] = 1.0;
        }

        EnvMap {
            width,
            height,
            pixels,
            conditional,
            marginal,
            total_weight,
        }
    }

    /// Radiance stored at a direction.
    ///
    /// Nearest-texel, deliberately. Bilinear filtering would return radiance
    /// that the CDF does not describe: the distribution is built over
    /// piecewise-constant cells, so a filtered lookup makes `pdf` and the value
    /// it is supposed to be proportional to disagree, and the estimator picks up
    /// a bias that no energy test would flag. It also keeps CPU and GPU
    /// identical, since `textureLoad` has no implementation-defined filtering.
    pub fn radiance(&self, direction: Vec3) -> Vec3 {
        if self.pixels.is_empty() {
            return Vec3::ZERO;
        }
        let uv = Self::uv_from_direction(direction);
        let (i, j) = self.texel_of(uv);
        let p = self.pixels[j * self.width as usize + i];
        Vec3::new(p[0], p[1], p[2])
    }

    #[inline]
    fn texel_of(&self, uv: Vec2) -> (usize, usize) {
        let i = ((uv.x * self.width as f32) as i64).clamp(0, self.width as i64 - 1) as usize;
        let j = ((uv.y * self.height as f32) as i64).clamp(0, self.height as i64 - 1) as usize;
        (i, j)
    }

    /// Solid-angle density this sampler assigns to a direction.
    ///
    /// Zero at the poles: `sin(theta)` vanishes there and the conversion from
    /// the unit square to solid angle divides by it. That is not a special case
    /// being papered over — a single direction exactly at the pole is the image
    /// of an entire edge of the square, so it genuinely has no density.
    pub fn pdf(&self, direction: Vec3) -> f32 {
        if self.is_empty() {
            return 0.0;
        }
        let uv = Self::uv_from_direction(direction);
        let (i, j) = self.texel_of(uv);
        let (w, h) = (self.width as usize, self.height as usize);

        let theta = (j as f32 + 0.5) / h as f32 * std::f32::consts::PI;
        let sin_theta = theta.sin();
        if sin_theta <= 0.0 {
            return 0.0;
        }
        let p = self.pixels[j * w + i];
        let weight = luminance(Vec3::new(p[0], p[1], p[2])).max(0.0) * sin_theta;

        // Density over the unit square: the texel's share of the total, spread
        // over a cell of area 1/(w*h).
        let pdf_uv = weight / self.total_weight * (w * h) as f32;
        // And into solid angle. dw = 2 pi^2 sin(theta) du dv.
        pdf_uv / (2.0 * std::f32::consts::PI * std::f32::consts::PI * sin_theta)
    }

    /// Draw a direction proportional to radiance times `sin(theta)`.
    pub fn sample(&self, u: Vec2) -> EnvSample {
        let invalid = EnvSample {
            direction: Vec3::Y,
            radiance: Vec3::ZERO,
            pdf: 0.0,
        };
        if self.is_empty() {
            return invalid;
        }
        let (w, h) = (self.width as usize, self.height as usize);

        // Row from the marginal, then column from that row's conditional.
        let (j, dv) = sample_cdf(&self.marginal, 0, h, u.y);
        let (i, du) = sample_cdf(&self.conditional, j * (w + 1), w, u.x);

        let uv = Vec2::new(
            (i as f32 + du) / w as f32,
            (j as f32 + dv) / h as f32,
        );
        let direction = Self::direction_from_uv(uv.x, uv.y);

        let theta = (j as f32 + 0.5) / h as f32 * std::f32::consts::PI;
        let sin_theta = theta.sin();
        if sin_theta <= 0.0 {
            return invalid;
        }
        let p = self.pixels[j * w + i];
        let radiance = Vec3::new(p[0], p[1], p[2]);
        let weight = luminance(radiance).max(0.0) * sin_theta;
        let pdf_uv = weight / self.total_weight * (w * h) as f32;
        let pdf = pdf_uv / (2.0 * std::f32::consts::PI * std::f32::consts::PI * sin_theta);
        if pdf <= 0.0 {
            return invalid;
        }

        EnvSample {
            direction,
            radiance,
            pdf,
        }
    }
}

/// Invert one CDF: find the cell containing `x` and where inside it.
///
/// `cdf[base ..= base + n]` is non-decreasing from 0 to 1. Returns the cell
/// index and the position within it, so the caller can place the sample
/// continuously rather than at the cell's corner — stratification within the
/// cell costs nothing and removes the blockiness a corner-only sample would
/// give at low resolution.
///
/// Binary search rather than a linear scan because this runs per sample per
/// bounce on the GPU: 11 iterations for a 2048-wide map against up to 2048.
/// Written so both branches do the same work, which is what keeps a warp from
/// diverging.
fn sample_cdf(cdf: &[f32], base: usize, n: usize, x: f32) -> (usize, f32) {
    // Largest index with cdf[index] <= x.
    let mut lo = 0usize;
    let mut len = n;
    while len > 0 {
        let half = len / 2;
        let mid = lo + half;
        if cdf[base + mid] <= x {
            lo = mid + 1;
            len -= half + 1;
        } else {
            len = half;
        }
    }
    let cell = lo.saturating_sub(1).min(n - 1);

    let c0 = cdf[base + cell];
    let c1 = cdf[base + cell + 1];
    let span = c1 - c0;
    // A zero-width cell carries no probability and can only be reached through
    // rounding; placing the sample at its start is as good as anywhere.
    let frac = if span > 0.0 {
        ((x - c0) / span).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (cell, frac)
}

/// # What importance sampling does not fix
///
/// The sky sampler removes the variance that comes from *finding* the sun. It
/// does nothing for paths that reach the sun only after a specular bounce — off
/// a mirror, or through glass — because those directions are chosen by the BSDF
/// and the environment sampler never gets to steer them. Measured on `sunset`,
/// which has a mirror and a glass ball next to a diffuse sphere: per-pixel noise
/// is 0.042 on the diffuse sphere alone but 0.091 for the full scene, and
/// quadrupling the sample count cuts it by 0.71x rather than the 0.5x of a
/// well-behaved estimator. That slower-than-`1/sqrt(N)` rate is the signature of
/// a heavy tail, and it is what caustics are.
///
/// Fixing it needs a different family of techniques — photon mapping, vertex
/// connection and merging, or path guiding — not a better environment CDF.
///
/// A procedural sky, so the renderer has a high-dynamic-range environment
/// without committing a megabyte of HDR to the repository.
///
/// Not a physical sky model — it is a gradient, a sun disc and a ground plane —
/// but it has the property that matters for testing an importance sampler: the
/// sun is about four orders of magnitude brighter than the sky and covers a
/// fraction of a percent of the sphere. Uniform sampling finds it roughly once
/// in ten thousand tries, so any failure of the CDF shows up immediately as
/// either a black image or a sparkling one.
pub fn procedural_sky(
    width: u32,
    height: u32,
    sun_direction: Vec3,
    sun_intensity: f32,
    sun_radius_degrees: f32,
) -> EnvMap {
    let sun = sun_direction.normalize_or(Vec3::Y);

    // The sun cannot be smaller than a texel, and the clamp is not a
    // convenience — it is the honest statement of a limit of image-based
    // lighting.
    //
    // The real sun is about 0.27 degrees in radius. One texel of a 256x128 map
    // spans 1.4 degrees, so a correctly-sized sun is *sub-texel*: whether it
    // appears at all depends on whether a texel centre happens to fall inside
    // it, and at this resolution it usually does not. Written without this
    // clamp, the generator produced a sky with no sun in it and the sampler was
    // blamed — 0.1% of samples found a sun that was not there.
    //
    // Production HDR skies solve this with resolution (4K and up) plus an
    // analytic sun evaluated separately from the map. Here the sun is simply
    // widened to something the map can represent, and the intensity is scaled
    // down by the area ratio so the total power it delivers stays put.
    let texel = (std::f32::consts::PI / height as f32).max(std::f32::consts::TAU / width as f32);
    let requested = sun_radius_degrees.to_radians();
    let radius = requested.max(texel);
    // Solid angle goes as (1 - cos r); widening the disc must dim it by the
    // same factor or the sky gains energy it was never given.
    let area_ratio = (1.0 - requested.cos()) / (1.0 - radius.cos()).max(1e-20);
    let intensity = sun_intensity * area_ratio.min(1.0);
    let cos_sun_radius = radius.cos();

    let mut pixels = Vec::with_capacity((width * height) as usize);
    for j in 0..height {
        let v = (j as f32 + 0.5) / height as f32;
        for i in 0..width {
            let u = (i as f32 + 0.5) / width as f32;
            let d = EnvMap::direction_from_uv(u, v);

            let mut c = if d.y >= 0.0 {
                // Sky: warm near the horizon, blue at the zenith. The exponent
                // is chosen by eye; nothing downstream depends on it.
                let t = d.y.powf(0.45);
                Vec3::new(0.75, 0.86, 1.00) * t + Vec3::new(0.85, 0.70, 0.55) * (1.0 - t)
            } else {
                // Ground: dim and neutral, so the lower hemisphere contributes
                // some bounce light without competing with the sky.
                Vec3::splat(0.12)
            };
            // Scaled so a white diffuse surface under the whole sky lands near
            // 0.4 rather than clipping. The renderer carries radiance, not
            // display values, and nothing downstream clamps — but a sky that
            // puts every surface above 1.0 tonemaps to a white frame and hides
            // exactly the detail these scenes exist to show.
            c *= 0.40;

            if d.dot(sun) >= cos_sun_radius {
                // Slightly warm, and far brighter than anything else.
                c += Vec3::new(1.0, 0.92, 0.78) * intensity;
            }
            pixels.push([c.x, c.y, c.z, 0.0]);
        }
    }
    EnvMap::new(width, height, pixels)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_map() -> EnvMap {
        procedural_sky(128, 64, Vec3::new(0.3, 0.6, 0.2), 4000.0, 1.5)
    }

    #[test]
    fn uv_and_direction_round_trip() {
        for &(u, v) in &[
            (0.0f32, 0.5f32),
            (0.25, 0.25),
            (0.5, 0.5),
            (0.75, 0.9),
            (0.999, 0.001),
        ] {
            let d = EnvMap::direction_from_uv(u, v);
            assert!(
                (d.length() - 1.0).abs() < 1e-5,
                "direction must be unit, got {}",
                d.length()
            );
            let back = EnvMap::uv_from_direction(d);
            assert!(
                (back.x - u).abs() < 1e-4 || (back.x - u).abs() > 0.999,
                "u round trip: {u} -> {}",
                back.x
            );
            assert!((back.y - v).abs() < 1e-4, "v round trip: {v} -> {}", back.y);
        }
    }

    #[test]
    fn poles_map_to_the_axis() {
        assert!((EnvMap::direction_from_uv(0.3, 0.0) - Vec3::Y).length() < 1e-6);
        assert!((EnvMap::direction_from_uv(0.3, 1.0) + Vec3::Y).length() < 1e-6);
    }

    /// The CDFs must be non-decreasing and end at exactly 1.
    ///
    /// A CDF that ends slightly below 1 lets a random number land past its end,
    /// and the search then returns the last cell for a whole sliver of the unit
    /// interval — a rare, bright, entirely wrong sample. Pinning the endpoint is
    /// cheaper than chasing that later.
    #[test]
    fn cdfs_are_monotonic_and_normalised() {
        let m = test_map();
        let (w, h) = (m.width as usize, m.height as usize);
        for k in 0..h {
            assert!(m.marginal[k] <= m.marginal[k + 1], "marginal decreased");
        }
        assert_eq!(m.marginal[0], 0.0);
        assert_eq!(m.marginal[h], 1.0);

        for j in 0..h {
            let base = j * (w + 1);
            assert_eq!(m.conditional[base], 0.0);
            for i in 0..w {
                assert!(
                    m.conditional[base + i] <= m.conditional[base + i + 1],
                    "conditional row {j} decreased at {i}"
                );
            }
            // Rows with any light must be normalised; a black row stays zero.
            let last = m.conditional[base + w];
            assert!(last == 0.0 || last == 1.0, "row {j} ends at {last}");
        }
    }

    /// The pdf must integrate to 1 over the sphere.
    ///
    /// Computed as a sum over texels rather than by sampling, so it is exact
    /// arithmetic rather than a Monte Carlo estimate: each texel covers a known
    /// solid angle, and the pdf is constant across it.
    #[test]
    fn pdf_integrates_to_one() {
        let m = test_map();
        let (w, h) = (m.width as usize, m.height as usize);
        let mut total = 0.0f64;
        for j in 0..h {
            let theta = (j as f32 + 0.5) / h as f32 * std::f32::consts::PI;
            // Solid angle of one texel in this row.
            let d_omega = (std::f32::consts::TAU / w as f32)
                * (std::f32::consts::PI / h as f32)
                * theta.sin();
            for i in 0..w {
                let u = (i as f32 + 0.5) / w as f32;
                let v = (j as f32 + 0.5) / h as f32;
                let d = EnvMap::direction_from_uv(u, v);
                total += (m.pdf(d) * d_omega) as f64;
            }
        }
        assert!(
            (total - 1.0).abs() < 1e-3,
            "pdf integrates to {total:.6}, not 1 — the sin(theta) weight or the \
             solid-angle Jacobian is wrong"
        );
    }

    /// Sampling must agree with the density it claims.
    #[test]
    fn sample_reports_the_same_pdf_as_pdf() {
        let m = test_map();
        let mut rng = crate::rng::Rng::new(0xE1, 0, 0);
        let mut checked = 0;
        for _ in 0..20_000 {
            let s = m.sample(rng.next_vec2());
            if s.pdf <= 0.0 {
                continue;
            }
            checked += 1;
            let q = m.pdf(s.direction);
            assert!(
                (s.pdf - q).abs() <= 1e-3 * s.pdf.max(q),
                "sample() reported {} but pdf() says {q} for {:?}",
                s.pdf,
                s.direction
            );
        }
        assert!(checked > 19_000, "too many samples rejected: {checked}");
    }

    /// Samples must land on the bright parts in proportion to the light there.
    ///
    /// The point of the whole module. Stated against the map's *own* weights
    /// rather than against a guessed percentage: whatever share of the total
    /// power sits in the brightest texels, that is the share of samples they
    /// should receive. An earlier version asserted a round number instead and
    /// failed for a reason that had nothing to do with the sampler.
    #[test]
    fn sampling_follows_the_energy() {
        let sun = Vec3::new(0.3, 0.6, 0.2).normalize();
        let m = procedural_sky(256, 128, sun, 4000.0, 1.5);
        let (w, h) = (m.width as usize, m.height as usize);

        // Which texels are "bright": the sun disc, found from the map itself.
        let mut bright: Vec<bool> = vec![false; w * h];
        let mut bright_weight = 0.0f64;
        let mut total = 0.0f64;
        for j in 0..h {
            let sin_theta = ((j as f32 + 0.5) / h as f32 * std::f32::consts::PI).sin();
            for i in 0..w {
                let p = m.pixels[j * w + i];
                let lum = luminance(Vec3::new(p[0], p[1], p[2]));
                let weight = (lum * sin_theta) as f64;
                total += weight;
                // The sun is orders of magnitude above the sky, so any
                // threshold between them separates the two cleanly.
                if lum > 50.0 {
                    bright[j * w + i] = true;
                    bright_weight += weight;
                }
            }
        }
        let expected = bright_weight / total;
        assert!(
            expected > 0.05,
            "the test sky has no sun in it ({:.4}% of its power is bright); \
             the generator, not the sampler, is at fault",
            100.0 * expected
        );

        let mut rng = crate::rng::Rng::new(0x5E8, 0, 0);
        let mut on_sun = 0usize;
        let n = 100_000;
        for _ in 0..n {
            let s = m.sample(rng.next_vec2());
            let uv = EnvMap::uv_from_direction(s.direction);
            let i = ((uv.x * w as f32) as usize).min(w - 1);
            let j = ((uv.y * h as f32) as usize).min(h - 1);
            if bright[j * w + i] {
                on_sun += 1;
            }
        }
        let measured = on_sun as f64 / n as f64;
        eprintln!(
            "sun holds {:.1}% of the power and received {:.1}% of the samples",
            100.0 * expected,
            100.0 * measured
        );
        assert!(
            (measured - expected).abs() < 0.02,
            "the sun holds {:.2}% of the map's power but received {:.2}% of the \
             samples. The CDF is not tracking brightness — check that it is built \
             on luminance * sin(theta) and that the marginal and conditional are \
             not transposed.",
            100.0 * expected,
            100.0 * measured
        );
    }

    /// Importance sampling must beat uniform sampling, and by a lot.
    ///
    /// Both estimators compute the same integral — the map's radiance over the
    /// sphere — so they must agree; what differs is the variance at equal
    /// sample count.
    ///
    /// The CDF's variance here comes out at ~1e-10, which is zero to float
    /// precision, and that is not a suspiciously good result: the density is
    /// built to be proportional to luminance, so `luminance / pdf` is
    /// *constant* and this particular integral is estimated exactly by one
    /// sample. The ratio is therefore not a meaningful speedup figure and is
    /// not reported as one.
    ///
    /// Real renders integrate radiance against a BSDF and a cosine, which the
    /// map's own distribution only approximates — the realistic gain is large
    /// but finite, and it is measured where it belongs, on an image, in
    /// `envmap_sampling_converges_faster`.
    #[test]
    fn importance_sampling_beats_uniform() {
        let sun = Vec3::new(0.3, 0.6, 0.2).normalize();
        let m = procedural_sky(256, 128, sun, 4000.0, 1.5);
        let n = 40_000;

        let mut rng = crate::rng::Rng::new(0x1B, 0, 0);
        let (mut s1, mut s2) = (0.0f64, 0.0f64);
        for _ in 0..n {
            let s = m.sample(rng.next_vec2());
            // L / p, the estimator of the integral of L over the sphere.
            let v = if s.pdf > 0.0 {
                (luminance(s.radiance) / s.pdf) as f64
            } else {
                0.0
            };
            s1 += v;
            s2 += v * v;
        }
        let cdf_mean = s1 / n as f64;
        let cdf_var = (s2 / n as f64 - cdf_mean * cdf_mean).max(0.0);

        let mut rng = crate::rng::Rng::new(0x1B, 0, 0);
        let (mut u1, mut u2) = (0.0f64, 0.0f64);
        let uniform_pdf = 1.0 / (4.0 * std::f32::consts::PI);
        for _ in 0..n {
            let uv = rng.next_vec2();
            // Uniform on the sphere: z uniform in [-1, 1], phi uniform.
            let z = 1.0 - 2.0 * uv.x;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let phi = std::f32::consts::TAU * uv.y;
            let d = Vec3::new(r * phi.cos(), z, r * phi.sin());
            let v = (luminance(m.radiance(d)) / uniform_pdf) as f64;
            u1 += v;
            u2 += v * v;
        }
        let uni_mean = u1 / n as f64;
        let uni_var = (u2 / n as f64 - uni_mean * uni_mean).max(0.0);

        eprintln!(
            "integral: cdf {cdf_mean:.3} (var {cdf_var:.2e}), uniform {uni_mean:.3} \
             (var {uni_var:.2e}) — the CDF variance is zero to float precision \
             because p is proportional to L by construction"
        );
        // Same integral: both are unbiased estimators of the same quantity.
        assert!(
            (cdf_mean - uni_mean).abs() < 0.15 * uni_mean.max(cdf_mean),
            "the two estimators disagree on the integral ({cdf_mean:.3} vs \
             {uni_mean:.3}); one of them is biased"
        );
        assert!(
            uni_var > 1.0e4 * cdf_var,
            "importance sampling barely reduced variance ({uni_var:.2e} against \
             {cdf_var:.2e}). For a density built to be proportional to luminance \
             this estimator should be essentially exact."
        );
    }

    /// An all-black map must not produce samples or densities.
    #[test]
    fn black_map_is_handled() {
        let m = EnvMap::new(4, 2, vec![[0.0; 4]; 8]);
        assert!(m.is_empty());
        assert_eq!(m.pdf(Vec3::Y), 0.0);
        assert_eq!(m.sample(Vec2::new(0.5, 0.5)).pdf, 0.0);
    }

    /// A uniform map's density must be uniform: exactly 1/(4 pi) everywhere.
    ///
    /// The case where every subtlety cancels, so any surviving error in the
    /// Jacobian shows up as a clean constant factor.
    #[test]
    fn uniform_map_has_uniform_density() {
        let m = EnvMap::new(64, 32, vec![[1.0, 1.0, 1.0, 0.0]; 64 * 32]);
        let expected = 1.0 / (4.0 * std::f32::consts::PI);
        for &d in &[
            Vec3::Y,
            Vec3::X,
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.5, 0.5, 0.5).normalize(),
        ] {
            let p = m.pdf(d);
            assert!(
                (p - expected).abs() < 2e-3 * expected.max(p),
                "uniform map gave density {p} at {d:?}, expected {expected}"
            );
        }
    }
}
