//! Edge-avoiding à-trous wavelet denoising (Dammertz et al. 2010).
//!
//! # What a denoiser can and cannot do
//!
//! Monte Carlo noise is *zero-mean*: average enough samples and it goes away.
//! A denoiser is a bet that the true image is locally smooth, so nearby pixels
//! can be averaged together to stand in for samples that were never taken.
//! Where that bet holds it converts noise into a few milliseconds; where it
//! fails — across a silhouette, a shadow boundary, a texture edge — it converts
//! noise into **blur**, which is a bias no amount of further sampling removes.
//!
//! So everything here is machinery for deciding *which neighbours are the same
//! surface*, and the denoised image is a display-time product: the accumulation
//! buffer is never touched, and one more sample still converges to the right
//! answer.
//!
//! # À-trous: a wide blur without a wide loop
//!
//! A 5x5 filter applied at stride 1, then 2, then 4, then 8 touches the same
//! pixels a 65x65 filter would, at a sixteenth of the cost — `à trous` is
//! "with holes", the gaps between the taps growing each pass. Each pass feeds
//! the next, so the effective kernel is the composition rather than any single
//! one.
//!
//! # Edge stopping
//!
//! Each tap is weighted by how much it looks like the centre pixel, judged on
//! three things that are *noise-free* because they come from one deterministic
//! intersection rather than from an integral:
//!
//! * **normal** — a dot product raised to a power, which falls off sharply and
//!   keeps a cube's faces from bleeding into one another;
//! * **depth** — separates surfaces that overlap on screen but are far apart;
//! * **luminance** — the only noisy input, and the one that preserves shadow
//!   edges. Its tolerance has to scale with the noise actually present, or it
//!   will treat noise as detail and refuse to filter anything.
//!
//! # Albedo demodulation
//!
//! The filter runs on `radiance / albedo`, and the albedo is multiplied back
//! afterwards. Without it the denoiser sees lighting *times* texture, and its
//! edge-stopping weights spend themselves preserving texture detail that was
//! never noisy — blurring the lighting it was meant to fix while it does so.

use crate::integrator::Film;
use glam::Vec3;

/// The guide channels, one per pixel, as the renderer records them.
#[derive(Clone, Debug, Default)]
pub struct Guides {
    pub albedo: Vec<Vec3>,
    pub normal: Vec<Vec3>,
    pub depth: Vec<f32>,
    /// BVH node visits per ray. Not used by the filter — it is here because the
    /// diagnostic modes read the same accumulated channels, and splitting them
    /// into two structs would mean two renders to see both.
    pub steps: Vec<f32>,
}

/// How hard to filter.
#[derive(Clone, Copy, Debug)]
pub struct DenoiseParams {
    /// À-trous passes. Each doubles the tap spacing, so `n` passes reach about
    /// `2^(n+1)` pixels.
    pub iterations: u32,
    /// Exponent on `max(0, dot(n, n'))`. Higher is stricter about normals.
    pub sigma_normal: f32,
    /// Relative depth tolerance. Compared against the centre depth, so it works
    /// at any scene scale.
    pub sigma_depth: f32,
    /// Luminance tolerance, in multiples of the *measured* local deviation.
    pub sigma_luminance: f32,
    /// Relative albedo tolerance, as a fraction of the centre pixel's albedo.
    ///
    /// Demodulation removes *texture* from the filter's view; this removes
    /// *material boundaries*, which is a different problem. The case that forced
    /// it: an area light and the ceiling it is set into share a normal and a
    /// depth, so neither of those weights separates them, and the filter smeared
    /// a radiance-18 emitter across a radiance-0.2 ceiling. Their albedos differ
    /// by a factor of twenty-five.
    pub sigma_albedo: f32,
}

impl Default for DenoiseParams {
    fn default() -> Self {
        Self {
            iterations: 4,
            sigma_normal: 64.0,
            sigma_depth: 0.08,
            sigma_luminance: 4.0,
            sigma_albedo: 0.1,
        }
    }
}

/// The 5-tap B-spline kernel, `[1, 4, 6, 4, 1] / 16`.
///
/// A B-spline rather than a box or a Gaussian because repeated convolution of it
/// converges to a Gaussian quickly, which is what makes the à-trous
/// approximation of a wide blur a good one after only a few passes.
const KERNEL: [f32; 5] = [1.0 / 16.0, 1.0 / 4.0, 3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0];

/// Estimate the local luminance deviation, which sets the luminance weight's
/// scale.
///
/// Without this the luminance test is meaningless: an absolute threshold either
/// rejects every tap in a noisy image (filtering nothing) or accepts every tap
/// in a clean one (blurring everything). Measuring the deviation in a small
/// neighbourhood makes the test "is this tap further away than the noise here
/// would explain", which is the question actually being asked.
fn local_deviation(data: &[Vec3], width: u32, height: u32, x: u32, y: u32) -> f32 {
    let (w, h) = (width as i32, height as i32);
    let (mut sum, mut sum_sq, mut n) = (0.0f32, 0.0f32, 0.0f32);
    for dy in -1..=1i32 {
        for dx in -1..=1i32 {
            let (px, py) = (x as i32 + dx, y as i32 + dy);
            if px < 0 || py < 0 || px >= w || py >= h {
                continue;
            }
            let l = crate::math::luminance(data[(py * w + px) as usize]);
            sum += l;
            sum_sq += l * l;
            n += 1.0;
        }
    }
    let mean = sum / n;
    (sum_sq / n - mean * mean).max(0.0).sqrt()
}

/// Run the filter. `film` is the averaged radiance; the result is a new image.
///
/// The input is never modified — this is a display-time product, and the
/// accumulation it came from still converges to the unbiased answer.
pub fn denoise(film: &Film, guides: &Guides, params: &DenoiseParams) -> Film {
    let (w, h) = (film.width as i32, film.height as i32);
    let n = (w * h) as usize;
    if n == 0 || guides.albedo.len() != n {
        return film.clone();
    }

    // Demodulate: filter the lighting, not the lighting times the texture.
    let mut current: Vec<Vec3> = (0..n)
        .map(|i| film.data[i] / guides.albedo[i].max(Vec3::splat(1.0e-3)))
        .collect();
    let mut next = current.clone();

    // Measured on the **demodulated** signal, because that is the quantity the
    // luminance weight compares. Measuring it on the modulated image instead is
    // wrong in exactly the place it matters most: an area light sits at radiance
    // 16 next to a ceiling at 0.2, so its local deviation is enormous, the
    // tolerance derived from it admits every neighbour, and the filter smears
    // the light across the ceiling. Measured before this was fixed, the Cornell
    // box lost 39% of its energy and the light dimmed from 18.4 to 4.7.
    //
    // Measured once, on the input, rather than per pass: recomputing it would
    // let it shrink as the filter smooths, and the filter would then keep
    // finding its own output clean enough and blur without limit.
    let deviation: Vec<f32> = (0..n)
        .map(|i| local_deviation(&current, film.width, film.height, i as u32 % film.width, i as u32 / film.width))
        .collect();

    for level in 0..params.iterations {
        let stride = 1i32 << level;
        for y in 0..h {
            for x in 0..w {
                let ci = (y * w + x) as usize;
                let cn = guides.normal[ci];
                let cd = guides.depth[ci];
                let cl = crate::math::luminance(current[ci]);
                let ca = crate::math::luminance(guides.albedo[ci]).max(1.0e-3);
                // The floor keeps a perfectly flat region from dividing by zero
                // and rejecting every tap.
                let lum_scale = params.sigma_luminance * deviation[ci].max(1.0e-4);

                let mut sum = Vec3::ZERO;
                let mut weight_sum = 0.0f32;
                for ky in 0..5i32 {
                    for kx in 0..5i32 {
                        let px = x + (kx - 2) * stride;
                        let py = y + (ky - 2) * stride;
                        if px < 0 || py < 0 || px >= w || py >= h {
                            continue;
                        }
                        let qi = (py * w + px) as usize;

                        // Normal: a dot product raised to a power. Sharp falloff,
                        // so adjacent faces of a box do not bleed together.
                        let nd = cn.dot(guides.normal[qi]).max(0.0);
                        let w_n = nd.powf(params.sigma_normal);

                        // Depth: relative to the centre, so one tolerance works
                        // at any scene scale. A background pixel has depth 0 and
                        // is excluded from a foreground pixel's neighbourhood by
                        // this alone.
                        let dd = (cd - guides.depth[qi]).abs();
                        let w_d = (-dd / (params.sigma_depth * cd.abs().max(1.0e-3))).exp();

                        // Luminance, against the noise actually present.
                        let dl = (cl - crate::math::luminance(current[qi])).abs();
                        let w_l = (-dl / lum_scale).exp();

                        // Albedo: different materials do not blend, however
                        // similar their geometry.
                        let da = (ca - crate::math::luminance(guides.albedo[qi])).abs();
                        let w_a = (-da / (params.sigma_albedo * ca)).exp();

                        let weight =
                            KERNEL[kx as usize] * KERNEL[ky as usize] * w_n * w_d * w_l * w_a;
                        sum += current[qi] * weight;
                        weight_sum += weight;
                    }
                }
                next[ci] = if weight_sum > 0.0 {
                    sum / weight_sum
                } else {
                    current[ci]
                };
            }
        }
        std::mem::swap(&mut current, &mut next);
    }

    let mut out = Film::new(film.width, film.height);
    for (i, px) in out.data.iter_mut().enumerate() {
        *px = current[i] * guides.albedo[i].max(Vec3::splat(1.0e-3));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat, noiseless image must survive the filter unchanged.
    ///
    /// The identity case. If a denoiser cannot leave a constant alone it is
    /// adding something of its own.
    #[test]
    fn a_constant_image_is_unchanged() {
        let (w, h) = (32u32, 32u32);
        let n = (w * h) as usize;
        let mut film = Film::new(w, h);
        for p in film.data.iter_mut() {
            *p = Vec3::new(0.4, 0.5, 0.6);
        }
        let guides = Guides {
            albedo: vec![Vec3::splat(0.7); n],
            normal: vec![Vec3::Z; n],
            depth: vec![5.0; n],
            steps: vec![0.0; n],
        };
        let out = denoise(&film, &guides, &DenoiseParams::default());
        for (i, p) in out.data.iter().enumerate() {
            assert!(
                (*p - film.data[i]).length() < 1e-4,
                "pixel {i} moved from {:?} to {p:?}",
                film.data[i]
            );
        }
    }

    /// Noise must go down.
    #[test]
    fn noise_is_reduced() {
        let (w, h) = (64u32, 64u32);
        let n = (w * h) as usize;
        let truth = Vec3::new(0.5, 0.5, 0.5);
        let mut rng = crate::rng::Rng::new(7, 0, 0);
        let mut film = Film::new(w, h);
        for p in film.data.iter_mut() {
            *p = truth * (0.4 + 1.2 * rng.next_f32());
        }
        let guides = Guides {
            albedo: vec![Vec3::splat(1.0); n],
            normal: vec![Vec3::Z; n],
            depth: vec![5.0; n],
            steps: vec![0.0; n],
        };
        let err = |f: &Film| -> f32 {
            (f.data.iter().map(|p| (*p - truth).length_squared()).sum::<f32>() / n as f32).sqrt()
        };
        let before = err(&film);
        let after = err(&denoise(&film, &guides, &DenoiseParams::default()));
        eprintln!("rms error {before:.4} -> {after:.4} ({:.1}x)", before / after);
        assert!(
            after * 4.0 < before,
            "denoising only reduced error from {before:.4} to {after:.4}"
        );
    }

    /// A normal discontinuity must not be crossed.
    ///
    /// Two halves of an image with opposing normals and very different
    /// brightness. A filter that ignores its guides averages them into a ramp;
    /// one that respects them leaves the step intact. Checked *at* the boundary,
    /// which is the only place it can fail.
    #[test]
    fn normals_stop_the_filter() {
        let (w, h) = (64u32, 64u32);
        let n = (w * h) as usize;
        let mut film = Film::new(w, h);
        let mut normal = vec![Vec3::Z; n];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize;
                let left = x < w / 2;
                film.data[i] = if left { Vec3::splat(0.9) } else { Vec3::splat(0.1) };
                normal[i] = if left { Vec3::X } else { Vec3::Y };
            }
        }
        let guides = Guides {
            albedo: vec![Vec3::splat(1.0); n],
            normal,
            depth: vec![5.0; n],
            steps: vec![0.0; n],
        };
        let out = denoise(&film, &guides, &DenoiseParams::default());
        let at = |x: u32, y: u32| out.data[(y * w + x) as usize].x;
        let (l, r) = (at(w / 2 - 1, h / 2), at(w / 2, h / 2));
        eprintln!("across the normal edge: {l:.4} | {r:.4}");
        assert!(
            l > 0.85 && r < 0.15,
            "the filter bled across a normal discontinuity: {l:.4} | {r:.4}. The \
             two sides are 0.9 and 0.1 and their normals are perpendicular, so \
             nothing should cross."
        );
    }

    /// A depth discontinuity must not be crossed either.
    #[test]
    fn depth_stops_the_filter() {
        let (w, h) = (64u32, 64u32);
        let n = (w * h) as usize;
        let mut film = Film::new(w, h);
        let mut depth = vec![5.0f32; n];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize;
                let near = x < w / 2;
                film.data[i] = if near { Vec3::splat(0.9) } else { Vec3::splat(0.1) };
                depth[i] = if near { 2.0 } else { 40.0 };
            }
        }
        let guides = Guides {
            albedo: vec![Vec3::splat(1.0); n],
            normal: vec![Vec3::Z; n],
            depth,
            steps: vec![0.0; n],
        };
        let out = denoise(&film, &guides, &DenoiseParams::default());
        let at = |x: u32, y: u32| out.data[(y * w + x) as usize].x;
        let (l, r) = (at(w / 2 - 1, h / 2), at(w / 2, h / 2));
        eprintln!("across the depth edge: {l:.4} | {r:.4}");
        assert!(
            l > 0.85 && r < 0.15,
            "the filter bled across a depth discontinuity: {l:.4} | {r:.4}"
        );
    }

    /// Demodulation must preserve texture the filter would otherwise smear.
    ///
    /// A checkerboard *albedo* under flat lighting. The radiance is the
    /// checkerboard, and a filter that ran on it directly would average it
    /// toward grey; running on `radiance / albedo` sees a constant, filters
    /// nothing, and multiplies the checker back untouched.
    #[test]
    fn demodulation_preserves_texture() {
        let (w, h) = (32u32, 32u32);
        let n = (w * h) as usize;
        let mut film = Film::new(w, h);
        let mut albedo = vec![Vec3::ONE; n];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize;
                let dark = (x + y) % 2 == 0;
                let a = if dark { 0.05 } else { 0.9 };
                albedo[i] = Vec3::splat(a);
                // Flat incoming light of 1.0, so radiance is just the albedo.
                film.data[i] = Vec3::splat(a);
            }
        }
        let guides = Guides {
            albedo,
            normal: vec![Vec3::Z; n],
            depth: vec![5.0; n],
            steps: vec![0.0; n],
        };
        let out = denoise(&film, &guides, &DenoiseParams::default());
        let mut worst = 0.0f32;
        for i in 0..n {
            worst = worst.max((out.data[i] - film.data[i]).length());
        }
        eprintln!("largest change to a checkerboard: {worst:.5}");
        assert!(
            worst < 0.02,
            "demodulation should have made the checkerboard invisible to the \
             filter, but a pixel moved by {worst:.4}"
        );
    }

    /// Zero iterations must be exactly the identity.
    #[test]
    fn zero_iterations_is_a_no_op() {
        let (w, h) = (16u32, 16u32);
        let n = (w * h) as usize;
        let mut rng = crate::rng::Rng::new(3, 0, 0);
        let mut film = Film::new(w, h);
        for p in film.data.iter_mut() {
            *p = Vec3::splat(rng.next_f32());
        }
        let guides = Guides {
            albedo: vec![Vec3::splat(0.6); n],
            normal: vec![Vec3::Z; n],
            depth: vec![3.0; n],
            steps: vec![0.0; n],
        };
        let out = denoise(
            &film,
            &guides,
            &DenoiseParams {
                iterations: 0,
                ..Default::default()
            },
        );
        for i in 0..n {
            assert!(
                (out.data[i] - film.data[i]).length() < 1e-5,
                "pixel {i} changed with zero iterations"
            );
        }
    }
}
