//! Pearson's chi-squared goodness-of-fit test for BSDF samplers.
//!
//! # What this catches, and why nothing else does
//!
//! A BSDF has two halves that must agree: `sample()` draws directions, and
//! `pdf()` claims how densely it draws them. If they disagree, the estimator
//! `f * cos / pdf` is weighted wrongly and **every image is biased** — not noisy,
//! biased, converging confidently to the wrong answer.
//!
//! Nothing else in the test suite finds this. The image looks plausible. Energy
//! conservation can still hold. CPU and GPU still agree, because they are wrong
//! together. Even the pairwise check "does `pdf()` return what `sample()`
//! reported for this direction" passes, because that only asks the two functions
//! to be *consistent about a number*, not that the number describes the actual
//! density of samples.
//!
//! The only way to know is to draw a great many samples, histogram where they
//! land, integrate the claimed density over the same bins, and ask whether the
//! difference is larger than chance allows.
//!
//! # The test
//!
//! Bin the hemisphere, count samples per bin, integrate `pdf` over each bin to
//! get an expected count, and form
//!
//! ```text
//!   chi^2 = sum over bins of (observed - expected)^2 / expected
//! ```
//!
//! Under the null hypothesis (sampler matches pdf) this follows a chi-squared
//! distribution, so the p-value says how surprising the discrepancy is. A
//! *small* p-value is a failure: it means the data would rarely look like this
//! if the two agreed.
//!
//! Two details that matter:
//!
//! * **Bins are uniform in `theta`**, not in `cos(theta)`. Equal-solid-angle
//!   bins sound more natural but they are wide in angle near the pole, and a
//!   glossy lobe is narrow in angle — with equal-area bins an entire GGX lobe
//!   can land in one cell, leaving nothing to test. Uniform `theta` resolves the
//!   lobe; the `sin(theta)` weight is handled by the quadrature.
//! * **Samples that fail are still counted in the denominator.** A VNDF sample
//!   whose reflection goes below the surface is rejected, and `pdf` integrates
//!   to less than 1 over the hemisphere by exactly that probability. Normalising
//!   by *attempts* rather than successes turns that into an additional assertion:
//!   the rejection rate must match the missing probability mass.

use crate::rng::Rng;
use glam::Vec3;

/// Minimum expected count per bin before the chi-squared approximation is
/// trustworthy. Bins below this are pooled with their neighbours — the standard
/// remedy, and necessary here because a glossy lobe leaves most of the
/// hemisphere nearly empty.
const MIN_EXPECTED: f64 = 5.0;

#[derive(Clone, Debug)]
pub struct Chi2Result {
    pub statistic: f64,
    pub degrees_of_freedom: usize,
    pub p_value: f64,
    /// Pooled bins actually used.
    pub bins_used: usize,
    /// Sampling attempts, including those that produced no direction.
    pub attempts: usize,
    pub accepted: usize,
    /// Total probability mass the pdf accounts for. Should equal the acceptance
    /// rate; a mismatch means the sampler rejects directions the pdf still
    /// claims, or vice versa.
    pub pdf_integral: f64,
    /// The pooled bin contributing most to the statistic, as
    /// (observed, expected, contribution).
    pub worst_bin: (f64, f64, f64),
}

impl Chi2Result {
    /// Conventional decision rule: reject the null hypothesis below `alpha`.
    pub fn passed(&self, alpha: f64) -> bool {
        self.p_value >= alpha
    }

    pub fn summary(&self) -> String {
        format!(
            "chi2 = {:.1}, dof = {}, p = {:.4}, {} pooled bins, \
             acceptance {:.4} vs pdf mass {:.4}, worst bin obs {:.0} exp {:.1} (+{:.1})",
            self.statistic,
            self.degrees_of_freedom,
            self.p_value,
            self.bins_used,
            self.accepted as f64 / self.attempts as f64,
            self.pdf_integral,
            self.worst_bin.0,
            self.worst_bin.1,
            self.worst_bin.2,
        )
    }
}

/// Run the test.
///
/// `sample` returns a direction in the local frame (+Z is the normal), or
/// `None` when the sampler legitimately produces nothing. `pdf` gives the
/// solid-angle density of that same sampler.
pub fn chi2_test<S, P>(
    mut sample: S,
    pdf: P,
    theta_bins: usize,
    phi_bins: usize,
    attempts: usize,
    seed: u32,
) -> Chi2Result
where
    S: FnMut(&mut Rng) -> Option<Vec3>,
    P: Fn(Vec3) -> f32,
{
    let mut observed = vec![0.0f64; theta_bins * phi_bins];
    let mut rng = Rng::new(seed, 0, 0);
    let mut accepted = 0usize;

    for _ in 0..attempts {
        let Some(w) = sample(&mut rng) else { continue };
        if w.z <= 0.0 {
            continue;
        }
        accepted += 1;
        let theta = w.z.clamp(-1.0, 1.0).acos();
        // atan2 returns (-pi, pi]; shift to [0, 2pi).
        let phi = w.y.atan2(w.x).rem_euclid(std::f32::consts::TAU);
        let ti = ((theta / (std::f32::consts::FRAC_PI_2) * theta_bins as f32) as usize)
            .min(theta_bins - 1);
        let pi_ = ((phi / std::f32::consts::TAU * phi_bins as f32) as usize).min(phi_bins - 1);
        observed[ti * phi_bins + pi_] += 1.0;
    }

    // Expected counts: integrate the pdf over each bin's solid angle.
    let mut expected = vec![0.0f64; theta_bins * phi_bins];
    let mut pdf_integral = 0.0f64;
    for ti in 0..theta_bins {
        let t0 = ti as f64 / theta_bins as f64 * std::f64::consts::FRAC_PI_2;
        let t1 = (ti + 1) as f64 / theta_bins as f64 * std::f64::consts::FRAC_PI_2;
        for pi_ in 0..phi_bins {
            let p0 = pi_ as f64 / phi_bins as f64 * std::f64::consts::TAU;
            let p1 = (pi_ + 1) as f64 / phi_bins as f64 * std::f64::consts::TAU;
            let integral = integrate_bin(&pdf, t0, t1, p0, p1);
            pdf_integral += integral;
            expected[ti * phi_bins + pi_] = integral * attempts as f64;
        }
    }

    // Pool adjacent bins until each has enough expected mass for the
    // chi-squared approximation to hold.
    let mut pooled: Vec<(f64, f64)> = Vec::new();
    let (mut po, mut pe) = (0.0f64, 0.0f64);
    for i in 0..observed.len() {
        po += observed[i];
        pe += expected[i];
        if pe >= MIN_EXPECTED {
            pooled.push((po, pe));
            po = 0.0;
            pe = 0.0;
        }
    }
    if pe > 0.0 || po > 0.0 {
        // Leftovers merge into the final bin rather than forming an
        // under-populated one of their own.
        if let Some(last) = pooled.last_mut() {
            last.0 += po;
            last.1 += pe;
        } else {
            pooled.push((po, pe));
        }
    }

    let mut statistic = 0.0f64;
    let mut worst = (0.0f64, 0.0f64, 0.0f64);
    for &(o, e) in &pooled {
        if e <= 0.0 {
            continue;
        }
        let d = o - e;
        let c = d * d / e;
        statistic += c;
        if c > worst.2 {
            worst = (o, e, c);
        }
    }

    // One degree of freedom is lost: the sampler's own total is not free, it is
    // pinned by the number of attempts.
    let dof = pooled.len().saturating_sub(1).max(1);
    let p_value = chi2_p_value(statistic, dof);

    Chi2Result {
        statistic,
        degrees_of_freedom: dof,
        p_value,
        bins_used: pooled.len(),
        attempts,
        accepted,
        pdf_integral,
        worst_bin: worst,
    }
}

/// Integrate `pdf` over a spherical rectangle, including the `sin(theta)`
/// Jacobian of the solid angle measure.
///
/// # Why this is adaptive
///
/// A fixed grid cannot handle the whole roughness range. A glossy lobe near
/// grazing incidence is narrower than a histogram cell — at roughness 0.1 and
/// `cos_theta_o = 0.2` the lobe spans about a degree, against cells of 0.7 by
/// 1.4 degrees — and its peak density is in the thousands. A fixed 8-point
/// Simpson rule underestimated that cell's integral by 2.7%, which the test
/// then read as a sampler producing too many samples there. An under-resolved
/// *expected* count is a false positive, and a false positive in a
/// statistical test is worse than no test, because it teaches you to ignore it.
///
/// So: compare a coarse and a fine estimate, and subdivide where they disagree.
/// Flat cells — the vast majority — terminate immediately.
///
/// The quadrature's accuracy is not taken on faith. `Chi2Result` reports the
/// total probability mass it found, and the callers assert that it matches the
/// sampler's measured acceptance rate; a systematically under-resolved integral
/// shows up there first.
fn integrate_bin<P>(pdf: &P, t0: f64, t1: f64, p0: f64, p1: f64) -> f64
where
    P: Fn(Vec3) -> f32,
{
    integrate_adaptive(pdf, t0, t1, p0, p1, 0)
}

/// Deepest subdivision, giving up to 4^6 sub-cells for a pathological bin.
const MAX_DEPTH: usize = 6;

/// Subdivisions taken *before* the agreement test is trusted.
///
/// Adaptive quadrature has a blind spot: when the integrand has structure finer
/// than the coarse grid, the coarse and fine estimates miss it in the same way
/// and their agreement reports convergence that has not happened. Neither a
/// tighter tolerance nor a deeper limit helps, because the loop never starts —
/// measured here, raising `MAX_DEPTH` from 6 to 10 changed no digit of any
/// result.
///
/// The transmission lobe is where this first bit. Refraction compresses angles
/// by roughly `1 / eta`, so a transmitted lobe is sharper than the reflection
/// lobe of the same roughness, and at roughness 0.2 the expected counts were off
/// by enough to read as a 4-sigma sampler failure. Forcing two levels of
/// subdivision costs 16 sub-cells on every bin and removes the blind spot.
const MIN_DEPTH: usize = 2;
/// Relative agreement required between the coarse and fine estimates.
///
/// Checked rather than guessed: tightening this to 1e-6 changes every p-value
/// in `chi2_seed_sweep` by less than the printed precision, so the quadrature is
/// converged here and any residual scatter in the results is sampling noise
/// rather than integration error.
const QUAD_REL_TOL: f64 = 1.0e-4;
/// Absolute floor, so a cell holding essentially no mass is never subdivided.
/// The total integral is order 1, so this is far below anything that matters.
const QUAD_ABS_TOL: f64 = 1.0e-12;

fn integrate_adaptive<P>(pdf: &P, t0: f64, t1: f64, p0: f64, p1: f64, depth: usize) -> f64
where
    P: Fn(Vec3) -> f32,
{
    let coarse = simpson_2d(pdf, t0, t1, p0, p1, 2);
    let fine = simpson_2d(pdf, t0, t1, p0, p1, 4);
    let converged = (fine - coarse).abs() <= QUAD_REL_TOL * fine.abs() + QUAD_ABS_TOL;
    if depth >= MAX_DEPTH || (converged && depth >= MIN_DEPTH) {
        return fine;
    }
    let tm = 0.5 * (t0 + t1);
    let pm = 0.5 * (p0 + p1);
    integrate_adaptive(pdf, t0, tm, p0, pm, depth + 1)
        + integrate_adaptive(pdf, tm, t1, p0, pm, depth + 1)
        + integrate_adaptive(pdf, t0, tm, pm, p1, depth + 1)
        + integrate_adaptive(pdf, tm, t1, pm, p1, depth + 1)
}

/// Composite Simpson over a spherical rectangle. `n` must be even.
fn simpson_2d<P>(pdf: &P, t0: f64, t1: f64, p0: f64, p1: f64, n: usize) -> f64
where
    P: Fn(Vec3) -> f32,
{
    let ht = (t1 - t0) / n as f64;
    let hp = (p1 - p0) / n as f64;
    let weight = |i: usize| -> f64 {
        if i == 0 || i == n {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        }
    };

    let mut sum = 0.0;
    for i in 0..=n {
        let theta = t0 + i as f64 * ht;
        let (sin_t, cos_t) = theta.sin_cos();
        let wt = weight(i);
        for j in 0..=n {
            let phi = p0 + j as f64 * hp;
            let (sin_p, cos_p) = phi.sin_cos();
            let w = Vec3::new((sin_t * cos_p) as f32, (sin_t * sin_p) as f32, cos_t as f32);
            // The sin(theta) factor is the solid angle Jacobian, not part of the
            // pdf: `pdf` is already a density with respect to solid angle.
            sum += wt * weight(j) * pdf(w) as f64 * sin_t;
        }
    }
    sum * ht * hp / 9.0
}

// ---------------------------------------------------------------------------
// The chi-squared distribution
// ---------------------------------------------------------------------------

/// Upper tail of the chi-squared distribution: `P(X > statistic)` for `dof`
/// degrees of freedom.
///
/// This is the regularized upper incomplete gamma function `Q(dof/2, x/2)`.
/// Implemented here rather than pulled in as a dependency — it is sixty lines
/// and the project's dependency list is short on purpose.
pub fn chi2_p_value(statistic: f64, dof: usize) -> f64 {
    if statistic <= 0.0 {
        return 1.0;
    }
    regularized_gamma_q(dof as f64 * 0.5, statistic * 0.5)
}

/// Log of the gamma function, Lanczos approximation (g = 7, n = 9).
///
/// Accurate to about 15 significant digits for positive arguments, which is far
/// more than a p-value needs, but it costs nothing and removes any doubt about
/// whether a borderline result is the test or the arithmetic.
pub fn ln_gamma(x: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection formula, so the series is only ever used where it converges.
        std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().abs().ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = C[0];
        let t = x + 7.5;
        for (i, &c) in C.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// Regularized upper incomplete gamma, `Q(a, x) = Gamma(a, x) / Gamma(a)`.
///
/// Two expansions, chosen by which converges: the series form for `x < a + 1`
/// and the continued fraction for `x >= a + 1`. Using either outside its region
/// converges slowly or not at all, which is the whole reason for the split.
pub fn regularized_gamma_q(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        1.0 - gamma_series(a, x)
    } else {
        gamma_continued_fraction(a, x)
    }
}

/// Lower regularized gamma `P(a, x)` by its series expansion.
fn gamma_series(a: f64, x: f64) -> f64 {
    const MAX_ITER: usize = 1000;
    const EPS: f64 = 1.0e-15;
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    for _ in 0..MAX_ITER {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * EPS {
            break;
        }
    }
    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

/// Upper regularized gamma `Q(a, x)` by the modified Lentz continued fraction.
fn gamma_continued_fraction(a: f64, x: f64) -> f64 {
    const MAX_ITER: usize = 1000;
    const EPS: f64 = 1.0e-15;
    const TINY: f64 = 1.0e-300;

    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=MAX_ITER {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + an / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ln_gamma_matches_known_values() {
        // Gamma(n) = (n-1)!
        for (x, expected) in [
            (1.0, 1.0f64),
            (2.0, 1.0),
            (3.0, 2.0),
            (5.0, 24.0),
            (10.0, 362_880.0),
        ] {
            let got = ln_gamma(x).exp();
            assert!(
                (got - expected).abs() / expected < 1e-10,
                "ln_gamma({x}).exp() = {got}, expected {expected}"
            );
        }
        // Gamma(1/2) = sqrt(pi)
        let half = ln_gamma(0.5).exp();
        assert!(
            (half - std::f64::consts::PI.sqrt()).abs() < 1e-12,
            "Gamma(0.5) = {half}"
        );
    }

    /// Published chi-squared critical values. If these are wrong, every p-value
    /// the suite reports is meaningless, so they are checked against a table
    /// rather than against the implementation's own output.
    #[test]
    fn chi2_p_values_match_published_tables() {
        // (dof, critical value at p = 0.05)
        let table_05 = [
            (1usize, 3.841),
            (2, 5.991),
            (3, 7.815),
            (5, 11.070),
            (10, 18.307),
            (20, 31.410),
            (50, 67.505),
            (100, 124.342),
        ];
        for (dof, critical) in table_05 {
            let p = chi2_p_value(critical, dof);
            assert!(
                (p - 0.05).abs() < 1e-3,
                "dof {dof}: p({critical}) = {p:.5}, expected 0.05"
            );
        }

        // ...and at p = 0.01, to pin the tail rather than one point.
        let table_01 = [
            (1usize, 6.635),
            (2, 9.210),
            (5, 15.086),
            (10, 23.209),
            (20, 37.566),
        ];
        for (dof, critical) in table_01 {
            let p = chi2_p_value(critical, dof);
            assert!(
                (p - 0.01).abs() < 1e-3,
                "dof {dof}: p({critical}) = {p:.5}, expected 0.01"
            );
        }
    }

    #[test]
    fn p_value_is_monotonic_and_bounded() {
        let mut prev = 1.1;
        for i in 0..200 {
            let stat = i as f64 * 0.5;
            let p = chi2_p_value(stat, 7);
            assert!((0.0..=1.0).contains(&p), "p = {p} for statistic {stat}");
            assert!(p <= prev + 1e-12, "p-value increased with the statistic");
            prev = p;
        }
        assert!(chi2_p_value(0.0, 7) == 1.0);
        assert!(chi2_p_value(1000.0, 7) < 1e-100);
    }

    /// A sampler that matches its pdf must pass. Uniform hemisphere sampling
    /// with the constant density `1/(2*pi)` is the simplest such pair, and it
    /// validates the harness itself — the binning, the quadrature, the pooling —
    /// before it is used to judge anything harder.
    #[test]
    fn harness_accepts_a_correct_sampler() {
        let result = chi2_test(
            |rng| {
                let u = rng.next_vec2();
                let z = u.x;
                let r = (1.0 - z * z).max(0.0).sqrt();
                let phi = std::f32::consts::TAU * u.y;
                Some(Vec3::new(r * phi.cos(), r * phi.sin(), z))
            },
            |_| 1.0 / std::f32::consts::TAU,
            32,
            64,
            500_000,
            0x0C_0FFEE,
        );
        eprintln!("uniform hemisphere: {}", result.summary());
        assert!(
            result.passed(0.01),
            "the harness rejected a correct sampler: {}",
            result.summary()
        );
        assert!(
            (result.pdf_integral - 1.0).abs() < 1e-3,
            "pdf integral {} should be 1 for a normalised density",
            result.pdf_integral
        );
    }

    /// ...and it must reject a wrong one. A test that cannot fail is not a test,
    /// and a subtly wrong pdf is exactly the thing this exists to find, so the
    /// error injected here is deliberately small: a density 4% too large.
    #[test]
    fn harness_rejects_a_mismatched_pdf() {
        let result = chi2_test(
            |rng| {
                let u = rng.next_vec2();
                let z = u.x;
                let r = (1.0 - z * z).max(0.0).sqrt();
                let phi = std::f32::consts::TAU * u.y;
                Some(Vec3::new(r * phi.cos(), r * phi.sin(), z))
            },
            // Claims a slightly higher density than the sampler actually has.
            |_| 1.04 / std::f32::consts::TAU,
            32,
            64,
            500_000,
            0x0BAD,
        );
        eprintln!("deliberately wrong pdf: {}", result.summary());
        assert!(
            !result.passed(0.01),
            "the harness accepted a pdf that is 4% too large: {}",
            result.summary()
        );
    }

    /// A sampler biased in *direction* rather than in normalisation must also be
    /// caught — this is the failure mode a total-energy check cannot see, because
    /// the mass is right and only its distribution is wrong.
    #[test]
    fn harness_rejects_a_directionally_biased_sampler() {
        let result = chi2_test(
            |rng| {
                let u = rng.next_vec2();
                // Skewed toward the pole, while the pdf still claims uniform.
                let z = u.x.powf(0.9);
                let r = (1.0 - z * z).max(0.0).sqrt();
                let phi = std::f32::consts::TAU * u.y;
                Some(Vec3::new(r * phi.cos(), r * phi.sin(), z))
            },
            |_| 1.0 / std::f32::consts::TAU,
            32,
            64,
            500_000,
            0x0B1A5,
        );
        eprintln!("directionally biased sampler: {}", result.summary());
        assert!(
            !result.passed(0.01),
            "the harness accepted a directionally biased sampler: {}",
            result.summary()
        );
    }
}
