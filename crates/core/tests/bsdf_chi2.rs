//! Build step 7: chi-squared sampling tests for every BSDF lobe.
//!
//! Each test draws several hundred thousand directions from `sample()`,
//! histograms them over the hemisphere, integrates `pdf()` over the same bins,
//! and asks whether the difference is larger than chance allows.
//!
//! This is the check that `sample()` and `pdf()` describe the *same*
//! distribution. Nothing else finds a mismatch between them: the image looks
//! fine, energy conservation can still hold, and CPU and GPU agree because they
//! are wrong together. The estimator `f * cos / pdf` is simply weighted wrongly,
//! and every render converges confidently to the wrong answer.
//!
//! # Significance level and multiple testing
//!
//! Each test runs at alpha = 0.01, so a *correct* sampler fails one time in a
//! hundred by chance. There are dozens of configurations here, so at least one
//! spurious failure would be likely on every run — which would train everyone to
//! ignore the suite.
//!
//! Two mitigations, both standard:
//!
//! * seeds are fixed, so a passing run stays passing and any failure is
//!   reproducible rather than intermittent;
//! * within a test that sweeps many configurations the threshold is
//!   Sidak-corrected for the number of configurations, so the *family-wise*
//!   false-positive rate stays at 1%.

use glam::Vec3;
use pt_core::bsdf::{self, conductors, surface::Surface};
use pt_core::chi2::chi2_test;
use pt_core::gpu_layout::GpuMaterial;

/// Fine enough that a glossy lobe spans many cells. Bins outside the lobe cost
/// nothing — they are pooled away — so there is no reason to be stingy.
const THETA_BINS: usize = 128;
const PHI_BINS: usize = 256;
const ATTEMPTS: usize = 600_000;

/// Sidak correction: to hold a family-wise error rate of `alpha` across `n`
/// tests, each individual test uses `1 - (1 - alpha)^(1/n)`.
fn sidak(alpha: f64, n: usize) -> f64 {
    1.0 - (1.0 - alpha).powf(1.0 / n as f64)
}

fn wo_from_cos(cos_o: f32) -> Vec3 {
    let sin_o = (1.0 - cos_o * cos_o).max(0.0).sqrt();
    Vec3::new(sin_o, 0.0, cos_o)
}

#[test]
fn diffuse_lobe() {
    let albedo = Vec3::splat(0.8);
    let r = chi2_test(
        |rng| Some(bsdf::diffuse::sample(albedo, rng.next_vec2()).wi),
        bsdf::diffuse::pdf,
        THETA_BINS,
        PHI_BINS,
        ATTEMPTS,
        0xD1FF,
    );
    eprintln!("diffuse: {}", r.summary());
    assert!(
        r.passed(0.01),
        "cosine-weighted diffuse sampler: {}",
        r.summary()
    );
}

/// The GGX specular lobe, swept over roughness and viewing angle.
///
/// Roughness starts at 0.1. Below that the lobe is narrower than a histogram
/// cell — at roughness 0.05 it spans about 0.6 degrees against a 0.7-degree bin
/// — so a uniform grid has nothing to resolve and the test degenerates to one or
/// two degrees of freedom. Near-specular lobes are covered instead by the
/// pairwise `sample_and_pdf_agree` check and by
/// `specular_weight_is_well_conditioned` on the GPU side.
#[test]
fn specular_lobe_across_roughness_and_angle() {
    let roughnesses = [0.1f32, 0.2, 0.35, 0.5, 0.75, 1.0];
    let cosines = [0.95f32, 0.7, 0.45, 0.2];
    let alpha_level = sidak(0.01, roughnesses.len() * cosines.len());

    let mut worst_p = 1.0f64;
    let mut worst_cfg = String::new();
    for (ri, &roughness) in roughnesses.iter().enumerate() {
        for (ci, &cos_o) in cosines.iter().enumerate() {
            let alpha = bsdf::roughness_to_alpha(roughness);
            let wo = wo_from_cos(cos_o);
            let r = chi2_test(
                |rng| {
                    let s = bsdf::specular::sample_single(alpha, Vec3::ONE, wo, rng.next_vec2());
                    if s.is_valid() {
                        Some(s.wi)
                    } else {
                        None
                    }
                },
                |wi| bsdf::specular::pdf(alpha, wo, wi),
                THETA_BINS,
                PHI_BINS,
                ATTEMPTS,
                0x66_0000 + (ri * 16 + ci) as u32,
            );
            eprintln!(
                "ggx roughness {roughness:<5} cos_o {cos_o:<5}: {}",
                r.summary()
            );
            if r.p_value < worst_p {
                worst_p = r.p_value;
                worst_cfg = format!("roughness {roughness}, cos_o {cos_o}");
            }
            assert!(
                r.passed(alpha_level),
                "GGX visible-normal sampler disagrees with its pdf at roughness {roughness}, \
                 cos_o {cos_o}:\n  {}",
                r.summary()
            );
            // The rejection rate must match the probability mass the pdf does
            // *not* account for. This catches a sampler that silently discards
            // directions the pdf still claims it produces.
            let acceptance = r.accepted as f64 / r.attempts as f64;
            assert!(
                (acceptance - r.pdf_integral).abs() < 5e-3,
                "roughness {roughness}, cos_o {cos_o}: acceptance {acceptance:.4} but the pdf \
                 accounts for {:.4} of the probability mass",
                r.pdf_integral
            );
        }
    }
    eprintln!("ggx worst configuration: {worst_cfg} (p = {worst_p:.4})");
}

/// The combined two-lobe BSDF, which is where the subtle mistakes live: the
/// selection probability and the mixture density both have to be right.
#[test]
fn combined_surface_bsdf() {
    let materials: Vec<(&str, GpuMaterial)> = vec![
        ("diffuse", GpuMaterial::diffuse(Vec3::splat(0.7))),
        (
            "glossy plastic",
            GpuMaterial::glossy(Vec3::new(0.2, 0.4, 0.8), 0.25),
        ),
        (
            "rough plastic",
            GpuMaterial::glossy(Vec3::new(0.8, 0.3, 0.2), 0.7),
        ),
        ("metal", GpuMaterial::metal(Vec3::new(0.95, 0.8, 0.4), 0.3)),
        (
            "rough metal",
            GpuMaterial::metal(Vec3::new(0.95, 0.8, 0.4), 0.85),
        ),
        (
            "copper",
            GpuMaterial::conductor(conductors::COPPER_ETA, conductors::COPPER_K, 0.2),
        ),
        (
            "gold",
            GpuMaterial::conductor(conductors::GOLD_ETA, conductors::GOLD_K, 0.5),
        ),
    ];
    let cosines = [0.9f32, 0.5, 0.25];
    let alpha_level = sidak(0.01, materials.len() * cosines.len());

    for (mi, (name, m)) in materials.iter().enumerate() {
        let s = Surface::from_material(m, true);
        for (ci, &cos_o) in cosines.iter().enumerate() {
            let wo = wo_from_cos(cos_o);
            let r = chi2_test(
                |rng| {
                    let u_lobe = rng.next_f32();
                    let u = rng.next_vec2();
                    let smp = bsdf::surface::sample(&s, wo, u_lobe, u);
                    if smp.is_valid() {
                        Some(smp.wi)
                    } else {
                        None
                    }
                },
                |wi| bsdf::surface::pdf(&s, wo, wi),
                THETA_BINS,
                PHI_BINS,
                ATTEMPTS,
                0x77_0000 + (mi * 16 + ci) as u32,
            );
            eprintln!("{name:<16} cos_o {cos_o:<5}: {}", r.summary());
            assert!(
                r.passed(alpha_level),
                "combined BSDF sampler disagrees with its pdf for {name} at cos_o {cos_o}:\n  {}",
                r.summary()
            );
        }
    }
}

/// **The bug this test exists to catch.**
///
/// A two-lobe BSDF must report the *mixture* density,
/// `p_spec * pdf_spec + p_diff * pdf_diff`, not the density of whichever lobe
/// happened to be chosen. Using only the chosen lobe's density is an easy and
/// very common mistake: it produces images that look entirely reasonable,
/// conserves energy, and agrees between CPU and GPU.
///
/// Here that mistake is injected deliberately, to prove the test would find it.
/// If this assertion ever fails — that is, if the broken version starts
/// *passing* — the harness has lost its sensitivity and every other
/// chi-squared result in this file is worthless.
#[test]
fn chi2_catches_a_single_lobe_pdf() {
    let m = GpuMaterial::glossy(Vec3::splat(0.6), 0.4);
    let s = Surface::from_material(&m, true);
    let wo = wo_from_cos(0.6);
    let alpha = bsdf::roughness_to_alpha(s.roughness);

    let r = chi2_test(
        |rng| {
            let u_lobe = rng.next_f32();
            let u = rng.next_vec2();
            let smp = bsdf::surface::sample(&s, wo, u_lobe, u);
            if smp.is_valid() {
                Some(smp.wi)
            } else {
                None
            }
        },
        // The injected error: only the specular lobe's density, as though the
        // diffuse lobe did not exist.
        |wi| bsdf::specular::pdf(alpha, wo, wi),
        THETA_BINS,
        PHI_BINS,
        ATTEMPTS,
        0x8AD_F00D,
    );
    eprintln!("single-lobe pdf (deliberately wrong): {}", r.summary());
    assert!(
        !r.passed(0.01),
        "the chi-squared test accepted a BSDF reporting only one lobe's density. \
         The harness is not sensitive enough for the other tests in this file to mean \
         anything:\n  {}",
        r.summary()
    );
}

/// A second injected error, of a different kind: the lobe *selection* probability
/// is wrong while both lobe densities are individually correct.
///
/// Subtler than the one above — the reported density is still a proper mixture,
/// just mixed in the wrong proportion — and it is exactly what happens when the
/// sampling and evaluation sides disagree about how often each lobe is chosen.
#[test]
fn chi2_catches_a_wrong_lobe_probability() {
    let m = GpuMaterial::glossy(Vec3::splat(0.5), 0.3);
    let s = Surface::from_material(&m, true);
    let wo = wo_from_cos(0.75);
    let alpha = bsdf::roughness_to_alpha(s.roughness);

    let r = chi2_test(
        |rng| {
            let u_lobe = rng.next_f32();
            let u = rng.next_vec2();
            let smp = bsdf::surface::sample(&s, wo, u_lobe, u);
            if smp.is_valid() {
                Some(smp.wi)
            } else {
                None
            }
        },
        |wi| {
            // A fixed 50/50 split instead of the Fresnel-weighted one the
            // sampler actually uses.
            0.5 * bsdf::specular::pdf(alpha, wo, wi) + 0.5 * bsdf::diffuse::pdf(wi)
        },
        THETA_BINS,
        PHI_BINS,
        ATTEMPTS,
        0x8AD_BEEF,
    );
    eprintln!(
        "wrong lobe probability (deliberately wrong): {}",
        r.summary()
    );
    assert!(
        !r.passed(0.01),
        "the chi-squared test accepted a mixture density with the wrong lobe weights:\n  {}",
        r.summary()
    );
}

/// Mirror a transmitted direction into the upper hemisphere.
///
/// The chi-squared harness bins the upper hemisphere only. Negating `z` maps the
/// transmitted lobe onto it, which preserves solid angle exactly — so the
/// histogram, the pdf integral and the statistic are all unchanged in meaning.
fn mirror(w: Vec3) -> Vec3 {
    Vec3::new(w.x, w.y, -w.z)
}

/// Transmission is binned twice as finely as reflection, on both axes.
///
/// Refraction compresses angles by roughly `1 / eta`, so a transmitted lobe is
/// sharper than the reflection lobe of the same roughness and varies faster
/// across a histogram cell. The *expected* counts then carry a quadrature error
/// large enough to read as a sampler failure — at roughness 0.5 and 76 degrees,
/// chi2/dof was 1.097 at 128x256 and fell to 1.004 at 512x1024 while the pdf's
/// total mass stayed put at 0.73581, which is the signature of an
/// under-resolved integral rather than a mismatched distribution.
///
/// 256x512 is where every configuration here clears the threshold with room to
/// spare. Doubling again is affordable if the grid ever grows harsher.
const T_THETA_BINS: usize = THETA_BINS * 2;
const T_PHI_BINS: usize = PHI_BINS * 2;

/// Below this share of samples, a lobe is not worth testing.
///
/// Past the critical angle a dielectric reflects essentially everything, and the
/// transmitted lobe collapses to a thin sliver near the TIR boundary carrying a
/// ten-thousandth of the energy. A chi-squared test there has no power — and
/// worse, the adaptive quadrature that computes the expected counts steps right
/// over a feature that narrow, so the test reports a confident failure about a
/// lobe that barely exists. Measured at roughness 0.05 leaving glass at 76
/// degrees: 21 samples out of 600 000 against an integrated expectation of 0.2.
///
/// What governs image correctness in that regime is the *reflection* lobe, which
/// `glass_combined_bsdf` covers, so nothing goes unchecked by skipping here.
const MIN_LOBE_SHARE: f64 = 0.01;

/// Fraction of samples landing in the transmitted hemisphere.
fn transmitted_share(s: &Surface, wo: Vec3) -> f64 {
    let mut rng = pt_core::rng::Rng::new(0x5EE, 0, 0);
    let n = 20_000;
    let hits = (0..n)
        .filter(|_| {
            let smp = bsdf::surface::sample(s, wo, rng.next_f32(), rng.next_vec2());
            smp.is_valid() && smp.wi.z < 0.0
        })
        .count();
    hits as f64 / n as f64
}

/// The transmission lobe of a rough dielectric, against its own pdf.
///
/// Run for both directions across the interface. Entering and leaving are not
/// the same distribution — the relative index is inverted, and leaving a dense
/// medium has a critical angle beyond which nothing transmits at all — so a sign
/// error in the half-vector or the Jacobian can easily pass one and fail the
/// other.
#[test]
fn dielectric_transmission_lobe() {
    let roughnesses = [0.05f32, 0.2, 0.5];
    let cosines = [0.95f32, 0.6, 0.25];
    let sides = [("entering", true), ("leaving", false)];
    let alpha_level = sidak(0.01, roughnesses.len() * cosines.len() * sides.len());

    for (si, (side, front)) in sides.iter().enumerate() {
        for (ri, &roughness) in roughnesses.iter().enumerate() {
            let m = GpuMaterial::glass(Vec3::ONE, roughness, 1.5);
            let s = Surface::from_material(&m, *front);
            for (ci, &cos_o) in cosines.iter().enumerate() {
                let wo = wo_from_cos(cos_o);
                let share = transmitted_share(&s, wo);
                if share < MIN_LOBE_SHARE {
                    eprintln!(
                        "glass {side:<8} r {roughness:<5} cos_o {cos_o:<5}: skipped, only \
                         {:.4}% of samples transmit (past the critical angle)",
                        100.0 * share
                    );
                    continue;
                }
                let r = chi2_test(
                    |rng| {
                        let smp = bsdf::surface::sample(&s, wo, rng.next_f32(), rng.next_vec2());
                        // Only the transmitted half of the distribution. The
                        // harness's pdf-integral check then also verifies the
                        // lobe's *weight*: the mass the pdf assigns below the
                        // surface must equal the fraction of samples that go
                        // there.
                        if smp.is_valid() && smp.wi.z < 0.0 {
                            Some(mirror(smp.wi))
                        } else {
                            None
                        }
                    },
                    |w| bsdf::surface::pdf(&s, wo, mirror(w)),
                    T_THETA_BINS,
                    T_PHI_BINS,
                    ATTEMPTS,
                    0x7A_0000 + ((si * 16 + ri) * 16 + ci) as u32,
                );
                eprintln!("glass {side:<8} r {roughness:<5} cos_o {cos_o:<5}: {}", r.summary());
                assert!(
                    r.passed(alpha_level),
                    "transmission sampler disagrees with its pdf ({side}, roughness \
                     {roughness}, cos_o {cos_o}):\n  {}\n\
                     Check, in order:\n\
                       1. the half-vector, normalize(wo + eta * wi), flipped into z > 0\n\
                       2. the Jacobian eta^2 |wi.h| / (wo.h + eta wi.h)^2\n\
                       3. whether `relative_ior` is inverted for the leaving case",
                    r.summary()
                );
            }
        }
    }
}

/// The whole BSDF of a glass material, both hemispheres at once.
///
/// The lobe-selection probabilities have to be consistent between `sample` and
/// `pdf` for this to pass, which the per-lobe tests above cannot check: each of
/// them sees only its own half.
#[test]
fn glass_combined_bsdf() {
    let cases = [
        ("smooth glass", GpuMaterial::glass(Vec3::ONE, 0.02, 1.5)),
        ("rough glass", GpuMaterial::glass(Vec3::ONE, 0.35, 1.5)),
        ("dense glass", GpuMaterial::glass(Vec3::ONE, 0.1, 2.4)),
    ];
    // Two angles rather than three: near-normal and well past the critical
    // angle for the leaving case. The grid is 2 materials-worth of work per
    // entry at the fine binning, and a third angle doubles the slowest test in
    // the suite without covering a regime the other two miss.
    let cosines = [0.9f32, 0.2];
    let alpha_level = sidak(0.01, cases.len() * cosines.len() * 2);

    for (mi, (name, m)) in cases.iter().enumerate() {
        for front in [true, false] {
            let s = Surface::from_material(m, front);
            for (ci, &cos_o) in cosines.iter().enumerate() {
                let wo = wo_from_cos(cos_o);
                // Reflection and transmission are tested as one distribution by
                // folding the lower hemisphere onto the upper one. That is only
                // sound because the two never overlap: a direction is either
                // reflected or transmitted, never both.
                for (hi, upper) in [true, false].into_iter().enumerate() {
                    if !upper && transmitted_share(&s, wo) < MIN_LOBE_SHARE {
                        continue;
                    }
                    let r = chi2_test(
                        |rng| {
                            let smp =
                                bsdf::surface::sample(&s, wo, rng.next_f32(), rng.next_vec2());
                            if !smp.is_valid() {
                                return None;
                            }
                            match (upper, smp.wi.z > 0.0) {
                                (true, true) => Some(smp.wi),
                                (false, false) => Some(mirror(smp.wi)),
                                _ => None,
                            }
                        },
                        |w| {
                            let wi = if upper { w } else { mirror(w) };
                            bsdf::surface::pdf(&s, wo, wi)
                        },
                        // Both halves get the finer grid. Using it only for the
                        // transmitted half was tried and was wrong: dense glass
                        // (IOR 2.4) at roughness 0.1 has a *reflection* lobe
                        // sharp enough to need it too, and under-resolving it
                        // read as a 3.9-sigma sampler failure.
                        T_THETA_BINS,
                        T_PHI_BINS,
                        ATTEMPTS,
                        0x6B_0000 + (((mi * 4 + hi) * 4 + usize::from(front)) * 16 + ci) as u32,
                    );
                    let half = if upper { "reflect" } else { "transmit" };
                    eprintln!(
                        "{name:<13} {} {half:<9} cos_o {cos_o:<5}: {}",
                        if front { "in " } else { "out" },
                        r.summary()
                    );
                    assert!(
                        r.passed(alpha_level),
                        "glass BSDF sampler disagrees with its pdf ({name}, front {front}, \
                         {half}, cos_o {cos_o}):\n  {}",
                        r.summary()
                    );
                }
            }
        }
    }
}
