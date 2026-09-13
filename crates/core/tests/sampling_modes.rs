//! Build steps 8 and 9: all three sampling strategies must estimate the *same*
//! integral.
//!
//! Two strategies that disagree about the answer is the classic next-event bug:
//! a missing measure conversion, or emission counted twice, produces an image
//! that looks entirely plausible on its own and is only revealed by rendering
//! the same scene the other way.
//!
//! # How to test "converges to the same image"
//!
//! Not by comparing two finite-sample renders directly. At equal sample counts
//! the two have wildly different noise — at 64 spp in a Cornell box, BSDF-only
//! is barely readable while NEE is clean — so a per-pixel difference is
//! dominated by variance and says nothing about whether the means agree.
//!
//! Two things are asserted instead:
//!
//! * **The mean radiance agrees**, which converges far faster than any pixel
//!   because it averages over the whole image as well as over samples.
//! * **The per-pixel RMSE between them falls as 1/sqrt(N)**. This is the
//!   stronger statement: if the two converged to *different* images, their
//!   difference would stop shrinking and plateau at the systematic offset. A
//!   difference that keeps halving with every quadrupling of samples is pure
//!   noise around a common answer.

use pt_core::image::compare;
use pt_core::integrator::{render, RenderParams, SamplingMode};
use pt_core::scenes;

#[test]
fn all_three_modes_converge_to_the_same_image() {
    let def = scenes::cornell_box();
    let sample_counts = [512u32, 2048, 8192];

    let mut rmse = Vec::new();
    let mut last_ratio = 0.0f64;
    let mut mis_ratio = 0.0f64;
    for &spp in &sample_counts {
        let base = RenderParams {
            width: 32,
            height: 32,
            samples: spp,
            max_depth: 6,
            ..Default::default()
        };
        let a = render(
            &def,
            &RenderParams {
                sampling: SamplingMode::BsdfOnly,
                ..base
            },
        );
        let b = render(
            &def,
            &RenderParams {
                sampling: SamplingMode::NeeOnly,
                ..base
            },
        );
        let c = render(
            &def,
            &RenderParams {
                sampling: SamplingMode::Mis,
                ..base
            },
        );
        let d = compare(&a, &b).expect("same size");
        let dm = compare(&a, &c).expect("same size");
        last_ratio = d.mean_b / d.mean_a;

        mis_ratio = dm.mean_b / dm.mean_a;
        eprintln!(
            "{spp:>5} spp: mean bsdf {:.6}, nee {:.6}, mis {:.6}, rmse {:.5}",
            d.mean_a, d.mean_b, dm.mean_b, d.rmse
        );
        rmse.push(d.rmse);
    }

    // Mean radiance at the highest sample count: the direct check that no
    // strategy is losing or inventing energy. Emission counted twice under NEE
    // would put this near 2.0; a missing d^2/cos conversion would make it drift
    // with scene scale.
    //
    // Asserted only at the top of the sweep, because BSDF-only converges slowly
    // here — at 512 spp it is still 1.2% away from its own limit, which says
    // nothing about whether the strategies agree.
    assert!(
        (last_ratio - 1.0).abs() < 5e-3,
        "BSDF-only and NEE disagree about total energy by {:.4}% — they are not \
         estimating the same integral",
        (last_ratio - 1.0).abs() * 100.0
    );

    // MIS must agree with BSDF-only too, not merely with NEE. It is possible to
    // get the weights wrong in a way that reproduces one strategy exactly — a
    // weight stuck at 1 for light samples and 0 for BSDF ones is just NEE
    // wearing a different name, and would sail through a NEE-only comparison.
    //
    // On this scene MIS and NEE do agree to six digits, and that is correct
    // rather than suspicious: the light pdf is roughly twenty times the BSDF pdf
    // from a diffuse surface, so light samples take weight 0.998 and the BSDF
    // path takes 0.002, and the two reconstruct the same total. The check that
    // MIS is doing real work lives in `mis_beats_both_strategies`, on a scene
    // built so neither strategy dominates.
    assert!(
        (mis_ratio - 1.0).abs() < 5e-3,
        "MIS and BSDF-only disagree about total energy by {:.4}%",
        (mis_ratio - 1.0).abs() * 100.0
    );

    // Quadrupling the samples must halve the difference. A systematic offset
    // would show up as a floor here even while the mean still looked fine,
    // because a bias that is positive in some pixels and negative in others
    // cancels in the mean but not in the RMSE.
    for w in rmse.windows(2) {
        let decay = w[1] / w[0];
        assert!(
            (0.40..0.68).contains(&decay),
            "the difference between the two modes fell by only {decay:.3} when samples \
             quadrupled (expected about 0.5). A plateau means they converge to different \
             images. RMSE sequence: {rmse:?}"
        );
    }
}

/// NEE must be dramatically *better* than BSDF sampling for a small area light.
///
/// Agreement alone is not enough — an NEE implementation that silently fell back
/// to BSDF sampling would agree perfectly and be useless. This measures the
/// variance reduction that is the entire reason next event estimation exists.
#[test]
fn nee_reduces_variance_for_a_small_light() {
    let def = scenes::cornell_box();
    let base = RenderParams {
        width: 48,
        height: 48,
        samples: 64,
        max_depth: 6,
        ..Default::default()
    };

    // Estimate each mode's noise by rendering twice with different seeds; the
    // difference between two independent runs is sqrt(2) times one run's noise.
    //
    // Measured with the *relative* metric, not RMSE. RMSE is dominated by the
    // brightest pixels — the emitter itself renders at radiance 18.4 and is
    // noise-free in both modes, and the directly lit walls are bright and
    // comparatively clean — so it badly under-weights the dark regions where
    // next event estimation does nearly all of its work. Measured on the same
    // renders, RMSE reports a 1.7x improvement and the relative metric reports
    // 8.6x; the images make it obvious which one describes what you see.
    let noise = |mode: SamplingMode| -> f64 {
        let a = render(
            &def,
            &RenderParams {
                sampling: mode,
                ..base
            },
        );
        let b = render(
            &def,
            &RenderParams {
                sampling: mode,
                frame_seed: base.frame_seed ^ 0xABCD,
                ..base
            },
        );
        compare(&a, &b).expect("same size").mean_rel / std::f64::consts::SQRT_2
    };

    let bsdf = noise(SamplingMode::BsdfOnly);
    let nee = noise(SamplingMode::NeeOnly);
    eprintln!(
        "relative noise at 64 spp: bsdf {bsdf:.5}, nee {nee:.5}, reduction {:.1}x",
        bsdf / nee
    );
    assert!(
        nee < bsdf * 0.25,
        "next event estimation reduced noise only {:.2}x (expected at least 4x for a \
         small area light). Is it actually connecting to the light?",
        bsdf / nee
    );
}

/// Emission from a *directly visible* light must survive under NEE.
///
/// NEE suppresses emission on every bounce after the first, because the shadow
/// ray from the previous vertex already accounted for it. The camera ray is the
/// exception — nothing preceded it — and forgetting that renders the light
/// fixture itself black while the room it illuminates looks perfect.
#[test]
fn the_light_itself_is_visible_under_nee() {
    let def = scenes::cornell_box();
    let params = RenderParams {
        width: 128,
        height: 128,
        samples: 64,
        max_depth: 4,
        sampling: SamplingMode::NeeOnly,
        ..Default::default()
    };
    let film = render(&def, &params);

    // Find the brightest pixel rather than hardcoding where the light appears.
    // An earlier version guessed the coordinates, guessed them wrong by four
    // pixels, and reported a failure that had nothing to do with the renderer.
    let mut brightest = 0.0f32;
    let mut at = (0u32, 0u32);
    for y in 0..film.height {
        for x in 0..film.width {
            let v = film.pixel(x, y).x;
            if v > brightest {
                brightest = v;
                at = (x, y);
            }
        }
    }
    eprintln!("brightest pixel {brightest:.3} at {at:?} (emitter radiance is 18.4)");

    // The emitter's radiance is 18.4 in red. Nothing else in the scene comes
    // close — a directly lit white wall is under 1 — so this is unambiguous.
    assert!(
        brightest > 15.0,
        "the brightest pixel is {brightest:.3}, far below the emitter's radiance of 18.4 — \
         directly visible emission is being suppressed under NEE. The camera ray has no \
         preceding shadow ray, so it is the one bounce whose emission must still be counted."
    );

    // ...and it must be the emitter, not a firefly: a real emitter covers many
    // pixels at nearly the same value.
    let bright_pixels = film.data.iter().filter(|p| p.x > 15.0).count();
    assert!(
        bright_pixels > 30,
        "only {bright_pixels} pixels are near the emitter's radiance; that is a firefly, \
         not the light fixture"
    );
}

/// **The point of multiple importance sampling: it must beat *both* strategies,
/// not split the difference.**
///
/// Measured on the Veach-style scene, which is built so each strategy fails
/// somewhere. BSDF sampling finds the small intensely bright emitter only by
/// luck; light sampling scatters points across the large dim emitter that a
/// near-mirror plate then evaluates at almost zero. MIS weights each direction
/// by how densely the strategy that produced it samples, so each takes credit
/// where it is strong.
///
/// If MIS ever lands *between* the two rather than below both, the weights are
/// wrong — most likely not summing to 1, or computed in mismatched measures.
#[test]
fn mis_beats_both_strategies() {
    let def = scenes::by_name("mis-scene").expect("scene exists");
    let base = RenderParams {
        width: 128,
        height: 96,
        samples: 64,
        max_depth: 5,
        ..Default::default()
    };

    // Noise level: the difference between two runs that share everything but
    // the seed, divided by sqrt(2).
    let noise = |mode: SamplingMode| -> f64 {
        let a = render(
            &def,
            &RenderParams {
                sampling: mode,
                ..base
            },
        );
        let b = render(
            &def,
            &RenderParams {
                sampling: mode,
                frame_seed: base.frame_seed ^ 0xABCD,
                ..base
            },
        );
        compare(&a, &b).expect("same size").mean_rel / std::f64::consts::SQRT_2
    };

    let bsdf = noise(SamplingMode::BsdfOnly);
    let nee = noise(SamplingMode::NeeOnly);
    let mis = noise(SamplingMode::Mis);
    eprintln!(
        "mis-scene relative noise at 64 spp: bsdf {bsdf:.5}, nee {nee:.5}, mis {mis:.5} \
         ({:.2}x and {:.2}x better)",
        bsdf / mis,
        nee / mis
    );

    assert!(
        mis < bsdf * 0.85,
        "MIS ({mis:.5}) is not meaningfully better than BSDF-only ({bsdf:.5})"
    );
    assert!(
        mis < nee * 0.85,
        "MIS ({mis:.5}) is not meaningfully better than NEE ({nee:.5}). A weight stuck \
         at 1 for light samples would reproduce NEE exactly."
    );
}

/// The power heuristic's weights must sum to exactly 1 for any pair of
/// densities.
///
/// This is what keeps the combined estimator unbiased: every direction either
/// strategy can produce is counted once in total, never twice and never dropped.
/// A normalisation slip here shows up as a uniformly too-bright or too-dark
/// image that looks entirely plausible.
#[test]
fn power_heuristic_weights_sum_to_one() {
    use pt_core::light::power_heuristic;

    let densities = [
        1e-8f32, 1e-4, 0.001, 0.01, 0.1, 1.0, 10.0, 1_000.0, 1e5, 1e8,
    ];
    for &a in &densities {
        for &b in &densities {
            let wa = power_heuristic(a, b);
            let wb = power_heuristic(b, a);
            assert!(
                (wa + wb - 1.0).abs() < 1e-5,
                "weights for ({a:e}, {b:e}) sum to {} rather than 1",
                wa + wb
            );
            assert!((0.0..=1.0).contains(&wa), "weight {wa} out of range");
        }
    }

    // Equal densities split evenly.
    assert!((power_heuristic(3.0, 3.0) - 0.5).abs() < 1e-6);
    // A strategy that cannot produce the direction gets nothing, and the other
    // takes all of it.
    assert_eq!(power_heuristic(0.0, 5.0), 0.0);
    assert_eq!(power_heuristic(5.0, 0.0), 1.0);
    // Squaring must make it *more* decisive than the balance heuristic: at a
    // 4:1 density ratio, balance gives 0.8 and power gives 16/17.
    let w = power_heuristic(4.0, 1.0);
    assert!(
        (w - 16.0 / 17.0).abs() < 1e-5,
        "power heuristic gave {w}, expected 16/17"
    );
}
