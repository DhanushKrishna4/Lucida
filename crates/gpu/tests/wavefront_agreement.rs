//! Build step 10: the wavefront architecture must render the same image as the
//! megakernel, not merely a plausible one.
//!
//! The two share every shading function — the WGSL is included from the same
//! `shaders/common` files — so what these tests actually pin is the plumbing
//! the split introduced: queue compaction, the deferred shadow rays, the order
//! random numbers are drawn in once shading is cut in half, and the per-sample
//! uniform updates.
//!
//! Agreement is checked against the *Monte Carlo noise floor* rather than an
//! absolute number. Two renderers drawing from bit-identical random streams
//! should differ only by float reassociation, which is orders of magnitude
//! below the noise; a threshold picked by hand would either be so loose it
//! catches nothing or so tight it flaps.

use pt_core::image::compare;
use pt_core::integrator::{self, RenderParams};
use pt_core::scenes;
use pt_gpu::{render_native_on, wavefront, Gpu};

fn gpu() -> Option<Gpu> {
    match Gpu::new() {
        Ok(g) => {
            eprintln!(
                "gpu: {} ({:?})",
                g.adapter_info.name, g.adapter_info.backend
            );
            Some(g)
        }
        Err(e) => {
            eprintln!("\n*** SKIPPING GPU TEST: {e} ***\n");
            None
        }
    }
}

/// Same threshold the megakernel is held to.
const AGREEMENT: f64 = 2.0e-3;

/// The headline check, on every scene that exercises a different code path:
/// analytic primitives, a BVH over a mesh, MIS-relevant geometry, and pure
/// specular transport.
#[test]
fn wavefront_matches_the_megakernel() {
    let Some(gpu) = gpu() else { return };
    for def in scenes::all() {
        let params = RenderParams {
            width: 96,
            height: 96,
            samples: 32,
            max_depth: 8,
            ..Default::default()
        };
        let mega = render_native_on(&gpu, &def, &params).expect("megakernel");
        let wave = wavefront::render_native_on(&gpu, &def, &params).expect("wavefront");
        let d = compare(&mega, &wave).expect("same size");
        eprintln!(
            "{:<14} mean rel {:.3e}  energy ratio {:.6}",
            def.name,
            d.mean_rel,
            d.mean_b / d.mean_a
        );
        assert!(
            d.mean_rel < AGREEMENT,
            "{}: wavefront and megakernel disagree by {:.3e} (max {:.3e} at {:?}).\n\
             Check, in order:\n\
               1. the order random numbers are drawn in SHADE (light select, light area, then BSDF)\n\
               2. whether a path is compacted out one bounce too early or too late\n\
               3. the MIS weight on the deferred shadow ray — CONNECT adds radiance the\n\
                  megakernel adds inline, and the throughput it multiplies by must be the\n\
                  one from *before* the BSDF sample advanced it",
            def.name,
            d.mean_rel,
            d.max_abs,
            d.max_abs_at
        );
        let ratio = d.mean_b / d.mean_a;
        assert!(
            (ratio - 1.0).abs() < 1e-3,
            "{}: energy ratio {ratio:.6} — the wavefront is losing or gaining light",
            def.name
        );
    }
}

/// And against the CPU oracle directly, so the wavefront is not merely
/// consistent with a megakernel that has itself drifted.
#[test]
fn wavefront_matches_the_cpu_reference() {
    let Some(gpu) = gpu() else { return };
    let def = scenes::cornell_box();
    let params = RenderParams {
        width: 96,
        height: 96,
        samples: 64,
        max_depth: 8,
        ..Default::default()
    };
    let cpu = integrator::render(&def, &params);
    let wave = wavefront::render_native_on(&gpu, &def, &params).expect("wavefront");
    let d = compare(&cpu, &wave).expect("same size");
    eprintln!("wavefront vs cpu: mean rel {:.3e}", d.mean_rel);
    assert!(d.mean_rel < AGREEMENT, "mean rel {:.3e}", d.mean_rel);
}

/// Agreement has to survive the bounce loop, where the compaction lives. At
/// `max_depth = 1` nothing is ever compacted, so a queue bug is invisible;
/// it only appears once a second bounce reads what the first wrote.
#[test]
fn agreement_holds_across_path_lengths() {
    let Some(gpu) = gpu() else { return };
    let def = scenes::cornell_mesh();
    for depth in [1u32, 2, 3, 4, 8, 16] {
        let params = RenderParams {
            width: 80,
            height: 80,
            samples: 32,
            max_depth: depth,
            ..Default::default()
        };
        let mega = render_native_on(&gpu, &def, &params).expect("megakernel");
        let wave = wavefront::render_native_on(&gpu, &def, &params).expect("wavefront");
        let d = compare(&mega, &wave).expect("same size");
        eprintln!("depth {depth:2}: mean rel {:.3e}", d.mean_rel);
        assert!(
            d.mean_rel < AGREEMENT,
            "diverged at max_depth = {depth}: mean rel {:.3e}",
            d.mean_rel
        );
    }
}

/// Every sampling mode, because NEE and MIS are what put work in the CONNECT
/// stage at all. Under `BsdfOnly` no shadow ray is ever queued, so the deferred
/// path is completely untested by the default configuration.
#[test]
fn agreement_holds_across_sampling_modes() {
    use pt_core::integrator::SamplingMode;
    let Some(gpu) = gpu() else { return };
    let def = scenes::mis_scene();
    for mode in [
        SamplingMode::BsdfOnly,
        SamplingMode::NeeOnly,
        SamplingMode::Mis,
    ] {
        let params = RenderParams {
            width: 96,
            height: 96,
            samples: 32,
            max_depth: 6,
            sampling: mode,
            ..Default::default()
        };
        let mega = render_native_on(&gpu, &def, &params).expect("megakernel");
        let wave = wavefront::render_native_on(&gpu, &def, &params).expect("wavefront");
        let d = compare(&mega, &wave).expect("same size");
        eprintln!("{:<5} mean rel {:.3e}", mode.name(), d.mean_rel);
        assert!(
            d.mean_rel < AGREEMENT,
            "{} sampling diverged: mean rel {:.3e}",
            mode.name(),
            d.mean_rel
        );
    }
}

/// Noise must actually fall as samples are added.
///
/// This is the test that would have caught the bug it was written for: every
/// sample was rendering with the *same* seed, because `Queue::write_buffer` is
/// flushed ahead of the command buffers submitted with it, so batching all
/// samples into one encoder applied every uniform update before any of them
/// ran. The image looked entirely correct — it simply never converged. Nothing
/// that compares two renderers catches that, because both were averaging the
/// same number.
///
/// Monte Carlo error falls as 1/sqrt(N), so quadrupling the samples should
/// roughly halve it. The bound is loose (any real reduction proves the seeds
/// differ) but the shape is the point.
#[test]
fn noise_falls_with_sample_count() {
    let Some(gpu) = gpu() else { return };
    let def = scenes::cornell_box();
    let base = RenderParams {
        width: 64,
        height: 64,
        max_depth: 6,
        ..Default::default()
    };

    // A converged reference to measure error against.
    let reference = wavefront::render_native_on(
        &gpu,
        &def,
        &RenderParams {
            samples: 512,
            ..base
        },
    )
    .expect("reference");

    let mut previous = f64::INFINITY;
    for samples in [4u32, 16, 64] {
        let f = wavefront::render_native_on(&gpu, &def, &RenderParams { samples, ..base })
            .expect("wavefront");
        let rmse = compare(&reference, &f).expect("same size").rmse;
        eprintln!("{samples:3} spp: rmse {rmse:.4}");
        assert!(
            rmse < previous * 0.75,
            "rmse did not fall meaningfully from {previous:.4} to {rmse:.4} at {samples} spp. \
             Every sample is probably drawing the same random stream — check that the \
             per-sample uniform write is ordered against the dispatches that read it."
        );
        previous = rmse;
    }
}
