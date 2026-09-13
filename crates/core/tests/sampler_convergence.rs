//! Build step 14: Sobol sampling, measured on images.
//!
//! `sobol.rs` proves the sequence is stratified and integrates a smooth function
//! faster than random. Neither claim transfers automatically to a renderer: a
//! path tracer's integrand is discontinuous (silhouettes, shadow boundaries),
//! high-dimensional, and sometimes heavy-tailed, and stratification helps less
//! the further from smooth it gets. So the question is settled on renders.

use glam::Vec3;
use pt_core::camera::Camera;
use pt_core::gpu_layout::{GpuMaterial, GpuPrimitive, SceneBlob};
use pt_core::integrator::{self, Film, RenderParams};
use pt_core::scene::Scene;
use pt_core::scenes::{self, SceneDef};
use pt_core::sobol::SamplerKind;

fn mean_of(film: &Film) -> f64 {
    film.data.iter().map(|v| v.x as f64).sum::<f64>() / film.data.len() as f64
}

/// Relative deviation between two independently seeded renders: a reference-free
/// estimate of Monte Carlo noise.
fn noise_between(a: &Film, b: &Film) -> f64 {
    let (mut sum, mut n) = (0.0f64, 0.0f64);
    for (x, y) in a.data.iter().zip(b.data.iter()) {
        let (x, y) = (x.x as f64, y.x as f64);
        let m = 0.5 * (x + y);
        if m > 1e-4 {
            sum += ((x - y) / m).powi(2);
            n += 1.0;
        }
    }
    (sum / n.max(1.0)).sqrt()
}

fn measure(def: &SceneDef, kind: SamplerKind, spp: u32) -> (f64, f64) {
    let base = RenderParams {
        width: 64,
        height: 64,
        samples: spp,
        max_depth: 5,
        sampler: kind,
        ..Default::default()
    };
    let a = integrator::render(def, &RenderParams { frame_seed: 1, ..base });
    let b = integrator::render(def, &RenderParams { frame_seed: 2, ..base });
    (0.5 * (mean_of(&a) + mean_of(&b)), noise_between(&a, &b))
}

/// Both samplers must converge to the same image.
///
/// The claim that matters most, and the one a noise measurement cannot make: a
/// sampler that is quieter and *wrong* looks like a triumph. Owen scrambling is
/// what keeps Sobol unbiased — the points move, but every region still gets its
/// correct share of probability — so this is a direct test of the scramble.
///
/// The tolerance comes from the estimators' own spread rather than a chosen
/// number. On `sunset` the two disagreed by 7% at 256 spp, which looked exactly
/// like bias and was not: taken to 16 384 spp both converge to 0.2701. Caustics
/// give the mean a heavy tail, and neither estimator is converged at any
/// affordable sample count, so a fixed tolerance there asserts precision that
/// does not exist.
#[test]
fn sobol_and_independent_agree() {
    for def in [scenes::cornell_box(), scenes::glass_box(), scenes::sunset()] {
        let base = RenderParams {
            width: 64,
            height: 64,
            samples: 256,
            max_depth: 5,
            ..Default::default()
        };
        // Two independent estimates per sampler; their difference has variance
        // 2 * Var(mean), so |difference| / sqrt(2) estimates the standard error.
        let se_of = |kind: SamplerKind| {
            let p = RenderParams { sampler: kind, ..base };
            let a = integrator::render(&def, &RenderParams { frame_seed: 1, ..p });
            let b = integrator::render(&def, &RenderParams { frame_seed: 2, ..p });
            let (ma, mb) = (mean_of(&a), mean_of(&b));
            (0.5 * (ma + mb), (ma - mb).abs() / std::f64::consts::SQRT_2)
        };
        let (sobol_mean, sobol_se) = se_of(SamplerKind::Sobol);
        let (indep_mean, indep_se) = se_of(SamplerKind::Independent);

        // Combined standard error of the difference, with a floor so a lucky
        // pair of seeds cannot make the tolerance absurdly tight.
        let se = (sobol_se * sobol_se + indep_se * indep_se).sqrt();
        let tolerance = (4.0 * se).max(0.004 * indep_mean);
        eprintln!(
            "{:<12} sobol {sobol_mean:.5} +/- {sobol_se:.5}, independent \
             {indep_mean:.5} +/- {indep_se:.5}, tolerance {tolerance:.5}",
            def.name
        );
        assert!(
            (sobol_mean - indep_mean).abs() < tolerance,
            "{}: Sobol converges to {sobol_mean:.5} and independent sampling to \
             {indep_mean:.5}, outside {tolerance:.5}. They estimate the same \
             integral, so one is biased — suspect the Owen scramble, which is \
             what makes a deterministic sequence an unbiased estimator.",
            def.name
        );
    }
}

/// Sobol must be quieter where stratification can help.
#[test]
fn sobol_reduces_noise_on_smooth_scenes() {
    eprintln!(
        "\n{:<14} {:>12} {:>12} {:>8}",
        "scene", "independent", "sobol", "ratio"
    );
    for def in [scenes::cornell_box(), scenes::cornell_mesh(), scenes::mis_scene()] {
        let (_, indep) = measure(&def, SamplerKind::Independent, 64);
        let (_, sobol) = measure(&def, SamplerKind::Sobol, 64);
        let ratio = indep / sobol;
        eprintln!("{:<14} {indep:>12.4} {sobol:>12.4} {ratio:>7.2}x", def.name);
        assert!(
            ratio > 1.1,
            "{}: Sobol is only {ratio:.2}x quieter than independent sampling on a \
             scene whose integrand is smooth enough for stratification to pay.",
            def.name
        );
    }
}

/// And it must be honest about where it does not help.
///
/// `sunset` has a mirror and a glass ball, so its variance is dominated by
/// caustic paths — rare samples carrying enormous values. Stratification spreads
/// samples evenly over the domain, which does nothing for an integrand whose
/// mass is concentrated in a region too small for the strata to resolve.
/// Measured, Sobol is *slightly worse* there.
///
/// Pinned rather than hidden, and asserted as a band rather than a bound: if
/// this ever moved sharply in either direction it would mean something changed
/// about how the sampler interacts with heavy tails, which is worth knowing.
#[test]
fn sobol_does_not_help_with_caustics() {
    let def = scenes::sunset();
    let (_, indep) = measure(&def, SamplerKind::Independent, 64);
    let (_, sobol) = measure(&def, SamplerKind::Sobol, 64);
    let ratio = indep / sobol;
    eprintln!("\nsunset (caustic-dominated): {ratio:.2}x — stratification does not help here");
    assert!(
        (0.7..1.2).contains(&ratio),
        "sunset: noise ratio {ratio:.2}x is outside the band this scene has \
         historically sat in. Sobol is expected to be roughly a wash on a \
         caustic-dominated scene; a large move either way means something \
         changed about the interaction with heavy tails."
    );
}

/// What the advantage actually is: a better constant, not a better rate.
///
/// This is the honest result of the step and it is worth stating plainly. The
/// *sequence* converges far better than random on a smooth integrand — measured
/// in `sobol.rs`, the error ratio grows from 79x at 64 points to 2246x at 4096,
/// which is a genuinely better rate. A **path tracer** does not see that,
/// because its integrand is discontinuous at every silhouette and shadow
/// boundary and runs to thirty-odd dimensions. Measured on a diffuse sphere
/// under a uniform sky, about as smooth as a render gets:
///
/// ```text
///   16 spp: 1.33x    64 spp: 1.40x    256 spp: 1.39x
/// ```
///
/// The ratio rises once and then flattens. That is a constant-factor win of
/// about 1.4x in noise — roughly halving the samples needed — and not the
/// asymptotic improvement the sequence is capable of in principle.
#[test]
fn the_advantage_is_a_constant_factor() {
    let mut def = SceneDef {
        name: "smooth",
        description: "",
        scene: Scene {
            blob: SceneBlob {
                materials: vec![GpuMaterial::diffuse(Vec3::splat(0.7))],
                primitives: vec![GpuPrimitive::sphere(Vec3::ZERO, 1.0, 0)],
                ..Default::default()
            },
            ..Default::default()
        },
        camera: Camera::look_at(Vec3::new(0.0, 0.0, -4.0), Vec3::ZERO, 45.0),
        background: Vec3::ONE,
    };
    def.scene.finalize();

    eprintln!("\n{:>6} {:>12} {:>12} {:>8}", "spp", "independent", "sobol", "ratio");
    let mut ratios = Vec::new();
    for spp in [16u32, 64, 256] {
        let (_, indep) = measure(&def, SamplerKind::Independent, spp);
        let (_, sobol) = measure(&def, SamplerKind::Sobol, spp);
        let ratio = indep / sobol;
        eprintln!("{spp:>6} {indep:>12.5} {sobol:>12.5} {ratio:>7.2}x");
        ratios.push(ratio);
    }
    for (spp, r) in [16, 64, 256].iter().zip(&ratios) {
        assert!(
            *r > 1.25,
            "at {spp} spp Sobol is only {r:.2}x quieter; the constant-factor \
             advantage on a smooth scene should be around 1.4x"
        );
    }
}
