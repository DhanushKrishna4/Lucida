//! Build step 13: environment lighting, through the integrator.
//!
//! The unit tests in `envmap.rs` prove the sampler matches its own density.
//! These prove the renderer uses it correctly — which is a different claim, and
//! the one that fails silently.

use glam::Vec3;
use pt_core::camera::Camera;
use pt_core::envmap::{procedural_sky, EnvMap};
use pt_core::gpu_layout::{GpuMaterial, GpuPrimitive, SceneBlob};
use pt_core::integrator::Film;
use pt_core::integrator::{self, RenderParams, SamplingMode};
use pt_core::scene::Scene;
use pt_core::scenes::SceneDef;

fn sphere_scene(env: EnvMap, background: Vec3, roughness_material: GpuMaterial) -> SceneDef {
    let mut def = SceneDef {
        name: "env-probe",
        description: "",
        scene: Scene {
            blob: SceneBlob {
                materials: vec![roughness_material],
                primitives: vec![GpuPrimitive::sphere(Vec3::ZERO, 1.0, 0)],
                ..Default::default()
            },
            env,
            ..Default::default()
        },
        camera: Camera::look_at(Vec3::new(0.0, 0.0, -4.0), Vec3::ZERO, 45.0),
        background,
    };
    def.scene.finalize();
    def
}

fn mean_of(film: &Film) -> f64 {
    film.data.iter().map(|v| v.x as f64).sum::<f64>() / film.data.len() as f64
}

/// Relative standard deviation between two independent renders of the same
/// scene, which is a direct measure of Monte Carlo noise.
fn noise_between(a: &Film, b: &Film) -> f64 {
    let mut sum = 0.0f64;
    let mut n = 0.0f64;
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

/// A uniform environment map must behave exactly like a constant background.
///
/// The case where the CDF machinery has nothing to do, so anything it gets
/// wrong shows up unmixed with the effects of an actual sky. A white
/// non-absorbing sphere under a uniform environment of radiance 1 is the furnace
/// test again, now driven through the environment sampler.
#[test]
fn uniform_env_map_matches_a_constant_background() {
    let params = RenderParams {
        width: 64,
        height: 64,
        samples: 64,
        max_depth: 6,
        ..Default::default()
    };
    let uniform = EnvMap::new(32, 16, vec![[1.0, 1.0, 1.0, 0.0]; 32 * 16]);
    let mat = GpuMaterial::diffuse(Vec3::ONE);

    let with_map = integrator::render(&sphere_scene(uniform, Vec3::ZERO, mat), &params);
    let with_const = integrator::render(
        &sphere_scene(EnvMap::default(), Vec3::ONE, mat),
        &params,
    );

    let (a, b) = (mean_of(&with_map), mean_of(&with_const));
    eprintln!("uniform env {a:.5} vs constant background {b:.5}");
    assert!(
        (a - b).abs() < 5e-3 * b,
        "a uniform environment map renders at {a:.5} where a constant background \
         of the same radiance gives {b:.5}. The two describe identical lighting, \
         so any difference is the environment path's own error — suspect the \
         solid-angle Jacobian or the MIS selection count."
    );
    // And it must be the furnace answer: a white non-absorbing sphere in a
    // uniform environment of radiance 1 is invisible.
    // Deliberately *not* asserting that this equals exactly 1. It nearly does
    // — 0.977 measured — and the shortfall is the material's own energy
    // behaviour (`GpuMaterial::diffuse` carries a specular layer whose
    // compensation is approximate), not the environment path's. The comparison
    // above is the claim that isolates this module; asserting the absolute
    // value here would be testing the BSDF through a second lens and would fail
    // for reasons having nothing to do with environment lighting.
    assert!(
        (a - 1.0).abs() < 0.05,
        "uniform environment of radiance 1 rendered at {a:.5}; too far from 1 to \
         be the material's specular-layer approximation"
    );
}

/// Importance sampling the sky must converge faster, and to the same answer.
///
/// # Why the tolerance is computed rather than chosen
///
/// BSDF-only sampling is unbiased and, on a sky with a small sun, extremely
/// noisy — so at any affordable sample count its mean wanders. Measured here:
/// 0.370 at 32 spp, 0.430 at 256, 0.417 at 2048, 0.427 at 16384, against MIS
/// sitting at 0.429 throughout. An earlier version of this test compared the two
/// against a fixed 8% tolerance and reported a bias that does not exist.
///
/// So the agreement is checked against the noisy estimator's *own* standard
/// error, estimated from two independently seeded renders. That is
/// self-calibrating: it tightens automatically as the sample count rises and
/// never asserts more precision than the estimator has.
///
/// BSDF-only is worth keeping as the reference because it shares none of the
/// machinery under test — in that mode the renderer never calls `sample` or
/// `pdf` on the environment at all, only `radiance`. If the CDF, the
/// `sin(theta)` weight or the selection count were wrong, MIS would disagree
/// with it.
#[test]
fn envmap_sampling_converges_faster() {
    let sky = procedural_sky(256, 128, Vec3::new(-0.45, 0.22, -0.86), 6000.0, 1.0);
    let mat = GpuMaterial::diffuse(Vec3::splat(0.8));
    let def = sphere_scene(sky, Vec3::ZERO, mat);

    let base = RenderParams {
        width: 64,
        height: 64,
        samples: 128,
        max_depth: 3,
        ..Default::default()
    };

    // (mean, standard error of that mean, relative noise between the two runs)
    let measure = |mode: SamplingMode| {
        let a = integrator::render(&def, &RenderParams { sampling: mode, frame_seed: 1, ..base });
        let b = integrator::render(&def, &RenderParams { sampling: mode, frame_seed: 2, ..base });
        let (ma, mb) = (mean_of(&a), mean_of(&b));
        // Two independent estimates of the same quantity: their difference has
        // variance 2 * Var(mean), so |difference| / sqrt(2) estimates the
        // standard error of one of them.
        let se = (ma - mb).abs() / std::f64::consts::SQRT_2;
        (0.5 * (ma + mb), se, noise_between(&a, &b))
    };

    let (bsdf_mean, bsdf_se, bsdf_noise) = measure(SamplingMode::BsdfOnly);
    let (nee_mean, _, nee_noise) = measure(SamplingMode::NeeOnly);
    let (mis_mean, mis_se, mis_noise) = measure(SamplingMode::Mis);
    eprintln!("bsdf  mean {bsdf_mean:.5} +/- {bsdf_se:.5}, per-pixel noise {bsdf_noise:.4}");
    eprintln!("nee   mean {nee_mean:.5},                per-pixel noise {nee_noise:.4}");
    eprintln!("mis   mean {mis_mean:.5} +/- {mis_se:.5}, per-pixel noise {mis_noise:.4}");

    // Unbiasedness: MIS must sit within a few standard errors of the reference.
    // The floor keeps a freakishly lucky pair of BSDF seeds from making the
    // tolerance absurdly tight.
    let tolerance = (4.0 * bsdf_se).max(0.01 * bsdf_mean);
    assert!(
        (bsdf_mean - mis_mean).abs() < tolerance,
        "MIS converges to {mis_mean:.5} but BSDF-only sampling — which uses none \
         of the environment sampler — gives {bsdf_mean:.5} +/- {bsdf_se:.5}. \
         That is outside {tolerance:.5}, so one of them is biased. Check that the \
         environment's MIS weight uses the same 1/strategy-count factor on the \
         light-sampling and BSDF-hit sides."
    );
    // And the two CDF-driven strategies must agree with each other far more
    // tightly, since they share a sampler.
    assert!(
        (nee_mean - mis_mean).abs() < 0.01 * mis_mean,
        "NEE gives {nee_mean:.5} and MIS {mis_mean:.5}; these share the \
         environment sampler and should agree closely"
    );

    eprintln!("noise ratio bsdf/mis {:.1}x", bsdf_noise / mis_noise.max(1e-12));
    assert!(
        mis_noise * 3.0 < bsdf_noise,
        "importance sampling the sky reduced per-pixel noise only {:.2}x \
         ({bsdf_noise:.4} to {mis_noise:.4}). For a sun four orders of magnitude \
         above the sky this should be large.",
        bsdf_noise / mis_noise.max(1e-12)
    );
}
