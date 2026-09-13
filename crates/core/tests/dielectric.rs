//! Build step 12: transmission and dielectrics.
//!
//! The chi-squared suite proves `sample` and `pdf` describe the same
//! distribution. That is necessary and not sufficient: a BSDF can be perfectly
//! self-consistent and still create or destroy energy, or bend light the wrong
//! way. These tests check the physics the sampling tests cannot see.

use glam::Vec3;
use pt_core::bsdf::{self, fresnel_dielectric, refract, surface::Surface};
use pt_core::gpu_layout::GpuMaterial;
use pt_core::rng::Rng;

fn wo_from_cos(cos_o: f32) -> Vec3 {
    Vec3::new((1.0 - cos_o * cos_o).max(0.0).sqrt(), 0.0, cos_o)
}

/// Snell's law, checked against the trigonometric form.
///
/// The vector form in `refract` is algebraically equivalent and is the one that
/// is easy to get subtly wrong, so it is worth confirming against the textbook
/// statement rather than against itself.
#[test]
fn refraction_obeys_snells_law() {
    let n = Vec3::Z;
    for &eta in &[1.5f32, 1.0 / 1.5, 2.42, 1.33] {
        for &cos_i in &[0.99f32, 0.8, 0.5, 0.2, 0.05] {
            let wo = wo_from_cos(cos_i);
            let Some(wt) = refract(wo, n, eta) else {
                // Only possible when leaving a denser medium.
                assert!(eta < 1.0, "unexpected TIR entering a denser medium");
                continue;
            };
            assert!(
                wt.z < 0.0,
                "a refracted ray must cross the surface (eta {eta}, cos_i {cos_i})"
            );
            let sin_i = (1.0 - cos_i * cos_i).max(0.0).sqrt();
            let sin_t = (1.0 - wt.z * wt.z).max(0.0).sqrt();
            // n_i sin(theta_i) = n_t sin(theta_t), and eta = n_t / n_i.
            assert!(
                (sin_i - eta * sin_t).abs() < 1e-4,
                "Snell violated: sin_i {sin_i:.6}, eta*sin_t {:.6} (eta {eta}, cos_i {cos_i})",
                eta * sin_t
            );
            // The transmitted ray stays in the plane of incidence.
            assert!(wt.y.abs() < 1e-6, "refraction left the plane of incidence");
        }
    }
}

/// Refracting and refracting back must return the original direction.
#[test]
fn refraction_is_reversible() {
    let n = Vec3::Z;
    for &eta in &[1.5f32, 2.42, 1.33] {
        for &cos_i in &[0.95f32, 0.6, 0.3] {
            let wo = wo_from_cos(cos_i);
            let wt = refract(wo, n, eta).expect("entering never TIRs");
            // Coming back the other way: the normal faces the ray, so flip it,
            // and the relative index inverts.
            let back = refract(-wt, -n, 1.0 / eta).expect("the reverse path exists");
            assert!(
                (back + wo).length() < 1e-3 || (back - wo).length() < 1e-3,
                "round trip gave {back:?} for {wo:?} (eta {eta})"
            );
        }
    }
}

/// Past the critical angle a dielectric reflects everything.
#[test]
fn total_internal_reflection_is_total() {
    let eta = 1.0f32 / 1.5;
    // sin(theta_c) = eta, so cos(theta_c) = sqrt(1 - eta^2).
    let cos_critical = (1.0 - eta * eta).sqrt();
    assert!((cos_critical - 0.7454).abs() < 1e-3, "critical angle moved");

    for &cos_o in &[0.7f32, 0.5, 0.2, 0.01] {
        assert!(cos_o < cos_critical);
        assert_eq!(
            fresnel_dielectric(cos_o, eta),
            1.0,
            "Fresnel must be exactly 1 past the critical angle (cos_o {cos_o})"
        );
        assert!(
            refract(wo_from_cos(cos_o), Vec3::Z, eta).is_none(),
            "there is no transmitted direction past the critical angle"
        );

        // And the sampler must never produce one.
        let m = GpuMaterial::glass(Vec3::ONE, 0.0, 1.5);
        let s = Surface::from_material(&m, false);
        let wo = wo_from_cos(cos_o);
        let mut rng = Rng::new(0x71, 0, 0);
        for _ in 0..20_000 {
            let smp = bsdf::surface::sample(&s, wo, rng.next_f32(), rng.next_vec2());
            if smp.is_valid() {
                assert!(
                    smp.wi.z > 0.0,
                    "sampled a transmitted direction past the critical angle"
                );
            }
        }
    }
}

/// Total reflectance over both hemispheres.
fn total_albedo(s: &Surface, wo: Vec3, samples: usize, seed: u32) -> (f32, f32) {
    let mut rng = Rng::new(seed, 0, 0);
    let (mut up, mut down) = (0.0f32, 0.0f32);
    for _ in 0..samples {
        let smp = bsdf::surface::sample(s, wo, rng.next_f32(), rng.next_vec2());
        if !smp.is_valid() {
            continue;
        }
        // `weight` is f * cos / pdf, so its mean is the directional albedo.
        let w = pt_core::math::luminance(smp.weight);
        if smp.wi.z > 0.0 {
            up += w;
        } else {
            down += w;
        }
    }
    (up / samples as f32, down / samples as f32)
}

/// Clear glass must not create light.
///
/// # Why the test is `reflect + eta^2 * transmit`, not `reflect + transmit`
///
/// A BTDF is not symmetric. Radiance is compressed by `eta^2` on entering a
/// denser medium — the same power squeezed into a narrower cone — so a path
/// tracer starting at the camera, which transports importance rather than light,
/// carries a `1 / eta^2` factor that the light-transport BTDF does not (Veach
/// 1997, §5.2).
///
/// The consequence is that a single interface's *radiance* albedo is `1 / eta^2`
/// for the transmitted half, not 1. That is correct and is not energy being
/// lost: the factor is undone on the way out of the object, which
/// `glass_round_trip_is_lossless` checks directly.
///
/// Getting this backwards is the classic dielectric bug in both directions — an
/// implementation that includes the BTDF's `eta^2` and forgets the transport
/// correction is too bright by 2.25x for glass and merely looks "glowy", and one
/// that applies the correction twice is too dark by the same factor. Writing the
/// invariant out here is what makes the difference checkable.
#[test]
fn smooth_glass_conserves_energy() {
    for &ior in &[1.33f32, 1.5, 2.42] {
        for front in [true, false] {
            let m = GpuMaterial::glass(Vec3::ONE, 0.01, ior);
            let s = Surface::from_material(&m, front);
            let eta = if front { ior } else { 1.0 / ior };
            for &cos_o in &[0.95f32, 0.7, 0.4, 0.15] {
                // Past the critical angle nothing transmits, which is its own
                // test (`total_internal_reflection_is_total`).
                if fresnel_dielectric(cos_o, eta) >= 1.0 {
                    continue;
                }
                let wo = wo_from_cos(cos_o);
                let (up, down) = total_albedo(&s, wo, 200_000, 0x9E ^ ior.to_bits());
                let total = up + eta * eta * down;
                eprintln!(
                    "ior {ior:<5} {} cos_o {cos_o:<5}: reflect {up:.4} transmit {down:.4} \
                     reflect + eta^2*transmit {total:.4}",
                    if front { "in " } else { "out" }
                );
                assert!(
                    total <= 1.01,
                    "ior {ior}, front {front}, cos_o {cos_o}: {total:.4} > 1 — \
                     a smooth interface cannot create light"
                );
                assert!(
                    total > 0.98,
                    "ior {ior}, front {front}, cos_o {cos_o}: {total:.4} — \
                     a smooth interface should neither absorb nor scatter"
                );
            }
        }
    }
}

/// Entering and leaving glass must leave radiance unchanged.
///
/// This is the claim that actually matters for an image, and the one that makes
/// the `eta^2` bookkeeping in `smooth_glass_conserves_energy` legible: the
/// compression on the way in is undone on the way out, so a ray that passes
/// clean through a pane of glass carries only the two Fresnel transmittances and
/// no index factor at all.
#[test]
fn glass_round_trip_is_lossless() {
    for &ior in &[1.33f32, 1.5, 2.42] {
        let cos_o = 0.92f32;
        let wo = wo_from_cos(cos_o);

        let inside = Surface::from_material(&GpuMaterial::glass(Vec3::ONE, 0.01, ior), true);
        let (_, t_in) = total_albedo(&inside, wo, 200_000, 0xAB ^ ior.to_bits());

        // Leaving along the refracted direction, with the frame flipped to face
        // the outgoing ray as the integrator does.
        let wt = refract(wo, Vec3::Z, ior).expect("entering never TIRs");
        let wo_out = Vec3::new(-wt.x, -wt.y, -wt.z);
        let outside = Surface::from_material(&GpuMaterial::glass(Vec3::ONE, 0.01, ior), false);
        let (_, t_out) = total_albedo(&outside, wo_out, 200_000, 0xCD ^ ior.to_bits());

        let through = t_in * t_out;
        let fresnel_only = (1.0 - fresnel_dielectric(cos_o, ior))
            * (1.0 - fresnel_dielectric(wo_out.z, 1.0 / ior));
        eprintln!(
            "ior {ior:<5}: through {through:.4}, Fresnel-only prediction {fresnel_only:.4}"
        );
        assert!(
            (through - fresnel_only).abs() < 0.02,
            "ior {ior}: a round trip transmitted {through:.4} where the two Fresnel \
             terms alone predict {fresnel_only:.4}. The eta^2 radiance compression \
             is not cancelling, so glass is uniformly too {} .",
            if through < fresnel_only { "dark" } else { "bright" }
        );
    }
}

/// An opaque material must behave exactly as it did before transmission
/// existed.
#[test]
fn transmission_zero_is_unchanged() {
    let m = GpuMaterial::glossy(Vec3::new(0.6, 0.5, 0.4), 0.3);
    assert_eq!(m.transmission, 0.0);
    let s = Surface::from_material(&m, true);
    let mut rng = Rng::new(0x0BA, 0, 0);
    for _ in 0..50_000 {
        let wo = {
            let c = 0.2 + 0.75 * rng.next_f32();
            wo_from_cos(c)
        };
        let smp = bsdf::surface::sample(&s, wo, rng.next_f32(), rng.next_vec2());
        if smp.is_valid() {
            assert!(
                smp.wi.z > 0.0,
                "an opaque material transmitted a direction"
            );
        }
        assert_eq!(
            bsdf::surface::pdf(&s, wo, Vec3::new(0.1, 0.1, -0.9).normalize()),
            0.0,
            "an opaque material has no density below the surface"
        );
    }
}

/// The white furnace test, for glass.
///
/// A non-absorbing dielectric sphere in a uniform environment of radiance 1 must
/// be **invisible**. Every ray that enters eventually leaves, the Fresnel split
/// loses nothing, and the `eta^2` compression on the way in is undone on the way
/// out — so whatever path a ray takes, it ends up carrying exactly the
/// background it started with.
///
/// This is the test that catches what the unit tests cannot: it exercises the
/// integrator, not the BSDF. A sign error in the ray-origin offset, a throughput
/// update on the wrong side, or a missed hemisphere flip all leave the BSDF's
/// own energy balance intact and still render glass as black — which is exactly
/// what happened here before the offset was tied to the sampled direction.
#[test]
fn glass_is_invisible_in_a_furnace() {
    use pt_core::integrator::{self, RenderParams};
    use pt_core::gpu_layout::SceneBlob;
    use pt_core::scene::Scene;
    use pt_core::scenes::SceneDef;
    use pt_core::camera::Camera;
    use pt_core::gpu_layout::GpuPrimitive;

    // The floor each case has to clear.
    //
    // A *smooth* dielectric has no microfacet multiple scattering to lose, so it
    // must come back essentially exact — that is the assertion that pins the
    // integrator's handling of transmission.
    //
    // A *rough* one is expected to lose energy, and the amount is not arbitrary:
    // the reflection lobe carries a directional-albedo compensation table (build
    // step 8) that recovers light lost between microfacets, and the transmission
    // lobe has no equivalent. What leaks is the second and later bounces off the
    // microsurface. Measured at 3.6% for roughness 0.3, which is in line with
    // published single-scattering figures; the bound is set to catch that
    // growing, not to hide it.
    for &(roughness, ior, floor, label) in &[
        (0.0f32, 1.5f32, 0.995f32, "smooth glass"),
        (0.3, 1.5, 0.95, "rough glass"),
        (0.0, 2.42, 0.995, "diamond"),
    ] {
        let mut def = SceneDef {
            name: "glass-furnace",
            description: "",
            scene: Scene {
                blob: SceneBlob {
                    materials: vec![GpuMaterial::glass(Vec3::ONE, roughness, ior)],
                    primitives: vec![GpuPrimitive::sphere(Vec3::ZERO, 1.0, 0)],
                    ..Default::default()
                },
                ..Default::default()
            },
            camera: Camera::look_at(Vec3::new(0.0, 0.0, -4.0), Vec3::ZERO, 45.0),
            background: Vec3::ONE,
        };
        def.scene.finalize();

        // Deep enough that a ray can bounce around inside a diamond sphere and
        // still get out: at IOR 2.42 the critical angle is 24 degrees, so most
        // internal hits are total internal reflection and paths are long.
        let params = RenderParams {
            width: 48,
            height: 48,
            samples: 256,
            max_depth: 64,
            ..Default::default()
        };
        let film = integrator::render(&def, &params);

        let n = film.data.len() as f64;
        let mean: f64 = film.data.iter().map(|v| v.x as f64).sum::<f64>() / n;
        let worst = film
            .data
            .iter()
            .map(|v| (v.x as f64 - 1.0).abs())
            .fold(0.0f64, f64::max);
        eprintln!("{label:<13} ior {ior:<5} r {roughness:<4}: mean {mean:.5}, worst pixel {worst:.4}");

        // Truncation at `max_depth` is a real loss for long internal paths, so
        // the tolerance is one-sided in spirit: too dark is explainable, too
        // bright is not.
        assert!(
            mean < 1.002,
            "{label}: mean {mean:.5} — a non-absorbing dielectric cannot create light"
        );
        assert!(
            mean > floor as f64,
            "{label}: mean {mean:.5} against a floor of {floor} — {:.2}% of the light \
             is being lost. For a smooth dielectric that is a bug, not a model \
             limitation: suspect the ray-origin offset side, the hemisphere check in \
             the throughput update, or paths truncated before they escape.",
            100.0 * (1.0 - mean)
        );
    }
}
