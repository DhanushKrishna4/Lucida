//! The WGSL microfacet code must agree with the Rust reference, term by term.
//!
//! An image cannot tell you that `D` is off by a coefficient or that `eta` and
//! `k` got swapped — you get a plausible highlight of the wrong shape and no
//! signal that anything is wrong. These tests evaluate each term at a few
//! thousand known inputs on the GPU and diff against the CPU.

use glam::Vec3;
use pt_core::bsdf::{self, conductors};
use pt_gpu::{
    eval::{run_vec3_kernel, run_vec4_kernel},
    Gpu,
};

fn gpu() -> Option<Gpu> {
    match Gpu::new() {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("\n*** SKIPPING GPU TEST: {e} ***\n");
            None
        }
    }
}

const SHADER: &str = "test/eval_bsdf.wgsl";

/// Roughness and cosine pairs spanning the usable range, including the extremes
/// where the numerics are most fragile.
fn alpha_cos_pairs() -> Vec<Vec3> {
    let mut v = Vec::new();
    for &roughness in &[0.0f32, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0] {
        let alpha = bsdf::roughness_to_alpha(roughness);
        for i in 0..=40 {
            let cos = i as f32 / 40.0;
            v.push(Vec3::new(alpha, cos, 0.0));
        }
    }
    v
}

fn compare(name: &str, cpu: &[Vec3], gpu: &[Vec3], tol_rel: f32, tol_abs: f32) {
    assert_eq!(cpu.len(), gpu.len());
    let mut worst = 0.0f32;
    let mut worst_i = 0usize;
    for i in 0..cpu.len() {
        for ch in 0..3 {
            let (a, b) = (cpu[i][ch], gpu[i][ch]);
            assert!(b.is_finite(), "{name}: GPU produced {b} at input {i}");
            let d = (a - b).abs();
            // Relative where the values are large, absolute where they are tiny.
            // D reaches 1e5 at low roughness, so a pure absolute tolerance would
            // be meaningless there, and a pure relative one meaningless near zero.
            let scale = tol_abs + tol_rel * a.abs().max(b.abs());
            if d / scale > worst {
                worst = d / scale;
                worst_i = i;
            }
        }
    }
    assert!(
        worst <= 1.0,
        "{name}: Rust and WGSL disagree by {worst:.2}x the tolerance, worst at input {worst_i} \
         (cpu {:?}, gpu {:?})",
        cpu[worst_i],
        gpu[worst_i]
    );
    eprintln!("{name:<22} worst {worst:.3} of tolerance");
}

#[test]
fn ggx_distribution_matches() {
    let Some(gpu) = gpu() else { return };
    let inputs = alpha_cos_pairs();
    let cpu: Vec<Vec3> = inputs
        .iter()
        .map(|v| Vec3::splat(bsdf::ggx_d(v.x, v.y)))
        .collect();
    let out = run_vec3_kernel(&gpu, SHADER, 0, &inputs).expect("eval");
    // D spans 1e-3 to 1e5 across this range, so the tolerance has to be relative.
    compare("ggx_d", &cpu, &out, 2e-4, 1e-6);
}

#[test]
fn smith_masking_matches() {
    let Some(gpu) = gpu() else { return };
    let inputs = alpha_cos_pairs();

    let cpu: Vec<Vec3> = inputs
        .iter()
        .map(|v| Vec3::splat(bsdf::smith_g1(v.x, v.y)))
        .collect();
    let out = run_vec3_kernel(&gpu, SHADER, 1, &inputs).expect("eval");
    compare("smith_g1", &cpu, &out, 1e-5, 1e-7);

    // G2 needs both cosines.
    let pairs: Vec<Vec3> = inputs
        .iter()
        .enumerate()
        .map(|(i, v)| Vec3::new(v.x, v.y, ((i % 17) as f32 + 1.0) / 17.0))
        .collect();
    let cpu: Vec<Vec3> = pairs
        .iter()
        .map(|v| Vec3::splat(bsdf::smith_g2(v.x, v.y, v.z)))
        .collect();
    let out = run_vec3_kernel(&gpu, SHADER, 2, &pairs).expect("eval");
    compare("smith_g2", &cpu, &out, 1e-5, 1e-7);
}

#[test]
fn fresnel_matches() {
    let Some(gpu) = gpu() else { return };
    let mut inputs = Vec::new();
    for &f0 in &[0.02f32, 0.04, 0.08, 0.2, 0.5, 0.9, 1.0] {
        for i in 0..=50 {
            inputs.push(Vec3::new(f0, i as f32 / 50.0, 0.0));
        }
    }

    let cpu: Vec<Vec3> = inputs
        .iter()
        .map(|v| bsdf::fresnel_schlick(Vec3::new(v.x, v.x * 0.6, v.x * 0.3), v.y))
        .collect();
    let out = run_vec3_kernel(&gpu, SHADER, 3, &inputs).expect("eval");
    compare("fresnel_schlick", &cpu, &out, 1e-5, 1e-7);

    // The conductor path exercises eta and k per channel, where a swap or a
    // transposed constant is otherwise undetectable.
    let cpu: Vec<Vec3> = inputs
        .iter()
        .map(|v| bsdf::fresnel_conductor(v.y, conductors::COPPER_ETA, conductors::COPPER_K))
        .collect();
    let out = run_vec3_kernel(&gpu, SHADER, 4, &inputs).expect("eval");
    compare("fresnel_conductor", &cpu, &out, 2e-5, 1e-6);

    // ...and the result must actually be copper-coloured on the GPU, not merely
    // consistent with a CPU implementation that is itself wrong.
    let normal_incidence = out
        .iter()
        .zip(inputs.iter())
        .find(|(_, i)| i.y == 1.0)
        .map(|(o, _)| *o)
        .expect("cos = 1 sample");
    assert!(
        normal_incidence.x > normal_incidence.y && normal_incidence.y > normal_incidence.z,
        "GPU copper F0 = {normal_incidence}, expected warm (r > g > b)"
    );
}

/// The energy table is generated from Rust, but the *lookup* is written twice.
/// A mismatched interpolation would break energy conservation on the GPU only.
#[test]
fn directional_albedo_lookup_matches() {
    let Some(gpu) = gpu() else { return };
    let mut inputs = Vec::new();
    for ri in 0..=64 {
        for ci in 0..=64 {
            inputs.push(Vec3::new(ri as f32 / 64.0, ci as f32 / 64.0, 0.0));
        }
    }
    let cpu: Vec<Vec3> = inputs
        .iter()
        .map(|v| Vec3::splat(bsdf::energy::directional_albedo(v.x, v.y)))
        .collect();
    let out = run_vec3_kernel(&gpu, SHADER, 5, &inputs).expect("eval");
    compare("directional_albedo", &cpu, &out, 1e-5, 1e-6);
}

/// The VNDF sampler and its pdf, which are the parts most likely to diverge:
/// the sampler has a branch, a normalisation, and a stretch/unstretch pair.
#[test]
fn vndf_sampler_matches() {
    let Some(gpu) = gpu() else { return };
    let mut padded = Vec::new();
    for &roughness in &[0.02f32, 0.1, 0.3, 0.6, 1.0] {
        let alpha = bsdf::roughness_to_alpha(roughness);
        for i in 0..24 {
            for j in 0..24 {
                let cos_o = ((i % 8) as f32 + 1.0) / 8.0;
                let u = [(i as f32 + 0.5) / 24.0, (j as f32 + 0.5) / 24.0];
                padded.push([alpha, cos_o, u[0], u[1]]);
            }
        }
    }

    let cpu: Vec<Vec3> = padded
        .iter()
        .map(|p| {
            let sin_o = (1.0 - p[1] * p[1]).max(0.0).sqrt();
            let wo = Vec3::new(sin_o, 0.0, p[1]);
            bsdf::sample_ggx_vndf(wo, p[0], glam::Vec2::new(p[2], p[3]))
        })
        .collect();
    let out = run_vec4_kernel(&gpu, SHADER, 6, &padded).expect("eval");
    compare("sample_ggx_vndf", &cpu, &out, 5e-5, 1e-6);

    // The raw pdf is compared only away from the precision floor.
    //
    // At MIN_ALPHA the pdf is hypersensitive by construction: `D`'s denominator
    // near the lobe peak is `alpha^2`, which is 4e-6, so the ~5e-5 difference
    // in the sampled normal that the test above tolerates moves `D` by a factor
    // of 25. That is amplification of a legitimate `f32` difference, not a
    // transcription error — and it is the same precision wall `MIN_ALPHA`
    // documents.
    //
    // The sensitivity scales roughly as 1/alpha^2, so it fades rather than
    // vanishing: at roughness 0.1 the residual is about 0.5%, against 11% at the
    // floor. The tolerance below accommodates that, because the quantity the
    // renderer depends on is the *weight*, which agrees to 1e-4 across the whole
    // range — see `specular_weight_is_well_conditioned`.
    let rough: Vec<[f32; 4]> = padded
        .iter()
        .copied()
        .filter(|p| p[0] >= bsdf::roughness_to_alpha(0.1))
        .collect();
    let cpu: Vec<Vec3> = rough
        .iter()
        .map(|p| {
            let sin_o = (1.0 - p[1] * p[1]).max(0.0).sqrt();
            let wo = Vec3::new(sin_o, 0.0, p[1]);
            let m = bsdf::sample_ggx_vndf(wo, p[0], glam::Vec2::new(p[2], p[3]));
            Vec3::splat(bsdf::ggx_vndf_pdf(p[0], wo, m))
        })
        .collect();
    let out = run_vec4_kernel(&gpu, SHADER, 7, &rough).expect("eval");
    compare("ggx_vndf_pdf", &cpu, &out, 1e-2, 1e-6);
}

/// The estimator weight must agree across the **whole** roughness range,
/// including the precision floor where `D` and the pdf individually do not.
///
/// This is the point of visible-normal sampling: `D` appears in both the BRDF
/// and the pdf and cancels exactly, leaving `F * G2 / G1(wo)`. So the quantity
/// that reaches the image is well conditioned even where its ingredients are
/// not — which is why the renders agree to 1e-5 while the raw pdf differs by
/// 11% at `MIN_ALPHA`.
#[test]
fn specular_weight_is_well_conditioned() {
    let Some(gpu) = gpu() else { return };
    let mut padded = Vec::new();
    for &roughness in &[0.0f32, 0.02, 0.05, 0.1, 0.3, 0.6, 1.0] {
        let alpha = bsdf::roughness_to_alpha(roughness);
        for i in 0..24 {
            for j in 0..24 {
                let cos_o = ((i % 8) as f32 + 1.0) / 8.0;
                padded.push([
                    alpha,
                    cos_o,
                    (i as f32 + 0.5) / 24.0,
                    (j as f32 + 0.5) / 24.0,
                ]);
            }
        }
    }
    let cpu: Vec<Vec3> = padded
        .iter()
        .map(|p| {
            let sin_o = (1.0 - p[1] * p[1]).max(0.0).sqrt();
            let wo = Vec3::new(sin_o, 0.0, p[1]);
            let s = bsdf::specular::sample_single(p[0], Vec3::ONE, wo, glam::Vec2::new(p[2], p[3]));
            if s.is_valid() {
                Vec3::splat(s.weight.x)
            } else {
                Vec3::ZERO
            }
        })
        .collect();
    let out = run_vec4_kernel(&gpu, SHADER, 8, &padded).expect("eval");
    compare("specular weight", &cpu, &out, 1e-4, 1e-6);
}
