//! The WGSL tone mapping operators must agree with the Rust ones.
//!
//! Both exist because the CLI needs them on the CPU and the browser needs them
//! in a shader. Duplication is unavoidable; silent divergence is not. A
//! transposed matrix or a mistyped coefficient renders a perfectly plausible
//! image, so this is checked numerically rather than visually — the AgX matrices
//! were in fact transposed when first written, and nothing in the picture would
//! have told us.

use glam::Vec3;
use pt_core::tonemap::Tonemap;
use pt_gpu::{eval::run_vec3_kernel, Gpu};

fn gpu() -> Option<Gpu> {
    match Gpu::new() {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("\n*** SKIPPING GPU TEST: {e} ***\n");
            None
        }
    }
}

/// Inputs spanning the range a real render produces: black, deep shadow, middle
/// grey, diffuse white, and emitter-bright values, in neutral and saturated
/// variants.
fn probe_inputs() -> Vec<Vec3> {
    let mut v = Vec::new();
    for &s in &[
        0.0f32, 1e-6, 1e-3, 0.018, 0.18, 0.5, 1.0, 2.0, 8.0, 18.4, 100.0, 1e4,
    ] {
        v.push(Vec3::splat(s));
        v.push(Vec3::new(s, s * 0.5, s * 0.1)); // warm
        v.push(Vec3::new(s * 0.1, s * 0.4, s)); // cool
        v.push(Vec3::new(s, 0.0, 0.0)); // fully saturated primary
        v.push(Vec3::new(0.0, s, 0.0));
        v.push(Vec3::new(0.0, 0.0, s));
    }
    v
}

#[test]
fn wgsl_tonemap_matches_rust() {
    let Some(gpu) = gpu() else { return };
    let inputs = probe_inputs();

    for t in Tonemap::ALL {
        let gpu_out =
            run_vec3_kernel(&gpu, "test/eval_vec3.wgsl", t.index(), &inputs).expect("eval kernel");
        assert_eq!(gpu_out.len(), inputs.len());

        let mut worst = 0.0f32;
        let mut worst_at = Vec3::ZERO;
        for (i, &input) in inputs.iter().enumerate() {
            let cpu = t.apply(input);
            let g = gpu_out[i];
            assert!(
                g.is_finite(),
                "{} produced {g} on the GPU for {input}",
                t.name()
            );
            for ch in 0..3 {
                // Absolute tolerance: outputs are bounded to [0, 1], so absolute
                // and relative error are the same order here. The tolerance
                // allows for `pow` and `log2` differing in the last couple of
                // ULP between the CPU's libm and the GPU's, which they do.
                let d = (cpu[ch] - g[ch]).abs();
                if d > worst {
                    worst = d;
                    worst_at = input;
                }
            }
        }
        assert!(
            worst < 2e-4,
            "{} differs between Rust and WGSL by {worst:.3e} (worst input {worst_at})",
            t.name()
        );
        eprintln!("{:<9} max |cpu - gpu| = {worst:.3e}", t.name());
    }
}

/// The sRGB encode is also duplicated, and it is applied to every pixel of every
/// frame, so it gets the same treatment.
#[test]
fn wgsl_srgb_encode_matches_rust() {
    let Some(gpu) = gpu() else { return };
    // Values clustered around the 0.0031308 breakpoint where the curve switches
    // from its linear segment to its power segment — the place an implementation
    // is most likely to disagree.
    let inputs: Vec<Vec3> = (0..200)
        .map(|i| {
            let x = (i as f32) / 199.0;
            Vec3::new(x, x * 0.0031308 * 2.0, x.powi(3))
        })
        .collect();

    let gpu_out = run_vec3_kernel(&gpu, "test/eval_srgb.wgsl", 0, &inputs).expect("eval kernel");
    let mut worst = 0.0f32;
    for (i, &input) in inputs.iter().enumerate() {
        for ch in 0..3 {
            let cpu = pt_core::image::linear_to_srgb(input[ch]);
            worst = worst.max((cpu - gpu_out[i][ch]).abs());
        }
    }
    assert!(worst < 2e-5, "sRGB encode differs by {worst:.3e}");
    eprintln!("srgb     max |cpu - gpu| = {worst:.3e}");
}
