//! Build step 17: the diagnostic colour ramp, WGSL against Rust.
//!
//! A colour ramp is exactly the kind of thing that looks fine while being wrong.
//! A mistyped stop, a transposed interpolation, an off-by-one in the segment
//! bounds — each produces a perfectly plausible gradient, and nobody can tell by
//! eye. So the two implementations are diffed at a few hundred points instead.

use glam::Vec3;
use pt_core::diagnostic::heat_ramp;
use pt_gpu::{eval::run_vec3_kernel, Gpu};

#[test]
fn the_heat_ramp_matches_between_wgsl_and_rust() {
    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("\n*** SKIPPING GPU TEST: {e} ***\n");
            return;
        }
    };

    // Dense through the segment boundaries, plus out-of-range values, which is
    // where a clamp written differently on the two sides would show.
    let mut ts: Vec<f32> = (0..=400).map(|i| i as f32 / 400.0).collect();
    ts.extend([-1.0, -0.001, 1.001, 2.0]);
    let inputs: Vec<Vec3> = ts.iter().map(|&t| Vec3::new(t, 0.0, 0.0)).collect();

    let gpu_out = run_vec3_kernel(&gpu, "test/eval_diagnostic.wgsl", 0, &inputs).expect("eval");

    let mut worst = (0.0f32, 0.0f32);
    for (i, &t) in ts.iter().enumerate() {
        let d = (gpu_out[i] - heat_ramp(t)).length();
        if d > worst.0 {
            worst = (d, t);
        }
    }
    eprintln!("largest ramp difference {:.2e} at t = {:.4}", worst.0, worst.1);
    assert!(
        worst.0 < 1.0e-5,
        "the WGSL heat ramp differs from the Rust one by {:.3e} at t = {:.4}. \
         Check the segment boundaries and that both clamp rather than wrap.",
        worst.0,
        worst.1
    );
}
