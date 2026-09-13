//! How closely do BSDF-only and NEE agree as sample counts rise?
//!
//! They estimate the same integral, so they must converge to the same image.
//! This measures at what cost that becomes assertable, which is what sets the
//! parameters of the test in `crates/core/tests/sampling_modes.rs`.
//!
//! `cargo run --release -p pt-core --example nee_convergence`

use pt_core::image::compare;
use pt_core::integrator::{render, RenderParams, SamplingMode};
use pt_core::scenes;

fn main() {
    let def = scenes::cornell_box();
    println!(
        "{:>6} {:>8} {:>14} {:>14} {:>12}",
        "size", "spp", "mean (bsdf)", "mean (nee)", "ratio"
    );
    for (w, spp) in [
        (32u32, 512u32),
        (32, 2048),
        (32, 8192),
        (48, 8192),
        (64, 16384),
    ] {
        let base = RenderParams {
            width: w,
            height: w,
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
        let d = compare(&a, &b).unwrap();
        println!(
            "{w:>6} {spp:>8} {:>14.6} {:>14.6} {:>12.6}   rmse {:.4}  mean rel {:.4}",
            d.mean_a,
            d.mean_b,
            d.mean_b / d.mean_a,
            d.rmse,
            d.mean_rel
        );
    }
}
