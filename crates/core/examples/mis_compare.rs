//! Variance of the three sampling strategies, at matched sample counts.
//!
//! `cargo run --release -p pt-core --example mis_compare`
//!
//! MIS should beat *both* of the strategies it combines, not split the
//! difference between them — if it only lands in the middle, the weights are
//! wrong.

use pt_core::image::compare;
use pt_core::integrator::{render, RenderParams, SamplingMode};
use pt_core::scenes;

fn main() {
    for scene in ["mis-scene", "cornell-box"] {
        let def = scenes::by_name(scene).expect("scene");
        let base = RenderParams {
            width: 128,
            height: 96,
            samples: 64,
            max_depth: 5,
            ..Default::default()
        };
        println!("\n{scene}  (64 spp)");
        println!(
            "  {:<6} {:>12} {:>12} {:>10}",
            "mode", "rel noise", "mean", "vs best"
        );

        let mut results = Vec::new();
        for mode in SamplingMode::ALL {
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
            let d = compare(&a, &b).unwrap();
            results.push((mode, d.mean_rel / std::f64::consts::SQRT_2, d.mean_a));
        }
        let best = results.iter().map(|r| r.1).fold(f64::INFINITY, f64::min);
        for (mode, noise, mean) in &results {
            println!(
                "  {:<6} {noise:>12.5} {mean:>12.6} {:>10.2}x",
                mode.name(),
                noise / best
            );
        }
    }
}
