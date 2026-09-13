//! Do all three sampling modes converge to the same mean?
use pt_core::integrator::{render, RenderParams, SamplingMode};
use pt_core::scenes;

fn main() {
    let def = scenes::by_name("mis-scene").expect("scene");
    println!(
        "{:>8} {:>12} {:>12} {:>12}   {:>10}",
        "spp", "bsdf", "nee", "mis", "nee/mis"
    );
    for spp in [64u32, 256, 1024, 4096, 16384] {
        let base = RenderParams {
            width: 64,
            height: 48,
            samples: spp,
            max_depth: 5,
            ..Default::default()
        };
        let mut means = Vec::new();
        for mode in SamplingMode::ALL {
            let f = render(
                &def,
                &RenderParams {
                    sampling: mode,
                    ..base
                },
            );
            let m = f.data.iter().map(|p| p.x as f64).sum::<f64>() / f.data.len() as f64;
            means.push(m);
        }
        println!(
            "{spp:>8} {:>12.5} {:>12.5} {:>12.5}   {:>10.4}",
            means[0],
            means[1],
            means[2],
            means[1] / means[2]
        );
    }
}
