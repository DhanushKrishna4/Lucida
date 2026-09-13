use pt_core::image::compare;
use pt_core::integrator::{render, RenderParams, SamplingMode};
use pt_core::scenes;

fn main() {
    let def = scenes::cornell_box();

    // Where is the light in the frame, and how bright does it render?
    let p = RenderParams {
        width: 128,
        height: 128,
        samples: 64,
        max_depth: 4,
        sampling: SamplingMode::NeeOnly,
        ..Default::default()
    };
    let film = render(&def, &p);
    let (mut best, mut at) = (0.0f32, (0u32, 0u32));
    for y in 0..film.height {
        for x in 0..film.width {
            let v = film.pixel(x, y).x;
            if v > best {
                best = v;
                at = (x, y);
            }
        }
    }
    println!("brightest pixel {best:.3} at {at:?}  (emitter radiance is 18.4)");
    // Bounding box of everything above 10.
    let (mut x0, mut x1, mut y0, mut y1) = (u32::MAX, 0u32, u32::MAX, 0u32);
    for y in 0..film.height {
        for x in 0..film.width {
            if film.pixel(x, y).x > 10.0 {
                x0 = x0.min(x);
                x1 = x1.max(x);
                y0 = y0.min(y);
                y1 = y1.max(y);
            }
        }
    }
    println!("pixels above 10.0 span x {x0}..={x1}, y {y0}..={y1}");

    // Noise, measured two ways.
    println!(
        "\n{:<10} {:>10} {:>12} {:>10}",
        "mode", "rmse", "mean_rel", "spp"
    );
    for spp in [16u32, 64, 256] {
        let base = RenderParams {
            width: 48,
            height: 48,
            samples: spp,
            max_depth: 6,
            ..Default::default()
        };
        for mode in [SamplingMode::BsdfOnly, SamplingMode::NeeOnly] {
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
            println!(
                "{:<10} {:>10.5} {:>12.5} {:>10}",
                mode.name(),
                d.rmse / std::f64::consts::SQRT_2,
                d.mean_rel / std::f64::consts::SQRT_2,
                spp
            );
        }
    }
}
