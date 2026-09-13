use glam::Vec3;
use pt_core::image::linear_to_srgb;
use pt_core::tonemap::Tonemap;

fn main() {
    println!("middle grey (scene-linear 0.18):");
    for t in Tonemap::ALL {
        let y = t.apply(Vec3::splat(0.18)).y;
        println!(
            "  {:<9} display-linear {:.4}  sRGB-encoded {:.4}",
            t.name(),
            y,
            linear_to_srgb(y)
        );
    }
    println!("\nbright saturated red (12.0, 1.2, 0.8) -> saturation (max-min)/max:");
    for t in Tonemap::ALL {
        let c = t.apply(Vec3::new(12.0, 1.2, 0.8));
        let sat = (c.max_element() - c.min_element()) / c.max_element().max(1e-6);
        // Hue angle in the r-g-b plane, as a crude hue-shift proxy.
        let hue = (c.z - c.y).atan2(c.x - 0.5 * (c.y + c.z)).to_degrees();
        println!(
            "  {:<9} {:?}  sat {:.3}  hue {:+.1} deg",
            t.name(),
            c.to_array().map(|v| (v * 1000.0).round() / 1000.0),
            sat,
            hue
        );
    }
    let c = Vec3::new(12.0, 1.2, 0.8);
    let sat = (c.max_element() - c.min_element()) / c.max_element();
    let hue = (c.z - c.y).atan2(c.x - 0.5 * (c.y + c.z)).to_degrees();
    println!(
        "  {:<9} (input)                sat {:.3}  hue {:+.1} deg",
        "none", sat, hue
    );

    println!("\nrolloff — input 1.0 and 4.0:");
    for t in Tonemap::ALL {
        println!(
            "  {:<9} f(1.0) = {:.4}   f(4.0) = {:.4}",
            t.name(),
            t.apply(Vec3::ONE).y,
            t.apply(Vec3::splat(4.0)).y
        );
    }
}
