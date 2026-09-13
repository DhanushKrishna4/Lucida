use glam::Vec3;
use pt_core::bsdf::*;

fn main() {
    println!(
        "Schlick vs exact dielectric Fresnel (ior 1.5, F0 = {:.4}):",
        f0_from_ior(1.5)
    );
    let f0 = f0_from_ior(1.5);
    let (mut worst, mut worst_at) = (0.0f32, 0.0f32);
    for i in 0..=1000 {
        let cos = i as f32 / 1000.0;
        let d = (fresnel_dielectric(cos, 1.5) - fresnel_schlick(Vec3::splat(f0), cos).x).abs();
        if d > worst {
            worst = d;
            worst_at = cos;
        }
    }
    println!("  max absolute error {worst:.5} at cos_theta = {worst_at:.3}");

    println!("\nNDF normalisation by tan-space quadrature (should be 1.0 for every alpha):");
    for &a in &[1e-5f32, 1e-4, 3e-4, 5e-4, 1e-3, 2e-3, 4e-3, 1e-2, 1e-1, 1.0] {
        println!(
            "  alpha {a:<9.5} (roughness {:<7.4}) integral {:.6}",
            a.sqrt(),
            ndf_integral(a)
        );
    }

    let cosines = [0.95f32, 0.8, 0.6, 0.4, 0.2, 0.1];
    let roughnesses = [0.05f32, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0];

    println!("\nWHITE FURNACE TEST — directional albedo E(wo), Fresnel forced to white.");
    println!("An energy-conserving BSDF gives exactly 1.0 everywhere; the object is invisible.");

    println!("\n  single scattering only (what GGX does on its own):");
    print!("  {:<10}", "roughness");
    for &c in &cosines {
        print!("{:>9}", format!("cos={c}"));
    }
    println!();
    for &r in &roughnesses {
        print!("  {r:<10.2}");
        for &c in &cosines {
            print!(
                "{:>9.4}",
                specular_directional_albedo(roughness_to_alpha(r), c, 300_000, 5)
            );
        }
        println!();
    }

    println!("\n  with multiple-scattering compensation:");
    print!("  {:<10}", "roughness");
    for &c in &cosines {
        print!("{:>9}", format!("cos={c}"));
    }
    println!();
    let mut worst = 0.0f32;
    for &r in &roughnesses {
        print!("  {r:<10.2}");
        for &c in &cosines {
            let e = compensated_albedo(r, c, 300_000, 7);
            worst = worst.max((e - 1.0).abs());
            print!("{:>9.4}", e);
        }
        println!();
    }
    println!("\n  worst deviation from 1.0: {worst:.4}");
}

/// Directional albedo of the *compensated* lobe, with Fresnel forced to white.
fn compensated_albedo(roughness: f32, cos_theta_o: f32, samples: u32, seed: u32) -> f32 {
    let mut rng = pt_core::rng::Rng::new(seed, 0, 0);
    let sin = (1.0 - cos_theta_o * cos_theta_o).max(0.0).sqrt();
    let wo = Vec3::new(sin, 0.0, cos_theta_o);
    let mut sum = 0.0f64;
    for _ in 0..samples {
        let s = specular::sample(roughness, Vec3::ONE, wo, rng.next_vec2());
        if s.is_valid() {
            sum += s.weight.x as f64;
        }
    }
    (sum / samples as f64) as f32
}

/// Exact-ish quadrature of `integral D(m)(n.m) dm` in tan space.
///
/// Substituting `x = tan(theta)/alpha` turns the integrand into
/// `t / (1 + t^2)^2`, which is smooth and **independent of alpha** — so one
/// fixed grid is accurate for every roughness, unlike uniform hemisphere
/// sampling, which cannot resolve a peak of angular width `alpha`.
fn ndf_integral(alpha: f32) -> f64 {
    let n = 400_000;
    let t_max = 5_000.0f64;
    let h = t_max / n as f64;
    let mut sum = 0.0f64;
    for i in 0..=n {
        let t = i as f64 * h;
        let theta = (alpha as f64 * t).atan();
        let (sin, cos) = theta.sin_cos();
        // dtheta/dt = alpha / (1 + (alpha t)^2)
        let dtheta = alpha as f64 / (1.0 + (alpha as f64 * t).powi(2));
        let f = ggx_d(alpha, cos as f32) as f64 * cos * sin * dtheta;
        let w = if i == 0 || i == n {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        sum += w * f;
    }
    sum * h / 3.0 * 2.0 * std::f64::consts::PI
}
