//! Re-run one chi-squared configuration across many seeds.
//!
//! A single low p-value is not evidence of a bug — with dozens of
//! configurations, the smallest of them is *expected* to be small. The way to
//! tell a real bias from chance is to resample: under the null hypothesis
//! p-values are uniform on [0, 1], so a correct sampler produces a scatter, and
//! a biased one produces consistently small values whatever the seed.
//!
//! `cargo run --release -p pt-core --example chi2_seed_sweep`

use glam::Vec3;
use pt_core::bsdf;
use pt_core::chi2::chi2_test;

fn main() {
    // The configuration that came closest to failing in the suite.
    let roughness = 0.75f32;
    let cos_o = 0.7f32;
    let alpha = bsdf::roughness_to_alpha(roughness);
    let sin_o = (1.0 - cos_o * cos_o).sqrt();
    let wo = Vec3::new(sin_o, 0.0, cos_o);

    println!("GGX roughness {roughness}, cos_theta_o {cos_o}, across 16 seeds:");
    let mut ps = Vec::new();
    for seed in 0..16u32 {
        let r = chi2_test(
            |rng| {
                let s = bsdf::specular::sample_single(alpha, Vec3::ONE, wo, rng.next_vec2());
                if s.is_valid() {
                    Some(s.wi)
                } else {
                    None
                }
            },
            |wi| bsdf::specular::pdf(alpha, wo, wi),
            128,
            256,
            600_000,
            0x5EED_0000 + seed,
        );
        println!(
            "  seed {seed:>2}: p = {:.4}  (chi2 {:.0}, dof {})",
            r.p_value, r.statistic, r.degrees_of_freedom
        );
        ps.push(r.p_value);
    }

    ps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = ps[ps.len() / 2];
    let below_01 = ps.iter().filter(|&&p| p < 0.01).count();
    println!();
    println!("  median p = {median:.3}   (uniform under the null, so expect about 0.5)");
    println!("  {below_01} of 16 below 0.01   (expect about 0.16)");
    if below_01 <= 1 && median > 0.2 {
        println!("  -> consistent with chance; the sampler and its pdf agree.");
    } else {
        println!("  -> NOT consistent with chance; investigate.");
    }
}
