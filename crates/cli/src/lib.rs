//! Shared helpers for the command-line tools.

use std::collections::HashMap;

/// A deliberately tiny `--key value` / `--flag` parser.
///
/// `clap` is not on the project's allowed dependency list and this is all the
/// tools need. Unknown keys are an error rather than being ignored, because a
/// silently-dropped `--spp 4096` that renders 256 samples instead is exactly the
/// kind of thing that wastes an afternoon.
pub struct Args {
    values: HashMap<String, String>,
    flags: Vec<String>,
}

impl Args {
    pub fn parse(known_values: &[&str], known_flags: &[&str]) -> Result<Self, String> {
        let mut values = HashMap::new();
        let mut flags = Vec::new();
        let argv: Vec<String> = std::env::args().skip(1).collect();
        let mut i = 0;
        while i < argv.len() {
            let a = &argv[i];
            let Some(key) = a.strip_prefix("--") else {
                return Err(format!("unexpected positional argument `{a}`"));
            };
            if known_flags.contains(&key) {
                flags.push(key.to_string());
                i += 1;
            } else if known_values.contains(&key) {
                let v = argv
                    .get(i + 1)
                    .ok_or_else(|| format!("`--{key}` expects a value"))?;
                values.insert(key.to_string(), v.clone());
                i += 2;
            } else {
                return Err(format!(
                    "unknown option `--{key}`\n  values: {}\n  flags:  {}",
                    known_values.join(", "),
                    known_flags.join(", ")
                ));
            }
        }
        Ok(Self { values, flags })
    }

    pub fn has(&self, flag: &str) -> bool {
        self.flags.iter().any(|f| f == flag)
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(|s| s.as_str())
    }

    pub fn parsed<T: std::str::FromStr>(&self, key: &str, default: T) -> Result<T, String> {
        match self.values.get(key) {
            None => Ok(default),
            Some(v) => v.parse().map_err(|_| {
                format!(
                    "`--{key} {v}` is not a valid {}",
                    std::any::type_name::<T>()
                )
            }),
        }
    }
}

/// Format a duration the way a person reads it.
pub fn human_duration(d: std::time::Duration) -> String {
    let s = d.as_secs_f64();
    if s < 1.0 {
        format!("{:.0} ms", s * 1000.0)
    } else if s < 60.0 {
        format!("{s:.2} s")
    } else {
        format!("{} m {:.1} s", (s / 60.0) as u64, s % 60.0)
    }
}

// ---------------------------------------------------------------------------
// Image comparison reporting
// ---------------------------------------------------------------------------

use pt_core::image::{compare, ImageDiff};
use pt_core::integrator::{self, Film, RenderParams};
use pt_core::scenes::SceneDef;

/// Absolute agreement threshold, used when no noise floor has been measured.
///
/// Because the CPU and GPU tracers draw from *bit-identical* random streams
/// (see `pt_core::rng`), their only legitimate source of disagreement is
/// floating-point reassociation — chiefly fused multiply-add contraction, which
/// WGSL permits and the CPU does not do.
///
/// That error is tiny per operation, but it does not stay tiny everywhere: at a
/// **silhouette**, a difference of one ULP decides whether a ray hits a
/// primitive or slips past it, and the two devices then follow entirely
/// different paths for that sample. So the disagreement is concentrated in a
/// small number of pixels rather than spread evenly, and its magnitude scales
/// with how much silhouette the scene has. Analytic geometry lands around
/// 2e-4; a scene of 10 thousand triangles, with a facet boundary every few
/// pixels, lands closer to 2e-3 for exactly the same reason and with exactly
/// the same (correct) renderer.
///
/// Monte Carlo noise, for comparison, is on the order of 1e-1. The gap between
/// "the same computation, reassociated" and "two independent estimates of the
/// same integral" is what this threshold sits inside.
pub const AGREEMENT_MEAN_REL: f64 = 4.0e-3;

/// Agreement threshold as a fraction of the Monte Carlo noise floor.
///
/// This is the criterion that actually means something, and it is scene
/// independent: a difference at a few thousandths of the noise floor cannot be
/// a light-transport bug, whatever its absolute size.
pub const AGREEMENT_NOISE_FRACTION: f64 = 0.05;

pub fn print_diff(label_a: &str, label_b: &str, d: &ImageDiff) {
    println!("  {label_a} mean radiance   {:.6}", d.mean_a);
    println!("  {label_b} mean radiance   {:.6}", d.mean_b);
    println!(
        "  energy ratio         {:.6}  ({label_b}/{label_a})",
        d.mean_b / d.mean_a.max(1e-12)
    );
    println!("  rmse                 {:.6e}", d.rmse);
    println!(
        "  max abs diff         {:.6e}  at pixel ({}, {})",
        d.max_abs, d.max_abs_at.0, d.max_abs_at.1
    );
    println!("  mean relative diff   {:.6e}", d.mean_rel);
}

/// Render the same scene again with a different seed, to establish how much two
/// independent estimates of the same integral differ. That is the scale against
/// which any CPU/GPU difference should be judged.
pub fn noise_floor(
    def: &SceneDef,
    params: &RenderParams,
    reference: &Film,
) -> Result<ImageDiff, String> {
    let other = integrator::render(
        def,
        &RenderParams {
            frame_seed: params.frame_seed ^ 0x00AB_CDEF,
            ..*params
        },
    );
    compare(reference, &other)
}

/// Print a verdict on a comparison.
///
/// When `noise_floor` is supplied (two renders of the same scene differing only
/// in seed), the verdict is stated **relative to it**. That question — "is this
/// difference far below what two runs of the same renderer show?" — is scene
/// independent, where an absolute threshold is not: see the note on
/// [`AGREEMENT_MEAN_REL`] about why triangle meshes legitimately sit an order of
/// magnitude higher than analytic geometry.
pub fn verdict(d: &ImageDiff, noise_floor: Option<&ImageDiff>) {
    println!();

    if let Some(nf) = noise_floor {
        println!("  monte carlo noise floor (same renderer, different seed)");
        println!("    rmse               {:.6e}", nf.rmse);
        println!("    mean relative diff {:.6e}", nf.mean_rel);
        println!();
        let ratio = d.mean_rel / nf.mean_rel.max(1e-18);
        println!("  difference is {ratio:.4}x the noise floor");
        if ratio <= AGREEMENT_NOISE_FRACTION {
            println!(
                "  PASS  {:.2e} mean relative error, {ratio:.4}x the Monte Carlo noise floor.",
                d.mean_rel
            );
            println!("        Nothing at this scale can be a light-transport bug: the devices");
            println!("        are running the same computation, reassociated.");
        } else {
            println!(
                "  FAIL  {:.2e} mean relative error is {ratio:.3}x the noise floor.",
                d.mean_rel
            );
            println!("        A genuine difference between the two renderers. Check, in order:");
            println!("          1. the order random numbers are drawn in");
            println!("          2. RNG seeding");
            println!("          3. GPU buffer layout (cargo test -p pt-core gpu_layout)");
            println!("          4. the scene actually uploaded — primitive and node counts");
        }
        return;
    }

    if d.mean_rel <= AGREEMENT_MEAN_REL {
        println!(
            "  PASS  the two renders agree to {:.2e} mean relative error (threshold {AGREEMENT_MEAN_REL:.0e}).",
            d.mean_rel
        );
        println!("        That is float-reassociation level, not Monte Carlo level:");
        println!("        the devices are running the same computation.");
        if d.mean_rel > 1.0e-3 {
            println!("        (the higher end of the range is normal for triangle meshes —");
            println!("         more silhouette edges means more samples where one ULP decides");
            println!("         hit or miss. Pass --noise-floor for a scene-independent verdict.)");
        }
    } else if d.mean_rel < 2.0e-2 {
        println!("  MARGINAL  {:.2e} mean relative error.", d.mean_rel);
        println!("        Larger than float reassociation should produce, but far below");
        println!("        Monte Carlo scale. Suspect a small numerical difference: an");
        println!("        epsilon, a normalize, or a differently-ordered dot product.");
        println!("        Re-run with --noise-floor for a scene-independent verdict.");
    } else {
        println!("  FAIL  {:.2e} mean relative error.", d.mean_rel);
        println!("        This is Monte Carlo scale, which means the two devices are not");
        println!("        drawing the same random stream, or are not tracing the same");
        println!("        scene. Check: RNG seeding, the order random numbers are drawn");
        println!("        in, and the buffer layout.");
    }
}
