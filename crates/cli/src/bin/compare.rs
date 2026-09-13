//! Compare two HDR renders.
//!
//! ```text
//! cargo run --release -p pt-cli --bin compare -- out/cornell-cpu.pfm out/cornell-gpu.pfm
//! ```
//!
//! Also writes a false-colour difference image when `--diff` is given, which is
//! usually more informative than the summary numbers: a uniform haze means a
//! numerical difference, a localised blob means a geometry or traversal bug, and
//! a difference concentrated on one material means a BSDF bug.

use pt_core::image::{self, compare};
use pt_core::integrator::Film;

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    // Walk the arguments rather than filtering on a "--" prefix: a naive filter
    // treats the *value* of `--diff out.png` as a positional argument too.
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut positional: Vec<String> = Vec::new();
    let mut diff_out: Option<String> = None;
    let mut scale: f32 = 20.0;
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--diff" => {
                diff_out = Some(argv.get(i + 1).ok_or("`--diff` expects a path")?.clone());
                i += 2;
            }
            "--scale" => {
                scale = argv
                    .get(i + 1)
                    .ok_or("`--scale` expects a number")?
                    .parse()
                    .map_err(|_| "`--scale` expects a number".to_string())?;
                i += 2;
            }
            other if other.starts_with("--") => return Err(format!("unknown option `{other}`")),
            other => {
                positional.push(other.to_string());
                i += 1;
            }
        }
    }
    if positional.len() != 2 {
        return Err("usage: compare <a.pfm> <b.pfm> [--diff out.png] [--scale N]".into());
    }

    let a = image::read_pfm(&positional[0]).map_err(|e| format!("{}: {e}", positional[0]))?;
    let b = image::read_pfm(&positional[1]).map_err(|e| format!("{}: {e}", positional[1]))?;
    let d = compare(&a, &b)?;

    println!("{} vs {}", positional[0], positional[1]);
    pt_cli::print_diff("a", "b", &d);
    pt_cli::verdict(&d, None);

    if let Some(path) = diff_out {
        let mut img = Film::new(a.width, a.height);
        for i in 0..a.data.len() {
            // Absolute difference, amplified. Not relative: for a visual read we
            // want to see *where* energy differs, and a relative map is dominated
            // by near-black pixels.
            img.data[i] = (a.data[i] - b.data[i]).abs() * scale;
        }
        // Clamp, deliberately: a difference map must not have a curve applied
        // to it, or the amplification factor stops meaning anything.
        image::write_png(&path, &img, 1.0, pt_core::tonemap::Tonemap::Clamp)
            .map_err(|e| format!("writing {path}: {e}"))?;
        println!("\n  wrote {path} (absolute difference, {scale}x)");
    }
    Ok(())
}
