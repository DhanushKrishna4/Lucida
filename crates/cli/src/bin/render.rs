//! Offline renderer: produces reference images from the CPU tracer, the native
//! GPU tracer, or both.
//!
//! ```text
//! cargo run --release -p pt-cli --bin render -- --device cpu --spp 512 --out out/cpu
//! cargo run --release -p pt-cli --bin render -- --device gpu --spp 512 --out out/gpu
//! cargo run --release -p pt-cli --bin render -- --device both --spp 512 --out out/cornell
//! ```
//!
//! `--device both` renders the identical scene, resolution, seed and sample
//! count on each device and prints the comparison — the core check for build
//! step 3.

use pt_cli::{human_duration, Args};
use pt_core::image;
use pt_core::integrator::{self, RenderParams, SamplingMode};
use pt_core::scenes;
use pt_core::tonemap::Tonemap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

const VALUES: &[&str] = &[
    "scene",
    "width",
    "height",
    "spp",
    "depth",
    "seed",
    "device",
    "out",
    "threads",
    "exposure",
    "tonemap",
    "sampling",
    "architecture",
    "bvh",
    "sampler",
    "denoise",
    "mode",
];
const FLAGS: &[&str] = &["quiet", "help", "list-scenes", "noise-floor"];

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse(VALUES, FLAGS)?;
    if args.has("help") {
        print_help();
        return Ok(());
    }
    if args.has("list-scenes") {
        for s in scenes::all() {
            println!("{:<16} {}", s.name, s.description);
        }
        return Ok(());
    }

    let scene_name = args.get("scene").unwrap_or("cornell-box");
    let mut def = scenes::by_name(scene_name)
        .ok_or_else(|| format!("unknown scene `{scene_name}` (try --list-scenes)"))?;

    let sampler = {
        let name = args.get("sampler").unwrap_or("sobol");
        pt_core::sobol::SamplerKind::parse(name)
            .ok_or_else(|| format!("unknown sampler `{name}`; expected: sobol, independent"))?
    };
    let width: u32 = args.parsed("width", 512)?;
    let params = RenderParams {
        width,
        height: args.parsed("height", width)?,
        samples: args.parsed("spp", 256)?,
        max_depth: args.parsed("depth", 8)?,
        frame_seed: args.parsed("seed", 0x5eed_1234u32)?,
        sample_offset: 0,
        threads: args.parsed("threads", 0usize)?,
        sampling: {
            let name = args.get("sampling").unwrap_or("nee");
            SamplingMode::parse(name)
                .ok_or_else(|| format!("unknown sampling mode `{name}`; expected: bsdf, nee"))?
        },
        sampler,
    };
    let device = args.get("device").unwrap_or("cpu");
    let architecture = {
        // The megakernel, matching the browser's default. At the default depth
        // of 8 it is the faster of the two on four of the six scenes, and it is
        // the path every CPU/GPU agreement test is written against. `bench`
        // prints the comparison that justifies picking the other one.
        let name = args.get("architecture").unwrap_or("megakernel");
        pt_gpu::Architecture::parse(name).ok_or_else(|| {
            format!("unknown architecture `{name}`; expected: megakernel, wavefront")
        })?
    };
    let bvh_builder = {
        let name = args.get("bvh").unwrap_or("sah");
        pt_core::scene::BvhBuilder::parse(name)
            .ok_or_else(|| format!("unknown bvh builder `{name}`; expected: sah, lbvh"))?
    };
    // Rebuild only when asked for something other than the default, so the
    // common path does not pay for a second build of a tree `by_name` already
    // produced. Lights are rebuilt with it: compaction permutes the triangle
    // array and the light list holds flattened copies of emissive triangles.
    if bvh_builder != pt_core::scene::BvhBuilder::BinnedSah {
        def.scene.build_bvh_with(bvh_builder);
        def.scene.blob.lights = pt_core::light::build_lights(&def.scene.blob);
    }
    let exposure: f32 = args.parsed("exposure", 1.0)?;
    let tonemap_name = args.get("tonemap").unwrap_or("clamp");
    let tonemap = Tonemap::parse(tonemap_name).ok_or_else(|| {
        format!(
            "unknown tone map `{tonemap_name}`; expected one of: {}",
            Tonemap::ALL.map(|t| t.name()).join(", ")
        )
    })?;
    let out = args.get("out").unwrap_or("out/render");
    let quiet = args.has("quiet");

    if let Some(parent) = std::path::Path::new(out).parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("cannot create {parent:?}: {e}"))?;
    }

    if !quiet {
        println!(
            "scene   {}\nsize    {}x{}\nsamples {} spp, max depth {}\nseed    0x{:08x}",
            def.name,
            params.width,
            params.height,
            params.samples,
            params.max_depth,
            params.frame_seed
        );
        println!("sampling {}", params.sampling.name());
        if matches!(device, "gpu" | "both") {
            println!("gpu arch {}", architecture.name());
        }

        let b = &def.scene.blob;
        if b.triangles.is_empty() {
            println!(
                "geom    {} analytic primitives, {} lights (no BVH)",
                b.primitives.len(),
                b.lights.len()
            );
        } else {
            let st = def.scene.bvh.stats;
            println!(
                "geom    {} triangles, {} analytic primitives, {} lights",
                b.triangles.len(),
                b.primitives.len(),
                b.lights.len()
            );
            println!(
                "bvh     {} nodes, {} leaves, depth {}, mean leaf {:.1}",
                st.nodes, st.leaves, st.max_depth, st.mean_leaf_size
            );
            // SAH cost is the machine-independent measure of tree quality: the
            // expected number of triangle tests per ray. Reported against brute
            // force so the number means something on its own.
            println!(
                "        SAH cost {:.2} vs {} brute force ({:.0}x), built in {}",
                st.sah_cost,
                st.triangles,
                st.triangles as f32 / st.sah_cost.max(1e-6),
                human_duration(std::time::Duration::from_secs_f64(st.build_seconds))
            );
        }
    }

    let want_cpu = matches!(device, "cpu" | "both");
    let want_gpu = matches!(device, "gpu" | "both");
    if !want_cpu && !want_gpu {
        return Err(format!(
            "`--device {device}` must be one of: cpu, gpu, both"
        ));
    }

    let mut cpu_film = None;
    // Off unless asked for, deliberately. Denoising is a display-time product
    // that trades variance for bias, and this renderer's default output should
    // be what it actually computed. Measured, it helps the typical pixel 2.4x at
    // 4 spp and *hurts* above roughly 32 — see the README.
    let render_mode = {
        let name = args.get("mode").unwrap_or("beauty");
        pt_core::diagnostic::RenderMode::parse(name).ok_or_else(|| {
            format!("unknown mode `{name}`; expected: beauty, normal, albedo, depth, heat")
        })?
    };
    let denoise_passes: u32 = args.parsed("denoise", 0u32)?;
    let denoise = (denoise_passes > 0).then(|| pt_core::denoise::DenoiseParams {
        iterations: denoise_passes,
        ..Default::default()
    });

    if want_cpu {
        let t0 = Instant::now();
        let last = AtomicU32::new(0);
        let film = integrator::render_with_progress(&def, &params, |done, total| {
            if quiet {
                return;
            }
            let pct = done * 100 / total;
            // Only redraw on a whole-percent change; progress printing from many
            // threads is otherwise a surprisingly large share of the runtime at
            // small resolutions.
            if pct > last.swap(pct, Ordering::Relaxed) {
                eprint!("\r  cpu  {pct:3}%");
            }
        });
        if !quiet {
            eprintln!("\r  cpu  done in {}", human_duration(t0.elapsed()));
        }
        let film = match (&denoise, render_mode.is_data()) {
            // A diagnostic replaces the image entirely, so it also replaces the
            // denoiser: filtering a normal map would be meaningless.
            (_, true) => {
                let (_, guides) = integrator::render_with_guides(&def, &params);
                let _ = film;
                diagnostic_film(&params, &guides, render_mode)
            }
            (Some(d), false) => {
                // Re-render with guides: the banded parallel path does not
                // produce them, and the denoiser needs them.
                let (noisy, guides) = integrator::render_with_guides(&def, &params);
                let _ = film;
                pt_core::denoise::denoise(&noisy, &guides, d)
            }
            (None, false) => film,
        };
        // A diagnostic is a measurement, not radiance: exposure and a filmic
        // curve would make it misreport its own values.
        let (ex, tm) = if render_mode.is_data() {
            (1.0, Tonemap::Clamp)
        } else {
            (exposure, tonemap)
        };
        write_outputs(&format!("{out}-cpu"), &film, ex, tm, quiet)?;
        cpu_film = Some(film);
    }

    let mut gpu_film = None;
    if want_gpu {
        let t0 = Instant::now();
        let gpu = pt_gpu::Gpu::new().map_err(|e| e.to_string())?;
        // The convergence figure comes free with an undenoised render: the sum of
        // squares is already accumulated, so it is one reduction over a buffer
        // the GPU has just finished. Not reported for a denoised render, where
        // the output is a filtered image whose per-pixel spread is no longer a
        // standard error of anything.
        let mut convergence = None;
        let film = match &denoise {
            Some(d) => pt_gpu::render_native_denoised(&gpu, &def, &params, d)
                .map_err(|e| e.to_string())?,
            None => {
                let (f, e) = pt_gpu::render_with_measured(&gpu, &def, &params, architecture)
                    .map_err(|e| e.to_string())?;
                convergence = Some(e);
                f
            }
        };
        if !quiet {
            eprintln!("  gpu  done in {}", human_duration(t0.elapsed()));
        }
        if let Some(e) = convergence.filter(|e| *e > 0.0) {
            // Measured, not modelled. See shaders/stats/convergence.wgsl.
            println!("noise    {:.2}% mean relative standard error", e * 100.0);
            if params.sampler == pt_core::sobol::SamplerKind::Sobol {
                // Saying so, because the figure assumes independent samples and
                // Sobol's are deliberately not — measured at 1.27x conservative
                // on the Cornell box.
                println!("         (conservative: the estimate assumes independent samples)");
            }
        }
        let film = if render_mode.is_data() {
            let (_, guides) = integrator::render_with_guides(&def, &params);
            diagnostic_film(&params, &guides, render_mode)
        } else {
            film
        };
        let (ex, tm) = if render_mode.is_data() {
            (1.0, Tonemap::Clamp)
        } else {
            (exposure, tonemap)
        };
        write_outputs(&format!("{out}-gpu"), &film, ex, tm, quiet)?;
        gpu_film = Some(film);
    }

    if let (Some(a), Some(b)) = (&cpu_film, &gpu_film) {
        println!();
        let d = pt_core::image::compare(a, b)?;
        println!("cpu vs gpu");
        pt_cli::print_diff("cpu", "gpu", &d);

        // The absolute size of a difference means little on its own. What
        // matters is how it compares to the difference between two runs of the
        // *same* renderer with different seeds — that is the scale of Monte
        // Carlo noise at this sample count, and a genuine agreement should be
        // orders of magnitude below it.
        let nf = if args.has("noise-floor") {
            if !quiet {
                eprintln!("  measuring the Monte Carlo noise floor (one more cpu render)...");
            }
            Some(pt_cli::noise_floor(&def, &params, a)?)
        } else {
            None
        };
        pt_cli::verdict(&d, nf.as_ref());
        if nf.is_none() {
            println!("        (pass --noise-floor to compare this against Monte Carlo noise)");
        }
    }

    Ok(())
}

/// Apply a diagnostic mode to a render's guide channels.
///
/// The scene's own scale sets the depth normalisation, taken from the furthest
/// hit rather than from a constant, so one image works for a Cornell box
/// measured in hundreds and a sphere measured in ones.
fn diagnostic_film(
    params: &RenderParams,
    guides: &pt_core::denoise::Guides,
    mode: pt_core::diagnostic::RenderMode,
) -> integrator::Film {
    let depth_scale = guides
        .depth
        .iter()
        .copied()
        .fold(0.0f32, f32::max)
        .max(1.0e-6);
    let mut film = integrator::Film::new(params.width, params.height);
    for (i, px) in film.data.iter_mut().enumerate() {
        *px = pt_core::diagnostic::shade(
            mode,
            glam::Vec3::ZERO,
            guides.albedo[i],
            guides.normal[i],
            guides.depth[i],
            guides.steps[i],
            depth_scale,
        );
    }
    film
}

fn write_outputs(
    stem: &str,
    film: &integrator::Film,
    exposure: f32,
    tonemap: Tonemap,
    quiet: bool,
) -> Result<(), String> {
    let pfm = format!("{stem}.pfm");
    let png = format!("{stem}.png");
    image::write_pfm(&pfm, film).map_err(|e| format!("writing {pfm}: {e}"))?;
    image::write_png(&png, film, exposure, tonemap).map_err(|e| format!("writing {png}: {e}"))?;
    if !quiet {
        println!("  wrote {pfm} and {png}");
    }
    Ok(())
}

fn print_help() {
    println!(
        "render — offline path tracer\n\n\
         OPTIONS\n\
         \x20 --scene NAME      scene to render (default: cornell-box)\n\
         \x20 --width N         image width in pixels (default: 512)\n\
         \x20 --height N        image height (default: same as width)\n\
         \x20 --spp N           samples per pixel (default: 256)\n\
         \x20 --depth N         maximum path length (default: 8)\n\
         \x20 --seed N          RNG seed; change it to measure the noise floor\n\
         \x20 --device D        cpu | gpu | both (default: cpu)\n\
         \x20 --out PATH        output stem; writes PATH-cpu.pfm/.png etc\n\
         \x20 --threads N       CPU worker threads (default: one per core)\n\
         \x20 --exposure F      exposure applied to the PNG only, in scene-linear (default: 1.0)\n\
         \x20 --mode M          beauty | normal | albedo | depth | heat (default: beauty)\n\
         \x20                   diagnostics bypass tone mapping; heat is BVH node visits\n\
         \x20 --denoise N       run N a-trous denoise passes on the output (default: 0)\n\
         \x20                   helps below ~32 spp and hurts above it; see the README\n\
         \x20 --sampler S       sobol | independent (default: sobol)\n\
         \x20                   sobol is Owen-scrambled and converges faster on smooth\n\
         \x20                   integrands; independent is the 1/sqrt(N) baseline\n\
         \x20 --bvh B           sah | lbvh (default: sah)\n\
         \x20                   sah builds the better tree and is sequential; lbvh\n\
         \x20                   sorts Morton codes and parallelises. See `bench --bvh-build`\n\
         \x20 --architecture A  megakernel | wavefront (default: megakernel)\n\
         \x20                   same image either way; they differ in how the work\n\
         \x20                   is scheduled. See `bench` for which wins where\n\
         \x20 --sampling MODE   bsdf | nee (default: nee)\n\
         \x20                   bsdf finds light by walking into emitters; nee connects\n\
         \x20                   every path vertex to a sampled point on a light\n\
         \x20 --tonemap NAME    clamp | reinhard | aces | agx (default: clamp)\n\
         \x20                   PNG only; the PFM is always untouched linear HDR\n\
         \x20 --noise-floor     also render with a different seed, to show the scale\n\
         \x20                   of Monte Carlo noise the cpu/gpu difference sits under\n\
         \x20 --quiet           suppress progress output\n\
         \x20 --list-scenes     print available scenes and exit"
    );
}
