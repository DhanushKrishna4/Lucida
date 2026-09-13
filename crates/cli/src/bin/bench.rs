//! Measure the renderer, so architectural changes can be argued from numbers.
//!
//! ```text
//! cargo run --release -p pt-cli --bin bench
//! cargo run --release -p pt-cli --bin bench -- --scene cornell-mesh --width 512
//! ```
//!
//! Timing is wall clock around a submit plus a device poll, so it includes
//! submission overhead. Each configuration is run once to warm up — the first
//! dispatch pays pipeline creation and shader compilation — and then several
//! times, reporting the median. The median rather than the mean because a single
//! scheduling hiccup should not move the number.

use pt_cli::{human_duration, Args};
use pt_core::integrator::{RenderParams, SamplingMode};
use pt_core::scenes::{self, SceneDef};
use pt_gpu::{Architecture, Gpu};
use std::time::Instant;

const VALUES: &[&str] = &[
    "scene",
    "width",
    "spp",
    "depth",
    "runs",
    "sampling",
    "architecture",
];
const FLAGS: &[&str] = &["help", "depth-sweep", "bvh-build"];

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse(VALUES, FLAGS)?;
    if args.has("help") {
        println!(
            "bench — measure GPU render time\n\n\
             \x20 --scene NAME     limit to one scene (default: all)\n\
             \x20 --width N        square resolution (default: 512)\n\
             \x20 --spp N          samples per dispatch (default: 8)\n\
             \x20 --depth N        max path length (default: 8)\n\
             \x20 --runs N         timed runs per configuration (default: 7)\n\
             \x20 --sampling MODE  bsdf | nee | mis (default: mis)\n\
             \x20 --architecture A megakernel | wavefront | both (default: both)\n\
             \x20 --depth-sweep    also measure how cost scales with max depth\n\
             \x20 --bvh-build      measure acceleration-structure build time instead"
        );
        return Ok(());
    }

    let width: u32 = args.parsed("width", 512)?;
    let spp: u32 = args.parsed("spp", 8)?;
    let depth: u32 = args.parsed("depth", 8)?;
    let runs: usize = args.parsed("runs", 7)?;
    let sampling = SamplingMode::parse(args.get("sampling").unwrap_or("mis"))
        .ok_or("sampling must be bsdf, nee or mis")?;
    // "both" is the default because the interesting number is the ratio, and a
    // ratio taken from two separate runs of this binary would be comparing
    // across two different thermal states of the same laptop.
    let arch_arg = args.get("architecture").unwrap_or("both");
    let archs: Vec<Architecture> = if arch_arg == "both" {
        Architecture::ALL.to_vec()
    } else {
        vec![Architecture::parse(arch_arg)
            .ok_or("architecture must be megakernel, wavefront or both")?]
    };

    let gpu = Gpu::new().map_err(|e| e.to_string())?;
    if args.has("bvh-build") {
        return bvh_build_benchmark(&gpu, args.get("scene"), runs);
    }
    println!(
        "device  {} ({:?})",
        gpu.adapter_info.name, gpu.adapter_info.backend
    );
    println!(
        "config  {width}x{width}, {spp} spp/dispatch, depth {depth}, {} sampling\n",
        sampling.name()
    );

    let all = scenes::all();
    let selected: Vec<&SceneDef> = match args.get("scene") {
        Some(name) => all.iter().filter(|s| s.name == name).collect::<Vec<_>>(),
        None => all.iter().collect(),
    };
    if selected.is_empty() {
        return Err("no matching scene".into());
    }

    print!("{:<14} {:>8} {:>7}", "scene", "tris", "lights");
    for a in &archs {
        print!("{:>14}", format!("{} ms", a.name()));
    }
    print!("{:>12}{:>10}", "Mpaths/s", "ns/path");
    if archs.len() > 1 {
        print!("{:>10}", "speedup");
    }
    println!();

    for def in &selected {
        let params = RenderParams {
            width,
            height: width,
            samples: spp,
            max_depth: depth,
            sampling,
            ..Default::default()
        };
        print!(
            "{:<14} {:>8} {:>7}",
            def.name,
            def.scene.blob.triangles.len(),
            def.scene.blob.lights.len(),
        );
        let mut times = Vec::new();
        for a in &archs {
            let ms = time_render(&gpu, def, &params, runs, *a)?;
            print!("{ms:>14.2}");
            times.push(ms);
        }
        // One path per sample per pixel. Not per *ray*: a path casts up to
        // `depth` extension rays plus a shadow ray per vertex, and how many it
        // actually casts is exactly what the architecture changes. Throughput is
        // quoted for whichever architecture was fastest.
        let paths = (width as f64) * (width as f64) * spp as f64;
        let best = times.iter().cloned().fold(f64::INFINITY, f64::min);
        print!(
            "{:>12.1}{:>10.1}",
            paths / (best * 1000.0),
            best * 1.0e6 / paths
        );
        if times.len() > 1 {
            // `Architecture::ALL` is megakernel first, so above 1.0 means the
            // wavefront won.
            print!("{:>10}", format!("{:.2}x", times[0] / times[1]));
        }
        println!();
    }

    if args.has("depth-sweep") {
        println!();
        println!("Cost versus maximum path length.");
        println!();
        println!("A megakernel runs one loop per pixel for the full bounce limit. Threads whose");
        println!("path terminated early sit in the same loop doing nothing, because the whole");
        println!("warp advances together — so cost tracks the *limit* rather than the average");
        println!("path length. That is the specific inefficiency the wavefront architecture");
        println!("removes, by compacting dead paths out between stages.");
        println!();
        print!("{:<14}", "scene");
        let depths = [1u32, 2, 4, 8, 16, 32];
        for d in depths {
            print!("{:>9}", format!("d={d}"));
        }
        println!("{:>12}", "32/1 ratio");

        for def in &selected {
            for &arch in &archs {
                print!("{:<14}", format!("{}/{}", def.name, &arch.name()[..4]));
                let mut first = 0.0;
                let mut last = 0.0;
                for (i, d) in depths.iter().enumerate() {
                    let params = RenderParams {
                        width,
                        height: width,
                        samples: spp,
                        max_depth: *d,
                        sampling,
                        ..Default::default()
                    };
                    let ms = time_render(&gpu, def, &params, runs.min(5), arch)?;
                    if i == 0 {
                        first = ms;
                    }
                    last = ms;
                    print!("{ms:>9.2}");
                }
                println!("{:>12.1}x", last / first.max(1e-9));
            }
        }
    }

    Ok(())
}

/// Median wall-clock milliseconds for one full render, after a warm-up run.
fn time_render(
    gpu: &Gpu,
    def: &SceneDef,
    params: &RenderParams,
    runs: usize,
    arch: Architecture,
) -> Result<f64, String> {
    // Warm up: the first dispatch pays pipeline creation and shader compilation,
    // which would otherwise dominate a short measurement entirely.
    pt_gpu::render_with(gpu, def, params, arch).map_err(|e| e.to_string())?;

    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Instant::now();
        pt_gpu::render_with(gpu, def, params, arch).map_err(|e| e.to_string())?;
        samples.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let _ = human_duration;
    Ok(samples[samples.len() / 2])
}

/// Measure acceleration-structure build time.
///
/// This is the number the linear BVH exists for. The binned-SAH builder makes a
/// better tree and cannot be parallelised — each split depends on the partition
/// its parent chose — so it sets the floor on how long a geometry change takes
/// to become renderable. The LBVH's stages are all maps and sorts, so the
/// question is simply how much of that floor the GPU removes, and what the
/// resulting tree costs to traverse.
///
/// Both are reported, because a builder that is ten times faster and produces a
/// tree twice as expensive to traverse is not obviously a win — it depends
/// entirely on how many frames are rendered per build.
fn bvh_build_benchmark(gpu: &Gpu, scene: Option<&str>, runs: usize) -> Result<(), String> {
    use pt_core::bvh::Bvh;
    use pt_core::lbvh::Lbvh;
    use std::time::Instant;

    let all = scenes::all();
    let selected: Vec<&SceneDef> = all
        .iter()
        .filter(|s| !s.scene.blob.triangles.is_empty())
        .filter(|s| scene.is_none_or(|n| s.name == n))
        .collect();
    if selected.is_empty() {
        return Err("no matching scene with triangles".into());
    }

    let median = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };

    println!("Acceleration-structure build time.\n");
    println!(
        "{:<14} {:>9} {:>12} {:>12} {:>12} {:>10} {:>11}",
        "scene", "tris", "sah cpu ms", "lbvh cpu ms", "lbvh gpu ms", "speedup", "cost ratio"
    );
    for def in &selected {
        let blob = &def.scene.blob;
        let (t, p) = (&blob.triangles, &blob.positions);

        let sah_ms = median(
            (0..runs)
                .map(|_| {
                    let t0 = Instant::now();
                    let b = Bvh::build(t, p);
                    std::hint::black_box(&b);
                    t0.elapsed().as_secs_f64() * 1000.0
                })
                .collect(),
        );
        let lbvh_cpu_ms = median(
            (0..runs)
                .map(|_| {
                    let t0 = Instant::now();
                    let b = Lbvh::build(t, p);
                    std::hint::black_box(&b);
                    t0.elapsed().as_secs_f64() * 1000.0
                })
                .collect(),
        );

        // Pipelines built once, outside the timed loop. Compiling six kernels
        // costs ~110 ms and a renderer rebuilding geometry every frame does not
        // pay it every frame; including it would measure the shader compiler.
        let builder = pt_gpu::lbvh::LbvhBuilder::new(gpu).map_err(|e| e.to_string())?;
        builder.build(gpu, t, p).map_err(|e| e.to_string())?;
        let lbvh_gpu_ms = median(
            (0..runs)
                .map(|_| {
                    let t0 = Instant::now();
                    let b = builder.build(gpu, t, p).unwrap();
                    std::hint::black_box(&b);
                    t0.elapsed().as_secs_f64() * 1000.0
                })
                .collect(),
        );

        let sah = Bvh::build(t, p);
        let lin = Lbvh::build(t, p);
        println!(
            "{:<14} {:>9} {:>12.2} {:>12.2} {:>12.2} {:>10.2}x {:>10.2}x",
            def.name,
            t.len(),
            sah_ms,
            lbvh_cpu_ms,
            lbvh_gpu_ms,
            sah_ms / lbvh_gpu_ms,
            lin.stats.sah_cost / sah.stats.sah_cost,
        );
    }
    println!(
        "\nspeedup is sah cpu / lbvh gpu; cost ratio is the LBVH tree's SAH traversal\n\
         cost relative to the binned-SAH tree's. The GPU timing includes the\n\
         readback and the host-side relayout, which is the honest number for\n\
         \"how long until this tree can be rendered\"."
    );
    Ok(())
}
