//! Build step 3: the GPU megakernel must reproduce the CPU reference tracer.
//!
//! These tests are the reason the native harness exists. They run the real WGSL
//! against the real CPU oracle in `cargo test`, with no browser involved, so a
//! light-transport regression is caught in seconds.
//!
//! Every test skips (rather than fails) when no GPU adapter is available, so the
//! suite still runs on a headless CI box. A skip prints a loud notice — a test
//! that silently passes by doing nothing is worse than no test.

use pt_core::image::compare;
use pt_core::integrator::{self, RenderParams};
use pt_core::scenes;
use pt_gpu::{render_native_on, Gpu};

/// Shared device. Adapter creation is slow enough that per-test setup dominates.
fn gpu() -> Option<Gpu> {
    match Gpu::new() {
        Ok(g) => {
            eprintln!(
                "gpu: {} ({:?})",
                g.adapter_info.name, g.adapter_info.backend
            );
            Some(g)
        }
        Err(e) => {
            eprintln!("\n*** SKIPPING GPU TEST: {e} ***\n");
            None
        }
    }
}

/// The headline check. Because both devices draw from bit-identical random
/// streams, agreement should be at float-reassociation level (~1e-4), three
/// orders of magnitude below the Monte Carlo noise floor (~2.6e-1 at this sample
/// count). Landing anywhere in between means the two are not running the same
/// computation.
#[test]
fn megakernel_matches_the_cpu_reference() {
    let Some(gpu) = gpu() else { return };
    let def = scenes::cornell_box();
    let params = RenderParams {
        width: 160,
        height: 160,
        samples: 96,
        max_depth: 8,
        ..Default::default()
    };

    let cpu = integrator::render(&def, &params);
    let gpu_film = render_native_on(&gpu, &def, &params).expect("gpu render");
    let d = compare(&cpu, &gpu_film).expect("same size");

    eprintln!(
        "mean rel {:.3e}, rmse {:.3e}, energy ratio {:.6}",
        d.mean_rel,
        d.rmse,
        d.mean_b / d.mean_a
    );

    assert!(
        d.mean_rel < pt_cli_threshold(),
        "CPU and GPU disagree by {:.3e} mean relative error (max {:.3e} at {:?}).\n\
         At this level the devices are not running the same computation. Check, in order:\n\
           1. the order random numbers are drawn in (pixel jitter, then lens, then per bounce)\n\
           2. RNG seeding — rng_init(frame_seed, pixel_index, sample_offset + s)\n\
           3. GPU buffer layout — run `cargo test -p pt-core gpu_layout`\n\
           4. the sign-bit handling in onb_sign (negative zero in flipped normals)",
        d.mean_rel,
        d.max_abs,
        d.max_abs_at
    );

    // Energy must match closely too: a systematic gain or loss would show up
    // here even if it were spread thinly enough to keep the relative error low.
    let ratio = d.mean_b / d.mean_a;
    assert!(
        (ratio - 1.0).abs() < 1e-3,
        "energy ratio {ratio:.6} — one device is losing or gaining light"
    );
}

fn pt_cli_threshold() -> f64 {
    // Same constant the CLI reports against; duplicated rather than depending on
    // the binary crate.
    2.0e-3
}

/// The agreement must hold as paths get longer. A bug in the bounce loop — a
/// throughput update in the wrong place, a mismatched termination condition —
/// typically shows up only past the first bounce or two.
#[test]
fn agreement_holds_across_path_lengths() {
    let Some(gpu) = gpu() else { return };
    let def = scenes::cornell_box();
    for depth in [1u32, 2, 4, 8, 16] {
        let params = RenderParams {
            width: 96,
            height: 96,
            samples: 48,
            max_depth: depth,
            ..Default::default()
        };
        let cpu = integrator::render(&def, &params);
        let g = render_native_on(&gpu, &def, &params).expect("gpu render");
        let d = compare(&cpu, &g).expect("same size");
        eprintln!("depth {depth:2}: mean rel {:.3e}", d.mean_rel);
        assert!(
            d.mean_rel < pt_cli_threshold(),
            "diverged at max_depth = {depth}: mean rel {:.3e}",
            d.mean_rel
        );
    }
}

/// Accumulating N samples over several dispatches must give the same answer as
/// taking them in one — that is the whole premise of progressive rendering in
/// the browser, and it is easy to break by mis-advancing `sample_offset`.
#[test]
fn progressive_accumulation_matches_a_single_dispatch() {
    let Some(gpu) = gpu() else { return };
    let def = scenes::cornell_box();
    let base = RenderParams {
        width: 96,
        height: 96,
        max_depth: 6,
        ..Default::default()
    };

    let one_shot = render_native_on(
        &gpu,
        &def,
        &RenderParams {
            samples: 64,
            ..base
        },
    )
    .expect("gpu render");

    // Four dispatches of 16, advancing the sample offset, is what the browser
    // does. The CPU tracer models it by rendering each chunk and averaging.
    let mut acc = vec![glam::Vec3::ZERO; (base.width * base.height) as usize];
    for chunk in 0..4u32 {
        let p = RenderParams {
            samples: 16,
            sample_offset: chunk * 16,
            ..base
        };
        let f = render_native_on(&gpu, &def, &p).expect("gpu render");
        for (a, b) in acc.iter_mut().zip(f.data.iter()) {
            *a += *b * 16.0;
        }
    }
    let progressive = pt_core::integrator::Film {
        width: base.width,
        height: base.height,
        data: acc.iter().map(|v| *v / 64.0).collect(),
    };

    let d = compare(&one_shot, &progressive).expect("same size");
    eprintln!("progressive vs one-shot: mean rel {:.3e}", d.mean_rel);
    assert!(
        d.mean_rel < 1e-5,
        "progressive accumulation drifts from a single dispatch ({:.3e}). \
         Each dispatch must advance sample_offset by exactly samples_per_launch.",
        d.mean_rel
    );
}

/// A scene with no spheres still has to bind a non-empty storage buffer. This is
/// a WebGPU validation rule that is easy to trip over later when scenes become
/// data-driven, so pin it now.
#[test]
fn empty_primitive_arrays_do_not_break_binding() {
    let Some(gpu) = gpu() else { return };
    let mut def = scenes::cornell_box();
    // Remove every analytic primitive: the shader must still bind a non-empty
    // buffer, and `num_primitives == 0` must stop the loop.
    def.scene.blob.primitives.clear();
    let params = RenderParams {
        width: 48,
        height: 48,
        samples: 8,
        max_depth: 3,
        ..Default::default()
    };

    let cpu = integrator::render(&def, &params);
    let g = render_native_on(&gpu, &def, &params).expect("gpu render with no spheres");
    let d = compare(&cpu, &g).expect("same size");
    assert!(
        d.mean_rel < pt_cli_threshold(),
        "mean rel {:.3e}",
        d.mean_rel
    );
}

/// Build step 5: the same agreement, now with triangle meshes and BVH
/// traversal in the shader.
///
/// Judged against the Monte Carlo noise floor rather than an absolute
/// threshold. Mesh scenes legitimately sit an order of magnitude higher in
/// absolute terms than analytic ones — every facet boundary is a silhouette
/// where one ULP decides hit or miss, and the two devices then follow entirely
/// different paths for that sample. The ratio to the noise floor is the
/// scene-independent question.
#[test]
fn mesh_scenes_match_the_cpu_reference() {
    let Some(gpu) = gpu() else { return };
    for name in ["cornell-mesh", "bvh-stress"] {
        let def = scenes::by_name(name).expect("scene exists");
        let params = RenderParams {
            width: 128,
            height: 128,
            samples: 48,
            max_depth: 6,
            ..Default::default()
        };

        let cpu = integrator::render(&def, &params);
        let g = render_native_on(&gpu, &def, &params).expect("gpu render");
        let d = compare(&cpu, &g).expect("same size");

        // Two runs of the CPU tracer differing only in seed: the scale any
        // difference has to be judged against.
        let other = integrator::render(
            &def,
            &RenderParams {
                frame_seed: params.frame_seed ^ 0x00AB_CDEF,
                ..params
            },
        );
        let nf = compare(&cpu, &other).expect("same size");
        let ratio = d.mean_rel / nf.mean_rel.max(1e-18);

        eprintln!(
            "{name:<13} {:>7} tris  mean rel {:.3e}  noise floor {:.3e}  ratio {ratio:.4}",
            def.scene.blob.triangles.len(),
            d.mean_rel,
            nf.mean_rel
        );
        assert!(
            ratio < 0.05,
            "{name}: CPU and GPU differ by {ratio:.4}x the Monte Carlo noise floor \
             (mean rel {:.3e}, max {:.3e} at {:?}).\n\
             At this scale it is a real difference. Check the BVH upload first: node count, \
             and whether the triangle array was compacted into traversal order.",
            d.mean_rel,
            d.max_abs,
            d.max_abs_at
        );
        assert!(
            (d.mean_b / d.mean_a - 1.0).abs() < 5e-3,
            "{name}: energy ratio {:.6}",
            d.mean_b / d.mean_a
        );
    }
}

/// The shader's brute-force fallback (`num_bvh_nodes == 0`) must produce the
/// same image as its BVH path.
///
/// This is the on-GPU version of the brute-force equivalence test, and it is
/// the one that matters most: it checks the *shader's* traversal, including the
/// fixed-size stack and the near-child ordering, rather than the CPU's.
#[test]
fn gpu_bvh_traversal_matches_gpu_brute_force() {
    let Some(gpu) = gpu() else { return };
    let with_bvh = scenes::by_name("cornell-mesh").expect("scene exists");

    let mut without = scenes::by_name("cornell-mesh").expect("scene exists");
    // Dropping the nodes makes the shader fall back to testing every triangle.
    without.scene.blob.bvh_nodes.clear();
    without.scene.bvh = Default::default();

    let params = RenderParams {
        width: 96,
        height: 96,
        samples: 16,
        max_depth: 5,
        ..Default::default()
    };
    let a = render_native_on(&gpu, &with_bvh, &params).expect("gpu render");
    let b = render_native_on(&gpu, &without, &params).expect("gpu render");
    let d = compare(&a, &b).expect("same size");

    eprintln!(
        "gpu bvh vs gpu brute force: mean rel {:.3e}, max abs {:.3e}",
        d.mean_rel, d.max_abs
    );
    // Identical code path through Möller–Trumbore, identical random streams:
    // these should agree bit for bit, not merely closely.
    assert!(
        d.mean_rel < 1e-6,
        "the shader's BVH traversal disagrees with its own brute-force path by {:.3e}. \
         The BVH is missing geometry — check the traversal stack depth and the \
         near/far child ordering.",
        d.mean_rel
    );
}

/// Build step 6: the GGX BSDF, the conductor Fresnel path, and the energy
/// compensation table must all reproduce the CPU reference.
#[test]
fn material_scenes_match_the_cpu_reference() {
    let Some(gpu) = gpu() else { return };
    for name in ["metal-sweep", "furnace-test"] {
        let def = scenes::by_name(name).expect("scene exists");
        let params = RenderParams {
            width: 128,
            height: 96,
            samples: 64,
            max_depth: 10,
            ..Default::default()
        };
        let cpu = integrator::render(&def, &params);
        let g = render_native_on(&gpu, &def, &params).expect("gpu render");
        let d = compare(&cpu, &g).expect("same size");

        let other = integrator::render(
            &def,
            &RenderParams {
                frame_seed: params.frame_seed ^ 0x00AB_CDEF,
                ..params
            },
        );
        let nf = compare(&cpu, &other).expect("same size");
        let ratio = d.mean_rel / nf.mean_rel.max(1e-18);
        eprintln!(
            "{name:<13} mean rel {:.3e}  ratio to noise floor {ratio:.4}",
            d.mean_rel
        );
        assert!(
            ratio < 0.05,
            "{name}: CPU and GPU differ by {ratio:.4}x the noise floor. \
             Check the energy table first — it is generated into WGSL, so a stale \
             shaders/common/ggx_energy.wgsl would show up exactly here.",
        );
    }
}

/// **The white furnace test, on the GPU.**
///
/// The spheres must be invisible in the shader too. This is the check that the
/// WGSL energy compensation, the baked albedo table and its lookup all work
/// together — a stale or mis-indexed table shows up here as visible spheres and
/// essentially nowhere else.
#[test]
fn gpu_furnace_test_renders_the_spheres_invisible() {
    let Some(gpu) = gpu() else { return };
    let def = scenes::by_name("furnace-test").expect("scene exists");
    let params = RenderParams {
        width: 160,
        height: 80,
        samples: 512,
        max_depth: 24,
        ..Default::default()
    };
    let film = render_native_on(&gpu, &def, &params).expect("gpu render");

    let mut sum = 0.0f64;
    let mut worst = 0.0f32;
    for p in &film.data {
        assert!(p.is_finite(), "non-finite pixel {p}");
        sum += p.x as f64;
        worst = worst.max((p.x - 1.0).abs());
    }
    let mean = sum / film.data.len() as f64;
    eprintln!("gpu furnace: mean {mean:.5}, worst pixel {worst:.4}");
    assert!(
        (mean - 1.0).abs() < 3e-3,
        "GPU furnace mean radiance {mean:.5}, expected 1.0 — the shader's BSDF is not \
         energy conserving"
    );
}

/// Build steps 8 and 9: the shader's light sampling must match the CPU's, in
/// every mode.
///
/// These have more moving parts than anything before them — light selection, the
/// area-to-solid-angle conversion, a second ray type, a rule about when emission
/// may be counted, and the MIS weights on both sides — and each can be got wrong
/// independently on one device.
#[test]
fn sampling_modes_match_the_cpu_reference() {
    let Some(gpu) = gpu() else { return };
    for name in ["cornell-box", "cornell-mesh", "mis-scene"] {
        let def = scenes::by_name(name).expect("scene exists");
        for mode in pt_core::integrator::SamplingMode::ALL {
            let params = RenderParams {
                width: 112,
                height: 112,
                samples: 48,
                max_depth: 8,
                sampling: mode,
                ..Default::default()
            };
            let cpu = integrator::render(&def, &params);
            let g = render_native_on(&gpu, &def, &params).expect("gpu render");
            let d = compare(&cpu, &g).expect("same size");

            let other = integrator::render(
                &def,
                &RenderParams {
                    frame_seed: params.frame_seed ^ 0x00AB_CDEF,
                    ..params
                },
            );
            let nf = compare(&cpu, &other).expect("same size");
            let ratio = d.mean_rel / nf.mean_rel.max(1e-18);

            eprintln!(
                "{name:<13} {:<5} mean rel {:.3e}, {ratio:.4}x noise floor",
                mode.name(),
                d.mean_rel
            );
            assert!(
                ratio < 0.05,
                "{name} in {} mode: CPU and GPU differ by {ratio:.4}x the noise floor. \
                 For NEE, check the light buffer upload and the order random numbers are \
                 drawn — the shader must draw the light sample before the BSDF sample, and \
                 must draw nothing at all when the scene has no lights.",
                mode.name()
            );
        }
    }
}

/// On the GPU too, all three strategies must converge to the same image.
///
/// The CPU test of this lives in `pt-core`; repeating it here is not redundant,
/// because the shader implements each mode separately and could easily have a
/// double-counted emission, a dropped measure conversion, or a mis-weighted MIS
/// term on one path only.
#[test]
fn gpu_sampling_modes_converge_to_the_same_image() {
    let Some(gpu) = gpu() else { return };
    let def = scenes::by_name("cornell-box").expect("scene exists");
    let base = RenderParams {
        width: 48,
        height: 48,
        max_depth: 6,
        ..Default::default()
    };

    let mut rmse = Vec::new();
    let mut ratio = 0.0;
    for &spp in &[1024u32, 4096, 16384] {
        let a = render_native_on(
            &gpu,
            &def,
            &RenderParams {
                samples: spp,
                sampling: pt_core::integrator::SamplingMode::BsdfOnly,
                ..base
            },
        )
        .expect("gpu render");
        let b = render_native_on(
            &gpu,
            &def,
            &RenderParams {
                samples: spp,
                sampling: pt_core::integrator::SamplingMode::NeeOnly,
                ..base
            },
        )
        .expect("gpu render");
        let c = render_native_on(
            &gpu,
            &def,
            &RenderParams {
                samples: spp,
                sampling: pt_core::integrator::SamplingMode::Mis,
                ..base
            },
        )
        .expect("gpu render");
        let d = compare(&a, &b).expect("same size");
        let dm = compare(&a, &c).expect("same size");
        ratio = d.mean_b / d.mean_a;
        eprintln!(
            "{spp:>6} spp: bsdf {:.6}, nee {:.6}, mis {:.6}, rmse {:.5}",
            d.mean_a, d.mean_b, dm.mean_b, d.rmse
        );
        assert!(
            (dm.mean_b / dm.mean_a - 1.0).abs() < 2e-2,
            "on the GPU, MIS and BSDF-only disagree: {:.6} vs {:.6}",
            dm.mean_b,
            dm.mean_a
        );
        rmse.push(d.rmse);
    }

    assert!(
        (ratio - 1.0).abs() < 5e-3,
        "on the GPU, BSDF-only and NEE disagree about total energy by {:.3}%",
        (ratio - 1.0).abs() * 100.0
    );
    // A plateau in the difference would mean they converge to *different*
    // images; continued 1/sqrt(N) decay means the difference is only noise.
    for w in rmse.windows(2) {
        let decay = w[1] / w[0];
        assert!(
            (0.40..0.68).contains(&decay),
            "the difference fell by only {decay:.3} when samples quadrupled \
             (expected about 0.5): {rmse:?}"
        );
    }
}

/// Build step 12: dielectrics, on both devices.
///
/// Glass is the hardest thing in the renderer to keep in step between two
/// implementations, because a transmitted path touches every part of the
/// contract at once: the hemisphere the frame is flipped into, which side the
/// ray origin is offset toward, the relative index for entering versus leaving,
/// and the order random numbers are drawn in when a third lobe exists. A
/// mismatch in any of them produces an image that still looks like glass.
#[test]
fn dielectrics_match_the_cpu_reference() {
    let Some(gpu) = gpu() else { return };
    let def = scenes::glass_box();
    let params = RenderParams {
        width: 128,
        height: 128,
        samples: 64,
        // Deep: a ray entering the diamond sphere bounces internally many times
        // before escaping, and a shallow limit would hide a difference in
        // exactly the paths that exercise total internal reflection.
        max_depth: 24,
        ..Default::default()
    };

    let cpu = integrator::render(&def, &params);
    let gpu_film = render_native_on(&gpu, &def, &params).expect("gpu render");
    let d = compare(&cpu, &gpu_film).expect("same size");
    eprintln!(
        "glass-box: mean rel {:.3e}, energy ratio {:.6}",
        d.mean_rel,
        d.mean_b / d.mean_a
    );
    assert!(
        d.mean_rel < pt_cli_threshold(),
        "CPU and GPU disagree on dielectrics by {:.3e} (max {:.3e} at {:?}).\n\
         Check, in order:\n\
           1. `relative_ior` — entering is the IOR, leaving is its reciprocal\n\
           2. which side the ray origin is offset toward after a transmitted sample\n\
           3. the |cos| in the sample weight, which is negative for wi.z < 0 if missed\n\
           4. the half-vector normalize(wo + eta * wi), flipped into z > 0",
        d.mean_rel,
        d.max_abs,
        d.max_abs_at
    );
    let ratio = d.mean_b / d.mean_a;
    assert!(
        (ratio - 1.0).abs() < 1e-3,
        "energy ratio {ratio:.6} — one device is losing or gaining light through glass"
    );
}

/// Build step 13: environment lighting, on both devices.
///
/// The environment sampler is the first thing in the renderer that reads from a
/// **texture** rather than a storage buffer, and the first that does a binary
/// search on the GPU. Either could differ from the CPU in ways an image barely
/// shows: a search that resolves ties differently picks a neighbouring texel,
/// which matters enormously at the edge of a sun four orders of magnitude
/// brighter than the sky next to it.
#[test]
fn environment_lighting_matches_the_cpu_reference() {
    let Some(gpu) = gpu() else { return };
    let def = scenes::sunset();
    let params = RenderParams {
        width: 128,
        height: 128,
        samples: 64,
        max_depth: 6,
        ..Default::default()
    };

    let cpu = integrator::render(&def, &params);
    let gpu_film = render_native_on(&gpu, &def, &params).expect("gpu render");
    let d = compare(&cpu, &gpu_film).expect("same size");
    eprintln!(
        "sunset: mean rel {:.3e}, energy ratio {:.6}",
        d.mean_rel,
        d.mean_b / d.mean_a
    );
    assert!(
        d.mean_rel < pt_cli_threshold(),
        "CPU and GPU disagree on environment lighting by {:.3e} (max {:.3e} at \
         {:?}).\nCheck, in order:\n\
           1. the CDF texture packing — row `height` holds the marginal\n\
           2. the binary search's tie handling, which must match `sample_cdf`\n\
           3. `env_total_weight` reaching the uniforms\n\
           4. the 1/strategy-count factor on both MIS sides",
        d.mean_rel,
        d.max_abs,
        d.max_abs_at
    );
    let ratio = d.mean_b / d.mean_a;
    assert!(
        (ratio - 1.0).abs() < 2e-3,
        "energy ratio {ratio:.6} — one device is gathering more sky than the other"
    );
}

/// Build step 15: instancing, on both devices.
///
/// A two-level hierarchy is where a CPU and a GPU implementation are most likely
/// to drift, because almost everything about it is a convention that has to be
/// chosen identically twice: which way the stored matrix points, whether the
/// transformed direction is normalised, which side the normal ends up on, and
/// how the shared node array is rebased. Every one of those produces an image
/// that still looks like the scene.
#[test]
fn instancing_matches_the_cpu_reference() {
    let Some(gpu) = gpu() else { return };
    let def = scenes::instance_forest();
    let params = RenderParams {
        width: 128,
        height: 128,
        samples: 48,
        max_depth: 5,
        ..Default::default()
    };

    let cpu = integrator::render(&def, &params);
    let gpu_film = render_native_on(&gpu, &def, &params).expect("gpu render");
    let d = compare(&cpu, &gpu_film).expect("same size");
    eprintln!(
        "instance-forest: mean rel {:.3e}, energy ratio {:.6}",
        d.mean_rel,
        d.mean_b / d.mean_a
    );
    assert!(
        d.mean_rel < pt_cli_threshold(),
        "CPU and GPU disagree on instancing by {:.3e} (max {:.3e} at {:?}).\n\
         Check, in order:\n\
           1. the stored matrix is world-to-object on both sides\n\
           2. the object-space direction is not normalised, or t rescales\n\
           3. normals come back by the inverse-transpose\n\
           4. leaf and child indices are rebased into the shared arrays\n\
           5. the material override sentinel is u32::MAX on both sides",
        d.mean_rel,
        d.max_abs,
        d.max_abs_at
    );
    let ratio = d.mean_b / d.mean_a;
    assert!(
        (ratio - 1.0).abs() < 2e-3,
        "energy ratio {ratio:.6} — one device is missing or duplicating instances"
    );
}
