//! Build step 17: the diagnostic modes, on real renders.
//!
//! A diagnostic that is wrong is worse than no diagnostic, because it is trusted
//! when something else is being debugged. So each mode is checked against
//! something independent of itself.

use glam::{Mat4, Vec3};
use pt_core::diagnostic::{shade, RenderMode};
use pt_core::integrator::{self, RenderParams};
use pt_core::scene::BvhBuilder;
use pt_core::scenes;

fn params() -> RenderParams {
    RenderParams {
        width: 96,
        height: 96,
        samples: 4,
        max_depth: 3,
        ..Default::default()
    }
}

/// The traversal heatmap must actually measure traversal.
///
/// Checked against a *worse tree*: the linear BVH is measurably more expensive
/// to traverse than the binned-SAH one — 1.26x by the SAH cost model, measured
/// at build step 11 — so rendering the same scene with each must show the
/// difference. Nothing else here would catch a counter that was, say, counting
/// leaf visits only, or resetting between bounces.
#[test]
fn the_heatmap_tracks_tree_quality() {
    let mean_steps = |builder: BvhBuilder| {
        let mut def = scenes::cornell_mesh();
        def.scene.build_bvh_with(builder);
        def.scene.blob.lights = pt_core::light::build_lights(&def.scene.blob);
        let (_, g) = integrator::render_with_guides(&def, &params());
        g.steps.iter().sum::<f32>() as f64 / g.steps.len() as f64
    };
    let sah = mean_steps(BvhBuilder::BinnedSah);
    let lbvh = mean_steps(BvhBuilder::Linear);
    eprintln!("mean node visits per ray: sah {sah:.1}, lbvh {lbvh:.1} ({:.2}x)", lbvh / sah);

    assert!(
        sah > 1.0,
        "the heatmap reports {sah:.2} node visits per ray on a scene with a BVH; \
         the counter is not being incremented"
    );
    assert!(
        lbvh > sah * 1.05,
        "the linear BVH ({lbvh:.1} visits) should be measurably more expensive to \
         traverse than the binned-SAH one ({sah:.1}); if they agree, the counter \
         is not measuring traversal"
    );
}

/// Both levels of an instanced hierarchy must be counted.
///
/// A heatmap that showed only the BLAS would hide a badly built TLAS entirely,
/// which is precisely the thing the top level exists to get right.
#[test]
fn the_heatmap_counts_both_levels() {
    let def = scenes::instance_forest();
    let (_, g) = integrator::render_with_guides(&def, &params());
    let hit_steps: Vec<f32> = g
        .steps
        .iter()
        .zip(g.depth.iter())
        .filter(|(_, d)| **d > 0.0)
        .map(|(s, _)| *s)
        .collect();
    assert!(!hit_steps.is_empty(), "no pixels hit the instanced geometry");
    let mean = hit_steps.iter().sum::<f32>() / hit_steps.len() as f32;
    eprintln!("instanced mean node visits per hit ray: {mean:.1}");
    // A TLAS over 49 instances plus a BLAS is several nodes deep on both levels,
    // so a count in the low single digits would mean one level is missing.
    assert!(
        mean > 8.0,
        "only {mean:.1} node visits per ray through a two-level hierarchy; one of \
         the levels is probably not being counted"
    );
}

/// Every mode must produce finite, displayable output on a real scene.
#[test]
fn every_mode_produces_displayable_output() {
    let def = scenes::cornell_mesh();
    let (film, g) = integrator::render_with_guides(&def, &params());
    let depth_scale = g.depth.iter().copied().fold(0.0f32, f32::max).max(1e-6);

    for mode in RenderMode::all() {
        let mut worst = Vec3::ZERO;
        for i in 0..film.data.len() {
            let c = shade(
                mode,
                film.data[i],
                g.albedo[i],
                g.normal[i],
                g.depth[i],
                g.steps[i],
                depth_scale,
            );
            assert!(
                c.is_finite(),
                "{} produced a non-finite pixel at {i}: {c:?}",
                mode.name()
            );
            assert!(
                c.min_element() >= -1e-6,
                "{} produced a negative pixel at {i}: {c:?}",
                mode.name()
            );
            worst = worst.max(c);
        }
        // Diagnostics bypass tone mapping, so they must already be in display
        // range. Beauty is radiance and legitimately is not.
        if mode.is_data() {
            assert!(
                worst.max_element() <= 1.001,
                "{} produced {worst:?}, above 1 — a diagnostic bypasses tone \
                 mapping, so it has to be display-ready",
                mode.name()
            );
        }
        eprintln!("{:<8} max {worst:?}", mode.name());
    }
}

/// The normal diagnostic must follow the geometry, not merely produce colour.
///
/// Checked by *rotating an instance* and requiring the displayed normals to
/// change correspondingly. A mode that returned a constant, or that read the
/// wrong channel, would pass every check above.
#[test]
fn normals_follow_the_geometry() {
    let sample_normals = |angle: f32| {
        let mut blob = pt_core::gpu_layout::SceneBlob {
            materials: vec![pt_core::gpu_layout::GpuMaterial::diffuse(Vec3::splat(0.7))],
            ..Default::default()
        };
        pt_core::mesh::cuboid(Vec3::splat(-1.0), Vec3::splat(1.0)).append_to(&mut blob, 0);
        let tris = blob.triangles.len() as u32;
        let mut def = scenes::SceneDef {
            name: "rot",
            description: "",
            scene: pt_core::scene::Scene {
                blob,
                ..Default::default()
            },
            camera: pt_core::camera::Camera::look_at(
                Vec3::new(0.0, 0.0, -5.0),
                Vec3::ZERO,
                45.0,
            ),
            background: Vec3::ONE,
        };
        def.scene
            .build_instances(&[(0, tris)], &[(Mat4::from_rotation_y(angle), 0, u32::MAX)]);
        def.scene.blob.lights = pt_core::light::build_lights(&def.scene.blob);
        let (_, g) = integrator::render_with_guides(&def, &params());
        g
    };

    let a = sample_normals(0.0);
    let b = sample_normals(0.6);
    // The face the camera sees head-on should rotate away, so the centre
    // pixel's normal must change.
    let mid = (96 * 48 + 48) as usize;
    let (na, nb) = (a.normal[mid], b.normal[mid]);
    eprintln!("centre normal: {na:?} -> {nb:?}");
    assert!(
        na.length() > 0.5 && nb.length() > 0.5,
        "the centre pixel should hit the cube in both renders"
    );
    assert!(
        (na - nb).length() > 0.2,
        "rotating the instance by 0.6 rad left the displayed normal unchanged \
         ({na:?} vs {nb:?}); the mode is not reading the geometry"
    );
}
