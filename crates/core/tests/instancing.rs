//! Build step 15: instancing and the two-level hierarchy.
//!
//! The claim instancing makes is narrow and total: **the image must not change**.
//! A hundred instances of a mesh must render exactly as a hundred copies of that
//! mesh with the transforms baked into their vertices — same pixels, using a
//! fraction of the memory. Everything else here is in service of checking that.

use glam::{Mat4, Vec3};
use pt_core::camera::Camera;
use pt_core::gpu_layout::{GpuMaterial, SceneBlob};
use pt_core::image::compare;
use pt_core::integrator::{self, RenderParams};
use pt_core::mesh;
use pt_core::scene::Scene;
use pt_core::scenes::SceneDef;

/// The placements both scenes use. Deliberately varied: a pure translation
/// exercises nothing, while rotation and non-uniform scale are what separate a
/// correct normal transform from the two classic wrong ones.
fn placements() -> Vec<Mat4> {
    vec![
        Mat4::from_translation(Vec3::new(-1.6, 0.0, 0.0)),
        Mat4::from_translation(Vec3::new(0.0, 0.0, 0.0)) * Mat4::from_rotation_y(0.9),
        Mat4::from_translation(Vec3::new(1.7, 0.1, 0.3))
            * Mat4::from_rotation_z(0.5)
            * Mat4::from_scale(Vec3::new(1.4, 0.6, 1.0)),
        Mat4::from_translation(Vec3::new(0.2, 1.3, -1.2)) * Mat4::from_scale(Vec3::splat(0.55)),
    ]
}

fn camera() -> Camera {
    Camera::look_at(Vec3::new(0.5, 1.2, -6.0), Vec3::new(0.0, 0.3, 0.0), 45.0)
}

fn params() -> RenderParams {
    RenderParams {
        width: 96,
        height: 96,
        samples: 32,
        max_depth: 4,
        ..Default::default()
    }
}

/// Every placement baked into its own copy of the vertices: the reference.
fn baked() -> SceneDef {
    let mut blob = SceneBlob {
        materials: vec![GpuMaterial::diffuse(Vec3::new(0.75, 0.6, 0.45))],
        ..Default::default()
    };
    for m in placements() {
        mesh::icosphere(Vec3::ZERO, 0.7, 2)
            .transformed(m)
            .append_to(&mut blob, 0);
    }
    let mut def = SceneDef {
        name: "baked",
        description: "",
        scene: Scene {
            blob,
            ..Default::default()
        },
        camera: camera(),
        background: Vec3::splat(0.55),
    };
    def.scene.finalize();
    def
}

/// One mesh, four placements.
fn instanced() -> SceneDef {
    let mut blob = SceneBlob {
        materials: vec![GpuMaterial::diffuse(Vec3::new(0.75, 0.6, 0.45))],
        ..Default::default()
    };
    mesh::icosphere(Vec3::ZERO, 0.7, 2).append_to(&mut blob, 0);
    let tri_count = blob.triangles.len() as u32;

    let mut def = SceneDef {
        name: "instanced",
        description: "",
        scene: Scene {
            blob,
            ..Default::default()
        },
        camera: camera(),
        background: Vec3::splat(0.55),
    };
    let places: Vec<(Mat4, u32, u32)> = placements()
        .into_iter()
        .map(|m| (m, 0u32, u32::MAX))
        .collect();
    def.scene.build_instances(&[(0, tri_count)], &places);
    def.scene.blob.lights = pt_core::light::build_lights(&def.scene.blob);
    def
}

/// The headline claim.
///
/// Not "close" — the two describe the same geometry, and the only reason they
/// are not bit-identical is that the transform is applied to the ray rather than
/// to the vertices, so the arithmetic happens in a different order and lands on
/// different floats. A systematic difference means a wrong transform somewhere.
#[test]
fn instancing_does_not_change_the_image() {
    let a = integrator::render(&baked(), &params());
    let b = integrator::render(&instanced(), &params());
    let d = compare(&a, &b).expect("same size");
    eprintln!(
        "baked vs instanced: mean rel {:.3e}, max abs {:.3e}, energy ratio {:.6}",
        d.mean_rel,
        d.max_abs,
        d.mean_b / d.mean_a
    );
    assert!(
        d.mean_rel < 5.0e-3,
        "instanced geometry renders differently from the same geometry baked \
         out: mean relative difference {:.3e}.\nCheck, in order:\n\
           1. the object-space ray direction is *not* normalised, or t rescales\n\
           2. normals come back by the inverse-transpose, not the matrix\n\
           3. world bounds transform all eight corners, not two\n\
           4. front/back facing is decided in world space, after the transform",
        d.mean_rel
    );
    let ratio = d.mean_b / d.mean_a;
    assert!(
        (ratio - 1.0).abs() < 2.0e-3,
        "energy ratio {ratio:.6} — the instanced scene is systematically \
         brighter or darker, which means geometry is being missed or duplicated"
    );
}

/// And it must actually save the memory it exists to save.
#[test]
fn instancing_stores_the_mesh_once() {
    let baked = baked();
    let inst = instanced();
    let (bt, it) = (
        baked.scene.blob.triangles.len(),
        inst.scene.blob.triangles.len(),
    );
    let (bn, in_) = (
        baked.scene.blob.bvh_nodes.len(),
        inst.scene.blob.bvh_nodes.len(),
    );
    eprintln!(
        "baked: {bt} triangles, {bn} nodes.  instanced: {it} triangles, {in_} nodes \
         ({} instances)",
        inst.scene.instances.instances.len()
    );
    let n = placements().len();
    assert_eq!(
        it * n,
        bt,
        "the instanced scene should store the mesh once ({} triangles), not {n} times",
        bt / n
    );
    assert!(
        in_ < bn,
        "the instanced scene's node count ({in_}) should be below the baked one's \
         ({bn}); a BLAS plus a small TLAS is less than a BVH over everything"
    );
}

/// A material override must recolour an instance without duplicating its
/// geometry.
///
/// The other half of what instancing is for: a hundred copies of one mesh in a
/// hundred colours should still store one mesh.
#[test]
fn material_override_applies_per_instance() {
    let mut blob = SceneBlob {
        materials: vec![
            GpuMaterial::diffuse(Vec3::new(0.8, 0.1, 0.1)),
            GpuMaterial::diffuse(Vec3::new(0.1, 0.1, 0.8)),
        ],
        ..Default::default()
    };
    mesh::icosphere(Vec3::ZERO, 0.7, 2).append_to(&mut blob, 0);
    let tri_count = blob.triangles.len() as u32;

    let mut def = SceneDef {
        name: "override",
        description: "",
        scene: Scene {
            blob,
            ..Default::default()
        },
        camera: camera(),
        background: Vec3::splat(0.55),
    };
    def.scene.build_instances(
        &[(0, tri_count)],
        &[
            // Left keeps the mesh's own red; right is overridden to blue.
            (Mat4::from_translation(Vec3::new(-1.0, 0.0, 0.0)), 0, u32::MAX),
            (Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0)), 0, 1),
        ],
    );
    def.scene.blob.lights = pt_core::light::build_lights(&def.scene.blob);

    let film = integrator::render(&def, &params());
    // Sample the middle of each sphere.
    //
    // The camera looks down +z with +y up, so **larger world x appears on the
    // left of the image** — the same convention the Cornell box's wall comments
    // record. The instance at x = +1 is therefore the left half of the frame.
    // Asserting the intuitive mapping instead is how this test first failed,
    // reporting a material swap that was not happening.
    let at = |x: u32, y: u32| film.data[(y * film.width + x) as usize];
    let overridden = at(film.width / 4, film.height / 2);
    let inherited = at(3 * film.width / 4, film.height / 2);
    eprintln!("x=+1 (screen left) {overridden:?}, x=-1 (screen right) {inherited:?}");
    assert!(
        inherited.x > inherited.z * 1.5,
        "the instance at x = -1 should keep the mesh's own red material, got \
         {inherited:?}"
    );
    assert!(
        overridden.z > overridden.x * 1.5,
        "the instance at x = +1 should take the blue override, got {overridden:?}"
    );
}
