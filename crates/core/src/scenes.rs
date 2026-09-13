//! Built-in scenes. Rust is the source of truth; `codegen` packs these into the
//! TypeScript bundle so the browser and the native renderers cannot disagree
//! about what they are drawing.

use crate::camera::Camera;
use crate::gpu_layout::{GpuMaterial, GpuPrimitive, SceneBlob};
use crate::mesh;
use crate::scene::Scene;
use glam::{Mat4, Vec3};

pub struct SceneDef {
    pub name: &'static str,
    pub description: &'static str,
    pub scene: Scene,
    pub camera: Camera,
    /// Constant environment radiance. Zero for closed scenes.
    pub background: Vec3,
}

/// Build a parallelogram from a corner and two edge vectors.
///
/// The normal is `normalize(cross(edge_u, edge_v))`, so **edge order determines
/// which way the quad faces**. Emission is one-sided and back-face culling of
/// light is a real behaviour here, so the order matters; the tests below assert
/// that every wall of the Cornell box faces inward.
fn quad(origin: Vec3, edge_u: Vec3, edge_v: Vec3, material: u32) -> GpuPrimitive {
    GpuPrimitive::quad(origin, edge_u, edge_v, material)
}

/// Material indices for [`cornell_box`], named so the scene reads clearly.
mod cornell_mat {
    pub const WHITE: u32 = 0;
    pub const RED: u32 = 1;
    pub const GREEN: u32 = 2;
    pub const LIGHT: u32 = 3;
    pub const SPHERE_WHITE: u32 = 4;
    pub const SPHERE_BLUE: u32 = 5;
}

/// The Cornell box: the reference scene, because it has a known correct answer.
///
/// Geometry follows the canonical Cornell dataset simplified to an exact
/// 555 x 555 x 555 cube. The camera sits at z = -800 looking down +z, which puts
/// the x = 555 (red) wall on the **left** of the image and x = 0 (green) on the
/// right — the orientation everyone's reference images use.
///
/// The two classic boxes are replaced by two spheres for now: build step 2 is
/// explicitly "sphere intersection, Lambertian only", and analytic spheres are
/// exact, so any disagreement between CPU and GPU is a light-transport bug
/// rather than a tessellation difference. Boxes return with triangle meshes at
/// build step 5.
///
/// The blue sphere is not decoration: colour bleeding from it onto the white
/// floor is the cheapest visual confirmation that indirect transport is actually
/// happening and is not being clamped away.
pub fn cornell_box() -> SceneDef {
    use cornell_mat::*;
    const S: f32 = 555.0;

    let materials = vec![
        GpuMaterial::diffuse(Vec3::new(0.725, 0.710, 0.680)), // WHITE
        GpuMaterial::diffuse(Vec3::new(0.630, 0.065, 0.050)), // RED
        GpuMaterial::diffuse(Vec3::new(0.140, 0.450, 0.091)), // GREEN
        // Warm white emitter. The value is radiance, in W/(m^2 sr); it is >> 1
        // because the emitter is small and we want the box to be reasonably
        // exposed. Nothing downstream may clamp this.
        GpuMaterial::emissive(Vec3::new(18.4, 15.6, 8.0)), // LIGHT
        GpuMaterial::diffuse(Vec3::splat(0.800)),          // SPHERE_WHITE
        GpuMaterial::diffuse(Vec3::new(0.200, 0.380, 0.800)), // SPHERE_BLUE
    ];

    // Walls. Edge order is chosen so each normal points into the box interior.
    let mut primitives = vec![
        // Floor, y = 0, normal +y.
        quad(
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, S),
            Vec3::new(S, 0.0, 0.0),
            WHITE,
        ),
        // Ceiling, y = S, normal -y.
        quad(
            Vec3::new(0.0, S, 0.0),
            Vec3::new(S, 0.0, 0.0),
            Vec3::new(0.0, 0.0, S),
            WHITE,
        ),
        // Back wall, z = S, normal -z.
        quad(
            Vec3::new(0.0, 0.0, S),
            Vec3::new(0.0, S, 0.0),
            Vec3::new(S, 0.0, 0.0),
            WHITE,
        ),
        // Left wall (red), x = S, normal -x.
        quad(
            Vec3::new(S, 0.0, 0.0),
            Vec3::new(0.0, 0.0, S),
            Vec3::new(0.0, S, 0.0),
            RED,
        ),
        // Right wall (green), x = 0, normal +x.
        quad(
            Vec3::ZERO,
            Vec3::new(0.0, S, 0.0),
            Vec3::new(0.0, 0.0, S),
            GREEN,
        ),
        // Ceiling light, normal -y (pointing down into the room). Dropped 0.1
        // below the ceiling so the two coplanar surfaces cannot z-fight.
        quad(
            Vec3::new(213.0, S - 0.1, 227.0),
            Vec3::new(130.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 105.0),
            LIGHT,
        ),
    ];

    primitives.push(GpuPrimitive::sphere(
        Vec3::new(185.0, 120.0, 175.0),
        120.0,
        SPHERE_WHITE,
    ));
    primitives.push(GpuPrimitive::sphere(
        Vec3::new(370.0, 90.0, 350.0),
        90.0,
        SPHERE_BLUE,
    ));

    let mut def = SceneDef {
        name: "cornell-box",
        description: "The reference scene. Diffuse-only, one ceiling area light, five walls \
                      with the front face open toward the camera. Colour bleed from the red and \
                      green walls (and the blue sphere) onto the neutral floor is the visual \
                      signature of correct indirect transport.",
        scene: Scene {
            blob: SceneBlob {
                materials,
                primitives,
                ..Default::default()
            },
            ..Default::default()
        },
        // The camera sits outside the box, looking in through the open front
        // face at z = 0. The FOV is chosen so that the image plane at z = 0
        // measures 267.7 units from the centre — comfortably inside the 277.5
        // half-width of the opening — so no primary ray escapes past the rim and
        // the frame is completely filled by the box. A 40-degree FOV, which is
        // what the original Cornell setup uses, overshoots the opening here and
        // leaves a black border.
        camera: Camera::look_at(
            Vec3::new(278.0, 278.0, -800.0),
            Vec3::new(278.0, 278.0, 0.0),
            37.0,
        ),
        background: Vec3::ZERO,
    };
    def.scene.finalize();
    def
}

/// The Cornell box with every surface as triangles.
///
/// The walls are the *same* parallelograms as [`cornell_box`], each split into
/// two triangles — which is geometrically exact, not an approximation. That
/// makes this the sharpest available test of the triangle path and the BVH: the
/// two scenes must converge to the same image, and any difference beyond
/// floating-point noise is a bug in mesh intersection or traversal.
///
/// The spheres become geodesic meshes, which is where the two scenes genuinely
/// differ — a tessellated sphere is not a sphere. Its silhouette is faceted and
/// its surface is a polyhedron, though smooth vertex normals hide that in
/// shading.
pub fn cornell_mesh() -> SceneDef {
    use cornell_mat::*;
    const S: f32 = 555.0;

    let mut def = cornell_box();
    let blob = &mut def.scene.blob;

    // Convert every analytic primitive to triangles, then drop them.
    let prims = std::mem::take(&mut blob.primitives);
    for p in prims.iter().filter(|p| p.is_quad()) {
        mesh::quad(
            Vec3::from_array(p.position),
            Vec3::from_array(p.edge_u),
            Vec3::from_array(p.edge_v),
        )
        .append_to(blob, p.material);
    }
    // Level 4: 5120 triangles each, enough that the silhouette reads as smooth
    // at typical resolutions while keeping the scene small enough to inline.
    for (sphere, material) in prims
        .iter()
        .filter(|p| !p.is_quad())
        .zip([SPHERE_WHITE, SPHERE_BLUE])
    {
        mesh::icosphere(Vec3::from_array(sphere.position), sphere.radius, 4)
            .append_to(blob, material);
    }

    def.name = "cornell-mesh";
    def.description = "The same Cornell box built entirely from triangles, traversed through a \
                       SAH BVH. The walls are geometrically identical to the analytic version, \
                       so the two must converge to the same image — which is how the mesh path \
                       and the BVH get validated.";
    let _ = S;
    def.scene.finalize();
    def
}

/// Many meshes, to exercise the acceleration structure.
///
/// A grid of geodesic spheres inside the Cornell box. The point is not the
/// picture: it is that the triangle count is two orders of magnitude above the
/// other scenes, so BVH quality becomes the thing that determines frame time and
/// the traversal heatmap has something to show.
pub fn bvh_stress() -> SceneDef {
    use cornell_mat::*;
    const S: f32 = 555.0;

    let mut def = cornell_box();
    let blob = &mut def.scene.blob;
    blob.primitives.retain(|p| p.is_quad());

    // 6 x 2 x 6 spheres on a lattice, alternating two materials.
    //
    // Two layers rather than three, and a radius well under half the spacing:
    // a denser lattice walls the ceiling light off entirely, and with BSDF-only
    // sampling (no next event estimation until build step 8) a scene of mutual
    // occluders converges appallingly. The triangle count comes from
    // subdivision instead, which is what the scene is actually for.
    let (nx, ny, nz) = (6, 2, 6);
    let radius = 30.0;
    let mut n = 0u32;
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let f = |i: u32, count: u32| (i as f32 + 1.0) / (count as f32 + 1.0) * S;
                let centre = Vec3::new(f(ix, nx), f(iy, ny) * 0.62 + 95.0, f(iz, nz));
                let material = if (ix + iy + iz) % 2 == 0 {
                    SPHERE_WHITE
                } else {
                    SPHERE_BLUE
                };
                // Level 4 = 5120 triangles each; 72 spheres is ~369k triangles.
                mesh::icosphere(centre, radius, 4).append_to(blob, material);
                n += 1;
            }
        }
    }
    debug_assert_eq!(n, nx * ny * nz);

    def.name = "bvh-stress";
    def.description = "72 geodesic spheres, about 369 thousand triangles, inside the Cornell \
                       box. Built to make acceleration structure quality the limiting factor \
                       rather than shading.";
    def.scene.finalize();
    def
}

/// **The white furnace test, as a scene you can look at.**
///
/// Five spheres of increasing roughness, all with a pure-white non-absorbing
/// BSDF, inside a uniform environment of radiance 1.
///
/// A correct, energy-conserving BSDF renders every one of them **completely
/// invisible** — the image is a flat field of 1.0 with nothing in it. Any sphere
/// you can pick out is energy the BSDF destroyed (darker) or invented
/// (brighter), and where it goes wrong tells you which roughness is at fault.
///
/// Without multiple-scattering compensation the rough spheres appear as
/// obvious dark discs: single-scattering GGX loses 68% of the energy at
/// roughness 1. This scene is the reason that compensation exists.
pub fn furnace_test() -> SceneDef {
    const N: usize = 5;
    let mut materials = Vec::new();
    let mut primitives = Vec::new();

    for i in 0..N {
        let roughness = i as f32 / (N - 1) as f32;
        // metallic = 1 with a white base colour gives F0 = 1: a perfectly
        // reflective, perfectly white surface, which is the purest form of the
        // test. No diffuse lobe, no absorption, nothing to hide behind.
        materials.push(GpuMaterial::metal(Vec3::ONE, roughness));
        let x = (i as f32 - (N - 1) as f32 * 0.5) * 2.4;
        primitives.push(GpuPrimitive::sphere(Vec3::new(x, 0.0, 0.0), 1.0, i as u32));
    }

    let mut def = SceneDef {
        name: "furnace-test",
        description: "Five spheres, roughness 0 to 1, pure white non-absorbing BSDF, in a \
                      uniform environment of radiance 1. A correct energy-conserving BSDF \
                      renders them completely invisible. Any sphere you can see is energy \
                      the BSDF destroyed or invented.",
        scene: Scene {
            blob: SceneBlob {
                materials,
                primitives,
                ..Default::default()
            },
            ..Default::default()
        },
        camera: Camera::look_at(Vec3::new(0.0, 0.8, -12.0), Vec3::ZERO, 45.0),
        // Radiance exactly 1 in every direction: the furnace.
        background: Vec3::ONE,
    };
    def.scene.finalize();
    def
}

/// Roughness and Fresnel showcase: two rows of metal spheres over a floor, lit
/// by a uniform environment.
///
/// Both rows use **measured complex IOR** — copper in front, gold behind — so
/// the Fresnel colour shift with angle is the real one rather than Schlick's
/// monotonic interpolation to white.
///
/// Roughness climbs left to right. What is actually visible is that the rough
/// end reads *paler*: a rough lobe averages Fresnel over a wide range of angles,
/// including grazing ones where every metal goes white. That is correct, and it
/// is distinct from the multiple-scattering compensation, whose job is to stop
/// the rough end going **dark** — single-scattering GGX would lose two thirds of
/// the energy there. The compensation's own contribution is more saturated than
/// the single-bounce term (it was filtered by Fresnel repeatedly), which shows
/// up in the directional albedo rather than as an obvious change in the picture.
pub fn metal_sweep() -> SceneDef {
    const N: usize = 6;
    let mut materials = vec![GpuMaterial::diffuse(Vec3::splat(0.35))]; // floor
    let mut primitives = Vec::new();

    for row in 0..2 {
        let (eta, k) = if row == 0 {
            (
                crate::bsdf::conductors::COPPER_ETA,
                crate::bsdf::conductors::COPPER_K,
            )
        } else {
            (
                crate::bsdf::conductors::GOLD_ETA,
                crate::bsdf::conductors::GOLD_K,
            )
        };
        for i in 0..N {
            // Skewed toward the low end, where roughness changes appearance
            // fastest; a linear ramp spends half its spheres on "rough".
            let t = i as f32 / (N - 1) as f32;
            let roughness = 0.02 + 0.78 * t * t;
            materials.push(GpuMaterial::conductor(eta, k, roughness));
            // Negated so the sweep reads left to right on screen: this camera
            // looks down +z, which puts world -x on the right of the image.
            let x = ((N - 1) as f32 * 0.5 - i as f32) * 2.3;
            primitives.push(GpuPrimitive::sphere(
                Vec3::new(x, 1.0, row as f32 * 2.6),
                1.0,
                materials.len() as u32 - 1,
            ));
        }
    }

    // A large floor so the spheres have something to reflect besides sky.
    primitives.push(quad(
        Vec3::new(-40.0, 0.0, -40.0),
        Vec3::new(0.0, 0.0, 80.0),
        Vec3::new(80.0, 0.0, 0.0),
        0,
    ));

    let mut def = SceneDef {
        name: "metal-sweep",
        description: "Copper (front) and gold (back), roughness increasing left to right, \
                      using measured complex index of refraction rather than Schlick. The rough \
                      end reads paler because a wide lobe averages Fresnel over grazing angles \
                      where every metal goes white; multiple-scattering compensation is what \
                      stops it also going dark.",
        scene: Scene {
            blob: SceneBlob {
                materials,
                primitives,
                ..Default::default()
            },
            ..Default::default()
        },
        camera: Camera::look_at(Vec3::new(0.0, 3.4, -13.0), Vec3::new(0.0, 1.0, 1.3), 40.0),
        background: Vec3::splat(0.75),
    };
    def.scene.finalize();
    def
}

/// **The MIS scene**, after Veach: glossy plates of increasing roughness lit by
/// emitters of increasing size, all of roughly equal power.
///
/// This is the configuration that makes multiple importance sampling
/// *necessary* rather than merely tidy, and it is built to make each strategy
/// fail somewhere:
///
/// * The **small, intensely bright** emitter subtends almost no solid angle, so
///   a BSDF bounce practically never wanders into it. Light sampling finds it
///   every time. BSDF-only renders it as sparse fireflies.
/// * The **large, dim** emitter reflected in a **near-mirror** plate is the
///   opposite. The BSDF lobe is narrow and picks exactly the directions that
///   matter; light sampling scatters points across a broad emitter, almost all
///   of which the narrow lobe then evaluates at nearly zero. Light sampling is
///   the noisy one there.
///
/// Neither strategy is good at both, which is the entire argument for combining
/// them. The power heuristic weights each direction by how densely the strategy
/// that produced it samples, so each takes credit where it is strong.
pub fn mis_scene() -> SceneDef {
    let mut materials = vec![GpuMaterial::diffuse(Vec3::splat(0.25))]; // backdrop
    let mut primitives = Vec::new();

    let eye = Vec3::new(0.0, 3.0, -8.5);
    // The row of emitters all four plates are aimed at.
    let light_y = 5.4;
    let light_z = 1.2;

    // Four plates, increasing roughness front to back.
    //
    // Each plate's tilt is **solved for**, not fixed: a plate reflects the light
    // row toward the camera only if its normal bisects the incoming view
    // direction and the outgoing direction to the lights. A single shared tilt
    // aims only the nearest plate correctly and leaves the rest dark, which is
    // exactly what a first attempt at this scene produced.
    let depth = 1.5f32;
    for (i, roughness) in [0.045f32, 0.09, 0.18, 0.35].into_iter().enumerate() {
        // Neutral metal, so the comparison is about variance rather than colour.
        materials.push(GpuMaterial::metal(Vec3::splat(0.85), roughness));

        // Rising as they recede, like stadium seating: descending plates would
        // simply hide behind the one in front of them.
        let centre = Vec3::new(0.0, 0.62 * i as f32, 0.6 + i as f32 * 1.9);
        let to_camera = (eye - centre).normalize();
        let to_light = (Vec3::new(0.0, light_y, light_z) - centre).normalize();
        // The half-vector between the two is the normal that reflects one into
        // the other.
        let n = (to_camera + to_light).normalize();

        // Build edges whose cross product is that normal. `edge_v` runs along x;
        // `edge_u` is then fixed up to a sign by requiring
        // cross(edge_u, edge_v) to be parallel to n.
        let edge_u = Vec3::new(0.0, -depth * n.z, depth * n.y);
        let edge_v = Vec3::new(10.0, 0.0, 0.0);
        let origin = centre - 0.5 * edge_u - 0.5 * edge_v;

        primitives.push(GpuPrimitive::quad(
            origin,
            edge_u,
            edge_v,
            materials.len() as u32 - 1,
        ));
    }

    // Four emitters of very different size but equal total power, so the only
    // difference between them is how hard each is to *find*. Area spans a factor
    // of 440 between the smallest and the largest.
    const POWER: f32 = 46.0;
    for (j, half_width) in [0.055f32, 0.16, 0.46, 1.15].into_iter().enumerate() {
        // Negated so the size ramp reads left to right on screen: this camera
        // looks down +z, which puts world -x on the right of the image.
        let x = 3.7 - j as f32 * 2.45;
        let area = 4.0 * half_width * half_width;
        materials.push(GpuMaterial::emissive(Vec3::splat(POWER / area)));
        primitives.push(GpuPrimitive::quad(
            Vec3::new(x - half_width, light_y, light_z - half_width),
            Vec3::new(2.0 * half_width, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 2.0 * half_width),
            materials.len() as u32 - 1,
        ));
    }

    // A backdrop so the plates are not floating in a void.
    primitives.push(GpuPrimitive::quad(
        Vec3::new(-16.0, -3.0, 10.5),
        Vec3::new(0.0, 16.0, 0.0),
        Vec3::new(32.0, 0.0, 0.0),
        0,
    ));

    let mut def = SceneDef {
        name: "mis-scene",
        description: "Glossy plates of increasing roughness lit by emitters of increasing size \
                      and equal power. BSDF sampling finds the small bright emitter only by \
                      luck; light sampling scatters samples across the large dim one that a \
                      near-mirror then evaluates at almost zero. Multiple importance sampling \
                      beats both.",
        scene: Scene {
            blob: SceneBlob {
                materials,
                primitives,
                ..Default::default()
            },
            ..Default::default()
        },
        camera: Camera::look_at(eye, Vec3::new(0.0, 1.35, 3.4), 42.0),
        background: Vec3::splat(0.015),
    };
    def.scene.finalize();
    def
}

/// All scenes, in menu order.
pub fn all() -> Vec<SceneDef> {
    vec![
        cornell_box(),
        glass_box(),
        sunset(),
        instance_forest(),
        cornell_mesh(),
        mis_scene(),
        metal_sweep(),
        furnace_test(),
        bvh_stress(),
    ]
}

/// The Cornell box with three dielectric spheres: clear, dense, and rough.
///
/// Built to make the specific failure modes of a transmission implementation
/// visible rather than to look pretty:
///
/// * the clear sphere inverts the image behind it, which only happens if the
///   refracted direction bends the right way;
/// * the dense sphere (IOR 2.42, diamond) bends far harder and shows total
///   internal reflection around its rim, which is where a missing TIR branch
///   turns into a bright halo;
/// * the rough sphere blurs what it transmits, which is the part a smooth-only
///   implementation cannot do at all;
/// * the tinted slab in front of the back wall colours what passes through it
///   while keeping a white highlight, which separates transmission tint from
///   diffuse albedo.
///
/// The light is larger and lower than the reference box's so the spheres are lit
/// from an angle that actually produces caustics on the floor.
pub fn glass_box() -> SceneDef {
    const S: f32 = 555.0;

    const WHITE: u32 = 0;
    const RED: u32 = 1;
    const GREEN: u32 = 2;
    const LIGHT: u32 = 3;
    const CLEAR: u32 = 4;
    const DENSE: u32 = 5;
    const ROUGH: u32 = 6;
    const TINTED: u32 = 7;

    let materials = vec![
        GpuMaterial::diffuse(Vec3::new(0.725, 0.710, 0.680)),
        GpuMaterial::diffuse(Vec3::new(0.630, 0.065, 0.050)),
        GpuMaterial::diffuse(Vec3::new(0.140, 0.450, 0.091)),
        GpuMaterial::emissive(Vec3::new(30.0, 26.0, 18.0)),
        // Window glass.
        GpuMaterial::glass(Vec3::ONE, 0.0, 1.5),
        // Diamond: bends hardest, and its critical angle is 24 degrees, so most
        // of the inside of the sphere is in total internal reflection.
        GpuMaterial::glass(Vec3::ONE, 0.0, 2.42),
        // Frosted.
        GpuMaterial::glass(Vec3::ONE, 0.25, 1.5),
        // Tinted, to separate transmission colour from diffuse colour.
        GpuMaterial::glass(Vec3::new(0.35, 0.75, 0.55), 0.02, 1.5),
    ];

    let mut primitives = vec![
        quad(Vec3::ZERO, Vec3::new(0.0, 0.0, S), Vec3::new(S, 0.0, 0.0), WHITE),
        quad(
            Vec3::new(0.0, S, 0.0),
            Vec3::new(S, 0.0, 0.0),
            Vec3::new(0.0, 0.0, S),
            WHITE,
        ),
        quad(
            Vec3::new(0.0, 0.0, S),
            Vec3::new(0.0, S, 0.0),
            Vec3::new(S, 0.0, 0.0),
            WHITE,
        ),
        quad(
            Vec3::new(S, 0.0, 0.0),
            Vec3::new(0.0, 0.0, S),
            Vec3::new(0.0, S, 0.0),
            RED,
        ),
        quad(
            Vec3::ZERO,
            Vec3::new(0.0, S, 0.0),
            Vec3::new(0.0, 0.0, S),
            GREEN,
        ),
        // A wider light than the reference box, so the caustics under the
        // spheres are bright enough to see without a thousand samples.
        quad(
            Vec3::new(163.0, S - 0.1, 177.0),
            Vec3::new(230.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 205.0),
            LIGHT,
        ),
        // A tinted pane standing in front of the back wall. Thin, so the
        // interface tint is a reasonable stand-in for Beer-Lambert absorption.
        quad(
            Vec3::new(60.0, 40.0, 430.0),
            Vec3::new(0.0, 300.0, 0.0),
            Vec3::new(230.0, 0.0, 0.0),
            TINTED,
        ),
    ];

    primitives.push(GpuPrimitive::sphere(Vec3::new(150.0, 110.0, 160.0), 110.0, CLEAR));
    primitives.push(GpuPrimitive::sphere(Vec3::new(390.0, 95.0, 140.0), 95.0, DENSE));
    primitives.push(GpuPrimitive::sphere(Vec3::new(300.0, 85.0, 330.0), 85.0, ROUGH));

    let mut def = SceneDef {
        name: "glass-box",
        description: "Dielectrics: a clear sphere, a diamond-IOR sphere showing total internal \
                      reflection, a frosted sphere, and a tinted pane. The inverted image \
                      through the clear sphere and the caustics on the floor are what a correct \
                      refraction looks like; a sign error in the half-vector produces glass that \
                      is merely shiny.",
        scene: Scene {
            blob: SceneBlob {
                materials,
                primitives,
                ..Default::default()
            },
            ..Default::default()
        },
        camera: Camera::look_at(
            Vec3::new(278.0, 278.0, -800.0),
            Vec3::new(278.0, 278.0, 0.0),
            37.0,
        ),
        background: Vec3::ZERO,
    };
    def.scene.finalize();
    def
}

/// An outdoor scene lit only by an environment map.
///
/// No area lights at all, which is the point: every photon in the image comes
/// from the sky, so the environment sampler is the only thing standing between
/// this and a black frame. The sun is small and about four orders of magnitude
/// brighter than the sky around it, so BSDF sampling alone finds it rarely
/// enough that the difference between sampling strategies is visible rather
/// than statistical — see `envmap_sampling_converges_faster`.
///
/// The spheres are chosen to exercise the three ways a surface can gather
/// environment light: diffuse (integrates the whole sky), smooth metal
/// (mirrors it, so the sun appears as a hard highlight), and glass (refracts
/// it).
pub fn sunset() -> SceneDef {
    const GROUND: u32 = 0;
    const WHITE: u32 = 1;
    const METAL: u32 = 2;
    const GLASS: u32 = 3;

    let materials = vec![
        GpuMaterial::diffuse(Vec3::new(0.38, 0.35, 0.30)),
        GpuMaterial::diffuse(Vec3::splat(0.78)),
        GpuMaterial::metal(Vec3::new(0.95, 0.93, 0.88), 0.06),
        GpuMaterial::glass(Vec3::ONE, 0.0, 1.5),
    ];

    let primitives = vec![
        // A large ground quad rather than an infinite plane: the renderer has no
        // infinite primitive, and the sky fills everything past its edge anyway.
        quad(
            Vec3::new(-40.0, 0.0, -40.0),
            Vec3::new(80.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 80.0),
            GROUND,
        ),
        GpuPrimitive::sphere(Vec3::new(-2.4, 1.0, 0.0), 1.0, WHITE),
        GpuPrimitive::sphere(Vec3::new(0.0, 1.0, 0.0), 1.0, METAL),
        GpuPrimitive::sphere(Vec3::new(2.4, 1.0, 0.0), 1.0, GLASS),
    ];

    let mut def = SceneDef {
        name: "sunset",
        description: "Lit entirely by an environment map — no area lights. A low sun about \
                      four orders of magnitude brighter than the sky around it, over a diffuse \
                      sphere, a near-mirror, and a glass ball. The diffuse sphere converges \
                      quickly; the speckle around the specular pair is caustic noise, which a \
                      plain path tracer samples poorly however good the sky sampler is.",
        scene: Scene {
            blob: SceneBlob {
                materials,
                primitives,
                ..Default::default()
            },
            // A low sun, so it rakes across the ground and casts long shadows.
            // 256x128 rather than 512x256, which is a size decision and not a
            // quality one: at 512 the packed map is 2.5 MB and crosses the
            // threshold above which scene assets are fetched rather than
            // committed, and a sky is not worth that for a demo. At 256 one
            // texel spans 1.4 degrees, so `procedural_sky` widens the 1-degree
            // sun to match and dims it by the area ratio — the sun covers more
            // pixels and delivers the same power.
            env: crate::envmap::procedural_sky(
                256,
                128,
                Vec3::new(-0.45, 0.22, -0.86),
                6000.0,
                1.0,
            ),
            ..Default::default()
        },
        camera: Camera::look_at(
            Vec3::new(0.0, 1.6, 7.5),
            Vec3::new(0.0, 0.9, 0.0),
            42.0,
        ),
        // Unused: the environment map takes precedence when it is non-empty.
        background: Vec3::ZERO,
    };
    def.scene.finalize();
    def
}

/// One mesh, many placements — the case a two-level hierarchy exists for.
///
/// Forty-nine copies of a single icosphere on a ground plane, under the sky.
/// Stored once: the whole scene's geometry is one mesh's worth of triangles plus
/// forty-nine 64-byte transforms, and the acceleration structure is one BLAS
/// plus a tiny TLAS.
///
/// The placements vary in rotation and **non-uniform** scale on purpose. A grid
/// of translated copies would exercise almost nothing — translation leaves
/// normals alone and maps boxes to boxes — whereas non-uniform scale is what
/// separates a correct inverse-transpose normal transform from the two usual
/// wrong ones, and rotation is what makes a lazy world-bounds computation cull
/// visible geometry.
pub fn instance_forest() -> SceneDef {
    const GROUND: u32 = 0;
    const SHELL: u32 = 1;
    const GOLD: u32 = 2;

    let mut blob = SceneBlob {
        materials: vec![
            GpuMaterial::diffuse(Vec3::new(0.34, 0.32, 0.29)),
            GpuMaterial::glossy(Vec3::new(0.70, 0.35, 0.25), 0.35),
            GpuMaterial::metal(Vec3::new(0.95, 0.80, 0.40), 0.18),
        ],
        primitives: vec![quad(
            Vec3::new(-30.0, 0.0, -30.0),
            Vec3::new(60.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 60.0),
            GROUND,
        )],
        ..Default::default()
    };

    // The one mesh every instance shares.
    mesh::icosphere(Vec3::ZERO, 0.5, 2).append_to(&mut blob, SHELL);
    let tri_count = blob.triangles.len() as u32;

    // A deterministic scatter. The renderer's own RNG, so the layout is
    // reproducible and the scene is a fixture rather than a one-off.
    let mut rng = crate::rng::Rng::new(0x1057, 0, 0);
    let mut placements = Vec::new();
    for gx in 0..7i32 {
        for gz in 0..7i32 {
            let jitter = |r: &mut crate::rng::Rng| r.next_f32() - 0.5;
            let x = gx as f32 * 1.6 - 4.8 + jitter(&mut rng) * 0.5;
            let z = gz as f32 * 1.6 - 4.8 + jitter(&mut rng) * 0.5;
            let scale = Vec3::new(
                0.7 + rng.next_f32() * 0.7,
                0.5 + rng.next_f32() * 1.1,
                0.7 + rng.next_f32() * 0.7,
            );
            let spin = rng.next_f32() * std::f32::consts::TAU;
            // Sit each one on the ground: the sphere's radius is 0.5 before
            // scaling, so its centre rides at half the scaled height.
            let y = 0.5 * scale.y;
            let m = Mat4::from_translation(Vec3::new(x, y, z))
                * Mat4::from_rotation_y(spin)
                * Mat4::from_scale(scale);
            // Every third one is gold, which costs a material index rather than
            // a second copy of the mesh.
            let material = if (gx + gz) % 3 == 0 { GOLD } else { u32::MAX };
            placements.push((m, 0u32, material));
        }
    }

    let mut def = SceneDef {
        name: "instance-forest",
        description: "Forty-nine copies of one icosphere, stored once. The scene's geometry is \
                      a single mesh plus forty-nine transforms, and its acceleration structure \
                      is one BLAS under a small TLAS. Rotation and non-uniform scale throughout, \
                      because a grid of translated copies would exercise none of the maths that \
                      can go wrong.",
        scene: Scene {
            blob,
            env: crate::envmap::procedural_sky(
                256,
                128,
                Vec3::new(-0.4, 0.35, -0.85),
                6000.0,
                1.0,
            ),
            ..Default::default()
        },
        camera: Camera::look_at(
            Vec3::new(0.0, 3.2, -11.0),
            Vec3::new(0.0, 0.7, 0.0),
            40.0,
        ),
        background: Vec3::ZERO,
    };
    def.scene.build_instances(&[(0, tri_count)], &placements);
    def.scene.blob.lights = crate::light::build_lights(&def.scene.blob);
    def
}

pub fn by_name(name: &str) -> Option<SceneDef> {
    all().into_iter().find(|s| s.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{Ray, T_MAX};

    /// Every wall normal must point toward the interior. A flipped wall is
    /// invisible in a diffuse render (the normal is re-oriented against the ray
    /// on hit) right up until one-sided emission or a transmissive material
    /// makes it matter, at which point it is baffling.
    #[test]
    fn cornell_normals_point_inward() {
        let def = cornell_box();
        let centre = Vec3::splat(277.5);
        for (i, q) in def
            .scene
            .blob
            .primitives
            .iter()
            .filter(|p| p.is_quad())
            .enumerate()
        {
            let n = Vec3::from_array(q.normal);
            let o = Vec3::from_array(q.position)
                + 0.5 * Vec3::from_array(q.edge_u)
                + 0.5 * Vec3::from_array(q.edge_v);
            let toward_centre = (centre - o).normalize();
            assert!(
                n.dot(toward_centre) > 0.0,
                "quad {i} normal {n} faces away from the box interior"
            );
            assert!(
                (n.length() - 1.0).abs() < 1e-5,
                "quad {i} normal is not unit length"
            );
        }
    }

    /// The box has five walls and an **open front** at z = 0 — that is the
    /// canonical setup, and it is how the camera sees inside. So the invariant
    /// is not "closed" but "no gaps": a ray fired from the interior may only
    /// escape through the front opening, never through a seam between walls.
    ///
    /// A seam leaks light and darkens the whole render by an amount that looks
    /// entirely plausible, which is why this is worth asserting rather than
    /// eyeballing.
    #[test]
    fn cornell_box_has_no_gaps_except_the_open_front() {
        let def = cornell_box();
        let centre = Vec3::splat(277.5);
        let mut rng = crate::rng::Rng::new(7, 0, 0);
        for _ in 0..50_000 {
            // Uniform on the sphere: z is uniform in [-1, 1] (Archimedes), and
            // the azimuth is uniform in [0, 2*pi).
            let z = 2.0 * rng.next_f32() - 1.0;
            let phi = 2.0 * std::f32::consts::PI * rng.next_f32();
            let r = (1.0 - z * z).max(0.0).sqrt();
            let dir = Vec3::new(r * phi.cos(), r * phi.sin(), z);
            let ray = Ray {
                origin: centre,
                dir,
            };

            if let Some(hit) = def.scene.intersect(&ray) {
                assert!(hit.t < T_MAX);
                continue;
            }

            // Escaped. It must be heading out the front, and it must cross the
            // z = 0 plane within the opening — otherwise it went through a wall.
            assert!(
                dir.z < 0.0,
                "ray escaped backwards through a wall: dir = {dir}"
            );
            let t = centre.z / -dir.z;
            let exit = ray.origin + t * dir;
            assert!(
                (0.0..=555.0).contains(&exit.x) && (0.0..=555.0).contains(&exit.y),
                "ray escaped through a seam, crossing z = 0 at {exit}"
            );
        }
    }

    /// No primary ray may escape through the open front before entering the box:
    /// that would put an unlit black border around the render. Equivalent to
    /// "the FOV frames the opening", asserted numerically so a camera tweak
    /// cannot silently reintroduce the border.
    #[test]
    fn every_primary_ray_enters_the_box() {
        let def = cornell_box();
        let params = crate::integrator::RenderParams {
            width: 64,
            height: 64,
            ..Default::default()
        };
        let u = crate::integrator::build_uniforms(&def, &params);
        for y in 0..64u32 {
            for x in 0..64u32 {
                // Corners of the pixel footprint, to catch the extreme rays.
                for (jx, jy) in [(0.0, 0.0), (0.999, 0.0), (0.0, 0.999), (0.999, 0.999)] {
                    let ray = crate::camera::generate_ray(
                        &u,
                        x,
                        y,
                        glam::Vec2::new(jx, jy),
                        glam::Vec2::ZERO,
                    );
                    let t = -ray.origin.z / ray.dir.z;
                    let p = ray.origin + t * ray.dir;
                    assert!(
                        (0.0..=555.0).contains(&p.x) && (0.0..=555.0).contains(&p.y),
                        "primary ray at pixel ({x}, {y}) jitter ({jx}, {jy}) misses the opening at {p}"
                    );
                }
            }
        }
    }

    #[test]
    fn spheres_are_inside_the_box_and_disjoint() {
        let def = cornell_box();
        for s in def.scene.blob.primitives.iter().filter(|p| !p.is_quad()) {
            let c = Vec3::from_array(s.position);
            for axis in 0..3 {
                assert!(
                    c[axis] - s.radius > -1e-3,
                    "sphere pokes through the low wall on axis {axis}"
                );
                assert!(
                    c[axis] + s.radius < 555.0 + 1e-3,
                    "sphere pokes through the high wall on axis {axis}"
                );
            }
        }
        let spheres: Vec<_> = def
            .scene
            .blob
            .primitives
            .iter()
            .filter(|p| !p.is_quad())
            .collect();
        let a = spheres[0];
        let b = spheres[1];
        let d = (Vec3::from_array(a.position) - Vec3::from_array(b.position)).length();
        assert!(d > a.radius + b.radius, "spheres intersect");
    }

    #[test]
    fn exactly_one_emitter() {
        let def = cornell_box();
        let emitters: Vec<_> = def
            .scene
            .blob
            .materials
            .iter()
            .enumerate()
            .filter(|(_, m)| Vec3::from_array(m.emissive).max_element() > 0.0)
            .collect();
        assert_eq!(emitters.len(), 1, "expected one emissive material");
    }

    /// Every diffuse albedo must be < 1 in every channel. An albedo of exactly 1
    /// in a closed box makes the light-transport series diverge, and > 1 is
    /// unphysical; both show up as an image that brightens without bound as the
    /// bounce limit rises.
    #[test]
    fn albedos_are_energy_conserving() {
        let def = cornell_box();
        for (i, m) in def.scene.blob.materials.iter().enumerate() {
            let a = Vec3::from_array(m.base_color);
            assert!(a.max_element() < 1.0, "material {i} albedo {a} is not < 1");
            assert!(
                a.min_element() >= 0.0,
                "material {i} albedo {a} is negative"
            );
        }
    }
}

#[cfg(test)]
mod mesh_scene_tests {
    use super::*;
    use crate::image::compare;
    use crate::integrator::{render, RenderParams};

    /// Every shipped scene must build a sound acceleration structure.
    ///
    /// Which structure that is depends on the scene: an instanced one has no
    /// single-level BVH at all, because every triangle belongs to a BLAS and
    /// lives in its mesh's object space. Validating each BLAS and the TLAS is
    /// the equivalent check, not a weaker one.
    #[test]
    fn all_scenes_have_valid_acceleration_structures() {
        for def in all() {
            let tris = def.scene.blob.triangles.len();

            if !def.scene.instances.is_empty() {
                for (i, blas) in def.scene.instances.blas.iter().enumerate() {
                    let problems = blas.bvh.validate(blas.triangle_count as usize);
                    assert!(
                        problems.is_empty(),
                        "{} BLAS {i}: {problems:#?}",
                        def.name
                    );
                    assert!(
                        !blas.bvh.is_empty(),
                        "{} BLAS {i} has no nodes",
                        def.name
                    );
                }
                let n = def.scene.instances.instances.len();
                let problems = def.scene.instances.tlas.validate(n);
                assert!(problems.is_empty(), "{} TLAS: {problems:#?}", def.name);
                assert_eq!(
                    def.scene.blob.instances.len(),
                    n,
                    "{}: packed instance count does not match",
                    def.name
                );
                continue;
            }

            let problems = def.scene.bvh.validate(tris);
            assert!(problems.is_empty(), "{}: {problems:#?}", def.name);
            if tris > 0 {
                assert!(
                    !def.scene.bvh.is_empty(),
                    "{} has triangles but no BVH",
                    def.name
                );
            }
        }
    }

    #[test]
    fn scene_names_are_unique() {
        let mut names: Vec<&str> = all().iter().map(|s| s.name).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(names.len(), before, "duplicate scene names");
    }

    /// A Cornell box whose walls are triangles must render *identically* to one
    /// whose walls are analytic quads.
    ///
    /// This is the sharpest test of the triangle path. A quad split into two
    /// triangles is exact, not approximate, so the two scenes are the same
    /// geometry expressed two ways — and they share the same RNG stream. Any
    /// difference beyond floating-point noise means Möller–Trumbore, the BVH, or
    /// the normal handling is wrong.
    ///
    /// The spheres stay analytic in both, so tessellation cannot contaminate
    /// the comparison.
    #[test]
    fn triangle_walls_match_analytic_quads() {
        let analytic = cornell_box();

        let mut triangulated = cornell_box();
        let prims = std::mem::take(&mut triangulated.scene.blob.primitives);
        // Keep the analytic spheres so tessellation cannot contaminate the
        // comparison; only the walls become triangles.
        for p in prims.iter().filter(|p| p.is_quad()) {
            crate::mesh::quad(
                Vec3::from_array(p.position),
                Vec3::from_array(p.edge_u),
                Vec3::from_array(p.edge_v),
            )
            .append_to(&mut triangulated.scene.blob, p.material);
        }
        triangulated
            .scene
            .blob
            .primitives
            .extend(prims.iter().filter(|p| !p.is_quad()));
        triangulated.scene.build_bvh();

        let params = RenderParams {
            width: 128,
            height: 128,
            samples: 64,
            max_depth: 6,
            ..Default::default()
        };
        let a = render(&analytic, &params);
        let b = render(&triangulated, &params);
        let d = compare(&a, &b).expect("same size");

        eprintln!(
            "quads vs triangles: mean rel {:.3e}, rmse {:.3e}, energy ratio {:.6}",
            d.mean_rel,
            d.rmse,
            d.mean_b / d.mean_a
        );
        assert!(
            d.mean_rel < 5e-3,
            "triangulated walls differ from analytic quads by {:.3e} mean relative error \
             (max {:.3e} at {:?})",
            d.mean_rel,
            d.max_abs,
            d.max_abs_at
        );
        assert!(
            (d.mean_b / d.mean_a - 1.0).abs() < 2e-3,
            "energy ratio {:.6} — the triangle path is losing or gaining light",
            d.mean_b / d.mean_a
        );
    }

    /// With the BVH removed, traversal falls back to brute force. Both must
    /// produce the same image — this is the whole-renderer version of the
    /// per-ray equivalence test, and it catches anything that only shows up
    /// through the full light transport.
    #[test]
    fn bvh_and_brute_force_render_the_same_image() {
        let with_bvh = cornell_mesh();
        let mut without = cornell_mesh();
        without.scene.bvh = Default::default();

        let params = RenderParams {
            width: 96,
            height: 96,
            samples: 24,
            max_depth: 5,
            ..Default::default()
        };
        let a = render(&with_bvh, &params);
        let b = render(&without, &params);
        let d = compare(&a, &b).expect("same size");

        eprintln!(
            "bvh vs brute force: mean rel {:.3e}, max abs {:.3e}",
            d.mean_rel, d.max_abs
        );
        // These take *identical* code paths through Möller–Trumbore, so they
        // should agree to the last bit, not merely closely.
        assert!(
            d.mean_rel < 1e-6,
            "BVH traversal and brute force disagree by {:.3e} — the BVH is missing geometry",
            d.mean_rel
        );
    }

    /// Report BVH quality for every scene. Not an assertion so much as a record
    /// that the numbers are sane, and the baseline the GPU LBVH gets compared
    /// against at build step 11.
    #[test]
    fn report_bvh_quality() {
        for def in all() {
            let s = def.scene.bvh.stats;
            if s.triangles == 0 {
                eprintln!("{:<14} no triangles", def.name);
                continue;
            }
            eprintln!(
                "{:<14} {:>7} tris  {:>7} nodes  {:>6} leaves  depth {:>2}  \
                 mean leaf {:.1}  SAH {:>7.2}  build {:.1} ms",
                def.name,
                s.triangles,
                s.nodes,
                s.leaves,
                s.max_depth,
                s.mean_leaf_size,
                s.sah_cost,
                s.build_seconds * 1000.0
            );
            assert!(
                s.sah_cost < s.triangles as f32 * 0.05,
                "{}: poor SAH cost",
                def.name
            );
            assert!(
                s.max_depth < 64,
                "{}: depth {} exceeds the traversal stack",
                def.name,
                s.max_depth
            );
        }
    }
}

#[cfg(test)]
mod furnace_scene_test {
    use super::*;
    use crate::integrator::{render, RenderParams};

    /// **The white furnace test, at whole-renderer level.**
    ///
    /// Renders the furnace scene and asserts that *every pixel* is 1.0 — that
    /// the spheres really are invisible, not merely close.
    ///
    /// This is stronger than the BSDF-level test in `bsdf`, because it goes
    /// through the entire pipeline: intersection, the local frame construction,
    /// lobe selection, the combined pdf, multi-bounce throughput, and the
    /// background. An error anywhere in that chain shows up here as a sphere you
    /// can see, and nowhere else would catch, say, a shading frame that is
    /// subtly non-orthonormal.
    ///
    /// The bounce limit has to be generous: with a white non-absorbing BSDF,
    /// energy only escapes by reaching the environment, so truncating paths
    /// early *removes* energy and darkens the spheres. That is not a BSDF bug —
    /// it is the truncation bias, and it is why the test uses depth 24.
    #[test]
    fn furnace_scene_renders_the_spheres_invisible() {
        let def = furnace_test();
        let params = RenderParams {
            width: 160,
            height: 80,
            samples: 400,
            max_depth: 24,
            ..Default::default()
        };
        let film = render(&def, &params);

        let mut worst = 0.0f32;
        let mut worst_at = (0u32, 0u32);
        let mut sum = 0.0f64;
        for y in 0..film.height {
            for x in 0..film.width {
                let p = film.pixel(x, y);
                sum += p.x as f64;
                let d = (p.x - 1.0).abs();
                if d > worst {
                    worst = d;
                    worst_at = (x, y);
                }
                assert!(p.is_finite(), "pixel ({x}, {y}) is {p}");
            }
        }
        let mean = sum / (film.width * film.height) as f64;
        // Root-mean-square deviation from 1: an estimate of the noise level,
        // which is a far more stable thing to assert on than the single worst
        // pixel.
        let rms = (film
            .data
            .iter()
            .map(|p| ((p.x - 1.0) as f64).powi(2))
            .sum::<f64>()
            / film.data.len() as f64)
            .sqrt();
        eprintln!(
            "furnace scene: mean {mean:.5}, rms deviation {rms:.4}, worst pixel {worst:.4} at {worst_at:?}"
        );

        // The mean is the real test. Per-pixel Monte Carlo noise averages out;
        // any systematic energy gain or loss survives.
        assert!(
            (mean - 1.0).abs() < 3e-3,
            "furnace mean radiance {mean:.5}, expected 1.0 — the BSDF is not energy conserving"
        );

        // The noise level is asserted only loosely, and the *worst* pixel not at
        // all tightly. A white non-absorbing BSDF over 24 bounces has high
        // per-sample variance, so the extreme pixel moves substantially with the
        // random stream — an earlier version of this test pinned it at 0.12 and
        // failed the moment next event estimation changed how many random
        // numbers each vertex draws, despite the mean being unchanged to five
        // decimal places. A bound that fails for reasons unrelated to what it is
        // testing is worse than a loose one.
        assert!(
            rms < 0.1,
            "furnace noise level {rms:.4} is far above expectation"
        );
        assert!(
            worst < 0.5,
            "pixel {worst_at:?} deviates from 1.0 by {worst:.4}, which is a hole or a firefly \
             rather than noise"
        );
    }
}
