//! Instancing: one mesh, many placements, through a two-level hierarchy.
//!
//! # What a second level buys
//!
//! A single BVH over every triangle in the scene costs memory proportional to
//! the triangles actually stored. Rendering a hundred copies of a 10 000-triangle
//! mesh that way means a million triangles and a BVH over all of them — even
//! though ninety-nine of the copies are the same geometry seen from a different
//! angle.
//!
//! A two-level hierarchy stores the mesh once:
//!
//! * the **BLAS** (bottom level) is an ordinary BVH over one mesh's triangles,
//!   in that mesh's own coordinate system, built once and shared;
//! * the **TLAS** (top level) is a BVH over *instances* — a transform plus a
//!   reference to a BLAS.
//!
//! A ray walks the TLAS in world space. When it reaches an instance it is
//! transformed into that instance's object space and walks the BLAS there.
//!
//! # The trick that makes it cheap: do not normalise
//!
//! Transforming a ray means transforming its origin and direction by the
//! instance's **inverse** matrix. The direction is deliberately left
//! *unnormalised*.
//!
//! That is not a shortcut, it is the point. If the object-space direction keeps
//! the world-space direction's scale, then a hit at parameter `t` in object
//! space is at the same `t` in world space — so `t` values from different
//! instances are directly comparable, the traversal's `t_max` culling keeps
//! working across levels, and the hit point needs no transforming back. Normalise
//! it and every `t` comes back in a different unit, which produces a scene where
//! scaled instances sort incorrectly against one another and nothing looks
//! obviously wrong.
//!
//! # Why only the inverse is stored
//!
//! The forward matrix is never needed. The ray goes world -> object through the
//! inverse; the hit point comes back for free (see above); and a **normal**
//! transforms by the inverse-transpose, which is the transpose of the 3x3 part
//! of the inverse — already in hand. Storing both would be 48 more bytes per
//! instance and one more thing to keep consistent.

use crate::bvh::Aabb;
use glam::{Mat3, Mat4, Vec3};

/// One placement of a BLAS.
#[derive(Clone, Copy, Debug)]
pub struct Instance {
    /// World-to-object. See the module note on why the forward matrix is absent.
    pub world_to_object: Mat4,
    /// Index into [`crate::scene::SceneBlobExt::blas`] — which mesh this is.
    pub blas: u32,
    /// Material for every triangle of this instance, overriding the mesh's own.
    ///
    /// `u32::MAX` means "keep the triangle's material", which is what makes a
    /// single mesh usable both as itself and as a recoloured copy without
    /// duplicating its triangles.
    pub material_override: u32,
}

/// A mesh's bottom-level acceleration structure.
#[derive(Clone, Debug, Default)]
pub struct Blas {
    /// Node range within the shared node array.
    pub node_offset: u32,
    pub node_count: u32,
    /// Triangle range within the shared triangle array.
    pub triangle_offset: u32,
    pub triangle_count: u32,
    /// Object-space bounds of the whole mesh, for building the TLAS.
    pub bounds: Aabb,
    /// The mesh's own BVH, in object space, built once and shared by every
    /// instance that references it.
    pub bvh: crate::bvh::Bvh,
}

impl Instance {
    /// Place a BLAS with a world transform.
    ///
    /// Takes the **forward** transform because that is what a caller thinks in —
    /// "put it here, this big" — and inverts once, here, rather than at every
    /// ray.
    pub fn new(object_to_world: Mat4, blas: u32, material_override: u32) -> Instance {
        Instance {
            world_to_object: object_to_world.inverse(),
            blas,
            material_override,
        }
    }

    /// World-space bounds of this instance's BLAS.
    ///
    /// All eight corners of the object-space box are transformed and re-bounded,
    /// rather than transforming the two extreme corners. Transforming only the
    /// corners is a classic error: under rotation the min and max corners do not
    /// map to the min and max of the result, and the box comes out too small —
    /// which silently culls geometry rather than merely wasting traversal.
    pub fn world_bounds(&self, object_bounds: &Aabb) -> Aabb {
        let to_world = self.world_to_object.inverse();
        let (lo, hi) = (object_bounds.min, object_bounds.max);
        let mut out = Aabb::default();
        for i in 0..8 {
            let corner = Vec3::new(
                if i & 1 == 0 { lo.x } else { hi.x },
                if i & 2 == 0 { lo.y } else { hi.y },
                if i & 4 == 0 { lo.z } else { hi.z },
            );
            out.grow_point(to_world.transform_point3(corner));
        }
        out.pad();
        out
    }

    /// Ray origin and direction in this instance's object space.
    ///
    /// The direction is **not** normalised — see the module note. `t` is then
    /// the same number in both spaces.
    #[inline]
    pub fn transform_ray(&self, origin: Vec3, direction: Vec3) -> (Vec3, Vec3) {
        (
            self.world_to_object.transform_point3(origin),
            self.world_to_object.transform_vector3(direction),
        )
    }

    /// Take an object-space normal back to world space.
    ///
    /// By the inverse-transpose, which for a world-to-object matrix `W` is
    /// `transpose(W_3x3)` — no inversion needed, because `W` is already the
    /// inverse.
    ///
    /// Using the matrix itself instead of its inverse-transpose is the other
    /// classic error, and it is invisible until something is *non-uniformly*
    /// scaled: under uniform scale and rotation the two agree up to length, and
    /// the normal is renormalised anyway.
    #[inline]
    pub fn transform_normal(&self, n: Vec3) -> Vec3 {
        let m = Mat3::from_mat4(self.world_to_object);
        (m.transpose() * n).normalize_or(Vec3::Z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_4;

    fn probe_transforms() -> Vec<(&'static str, Mat4)> {
        vec![
            ("identity", Mat4::IDENTITY),
            ("translate", Mat4::from_translation(Vec3::new(3.0, -2.0, 5.0))),
            ("uniform scale", Mat4::from_scale(Vec3::splat(2.5))),
            (
                "non-uniform scale",
                Mat4::from_scale(Vec3::new(2.0, 0.5, 3.0)),
            ),
            ("rotate", Mat4::from_rotation_y(FRAC_PI_4)),
            (
                "compound",
                Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0))
                    * Mat4::from_rotation_z(0.7)
                    * Mat4::from_scale(Vec3::new(1.5, 0.4, 2.0)),
            ),
        ]
    }

    /// A hit's `t` must mean the same thing in both spaces.
    ///
    /// The property the whole design rests on. If it fails, instances at
    /// different scales sort incorrectly against each other and against
    /// un-instanced geometry.
    #[test]
    fn t_is_preserved_by_the_ray_transform() {
        for (name, m) in probe_transforms() {
            let inst = Instance::new(m, 0, u32::MAX);
            let origin = Vec3::new(-1.0, 0.5, -4.0);
            let dir = Vec3::new(0.2, 0.1, 1.0).normalize();
            let (o, d) = inst.transform_ray(origin, dir);

            for &t in &[0.5f32, 1.0, 7.25] {
                let world_point = origin + t * dir;
                let object_point = o + t * d;
                // The object-space point at parameter t must be the world point
                // taken through the same transform.
                let expected = m.inverse().transform_point3(world_point);
                assert!(
                    (object_point - expected).length() < 1e-3,
                    "{name}: at t = {t} the object-space point is {object_point:?} \
                     but transforming the world point gives {expected:?}. The \
                     direction is probably being normalised, which rescales t."
                );
            }
        }
    }

    /// Normals must stay perpendicular to the surface after transforming.
    ///
    /// Checked against a surface rather than against a formula: take two
    /// tangents, transform them and the normal, and require the normal to still
    /// be perpendicular to both. That catches using the matrix where the
    /// inverse-transpose belongs, which a formula comparison would not if the
    /// formula itself were wrong.
    #[test]
    fn normals_stay_perpendicular_under_transform() {
        for (name, m) in probe_transforms() {
            let inst = Instance::new(m, 0, u32::MAX);
            // An arbitrary surface: normal n with two tangents spanning it.
            let n = Vec3::new(0.3, 0.8, -0.5).normalize();
            let t1 = n.cross(Vec3::X).normalize();
            let t2 = n.cross(t1).normalize();

            let n_world = inst.transform_normal(n);
            // Tangents transform by the forward matrix.
            let t1_world = m.transform_vector3(t1);
            let t2_world = m.transform_vector3(t2);

            assert!(
                n_world.dot(t1_world.normalize()).abs() < 1e-3,
                "{name}: transformed normal is not perpendicular to the first \
                 tangent (dot = {}). Suspect the matrix being used where the \
                 inverse-transpose belongs — which only shows up under \
                 non-uniform scale.",
                n_world.dot(t1_world.normalize())
            );
            assert!(
                n_world.dot(t2_world.normalize()).abs() < 1e-3,
                "{name}: transformed normal is not perpendicular to the second \
                 tangent (dot = {})",
                n_world.dot(t2_world.normalize())
            );
            assert!(
                (n_world.length() - 1.0).abs() < 1e-4,
                "{name}: transformed normal is not unit length"
            );
        }
    }

    /// The world bounds must contain the transformed box, and not be loose to
    /// the point of uselessness.
    #[test]
    fn world_bounds_contain_the_rotated_box() {
        let object = Aabb {
            min: Vec3::new(-1.0, -2.0, -0.5),
            max: Vec3::new(1.0, 2.0, 0.5),
        };
        for (name, m) in probe_transforms() {
            let inst = Instance::new(m, 0, u32::MAX);
            let world = inst.world_bounds(&object);
            // Every corner of the object box must land inside.
            for i in 0..8 {
                let c = Vec3::new(
                    if i & 1 == 0 { object.min.x } else { object.max.x },
                    if i & 2 == 0 { object.min.y } else { object.max.y },
                    if i & 4 == 0 { object.min.z } else { object.max.z },
                );
                let w = m.transform_point3(c);
                assert!(
                    w.cmpge(world.min).all() && w.cmple(world.max).all(),
                    "{name}: transformed corner {w:?} is outside the world bounds \
                     {:?}..{:?}. Transforming only the two extreme corners rather \
                     than all eight gives exactly this, and it culls geometry.",
                    world.min,
                    world.max
                );
            }
        }
    }

    /// A rotated box's world bounds must actually be larger than the box.
    ///
    /// Guards the degenerate implementation that transforms `min` and `max` and
    /// calls it done: for a 45-degree rotation that returns a box the same size,
    /// which is provably too small.
    #[test]
    fn rotation_grows_the_bounds() {
        let object = Aabb {
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
        };
        let inst = Instance::new(Mat4::from_rotation_y(FRAC_PI_4), 0, u32::MAX);
        let world = inst.world_bounds(&object);
        let extent = world.max - world.min;
        // A unit cube rotated 45 degrees about Y spans 2*sqrt(2) in x and z.
        assert!(
            extent.x > 2.7 && extent.z > 2.7,
            "a 45-degree rotation should widen the box to about 2.83, got {extent:?}"
        );
        assert!(
            (extent.y - 2.0).abs() < 0.01,
            "the rotation axis should be unchanged, got {}",
            extent.y
        );
    }

    /// Inverting twice must return the original.
    #[test]
    fn the_stored_inverse_round_trips() {
        for (name, m) in probe_transforms() {
            let inst = Instance::new(m, 0, u32::MAX);
            let back = inst.world_to_object.inverse();
            for i in 0..4 {
                for j in 0..4 {
                    assert!(
                        (back.col(i)[j] - m.col(i)[j]).abs() < 1e-4,
                        "{name}: inverse round trip differs at ({i}, {j})"
                    );
                }
            }
        }
    }
}

/// A scene's instancing data: the shared meshes and their placements.
///
/// Kept beside the blob rather than inside it because the blob is the
/// byte-for-byte GPU upload and this carries host-side bookkeeping — which BLAS
/// owns which node range — that the shader recovers from the packed
/// [`crate::gpu_layout::GpuInstance`] instead.
#[derive(Clone, Debug, Default)]
pub struct InstanceSet {
    pub instances: Vec<Instance>,
    pub blas: Vec<Blas>,
    /// BVH over the instances' **world** bounds. Its leaves index `instances`.
    pub tlas: crate::bvh::Bvh,
}

impl InstanceSet {
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// Closest triangle hit across every instance.
    ///
    /// Two levels: walk the TLAS in world space, and at each instance it reaches,
    /// transform the ray and walk that instance's BLAS.
    ///
    /// `t_max` is threaded through both levels unchanged, which is only sound
    /// because the transform preserves `t` — see the module note. It is also what
    /// makes the second level cheap: an instance whose bounds start beyond the
    /// closest hit so far is rejected at the TLAS without transforming anything.
    pub fn intersect(
        &self,
        triangles: &[crate::gpu_layout::GpuTriangle],
        positions: &[[f32; 4]],
        ray: &crate::scene::Ray,
        t_min: f32,
        t_max: f32,
        stats: &mut crate::bvh::TraversalStats,
    ) -> Option<InstanceHit> {
        if self.instances.is_empty() || self.tlas.is_empty() {
            return None;
        }
        let inv_dir = crate::bvh::safe_inv_dir(ray.dir);
        let mut closest = t_max;
        let mut best: Option<InstanceHit> = None;

        // The TLAS is an ordinary BVH; only what sits in its leaves differs, so
        // it is walked here rather than through `Bvh::intersect` (which would
        // test triangles).
        let mut stack = [0u32; 32];
        let mut sp = 0usize;
        let mut node = 0u32;
        loop {
            let n = &self.tlas.nodes[node as usize];
            // Both levels count: a heatmap showing only the BLAS would hide a
            // badly built TLAS entirely, which is what the top level exists to
            // get right.
            stats.node_visits += 1;
            let bounds = self.tlas.node_bounds(node as usize);
            if bounds.hit(ray.origin, inv_dir, t_min, closest).is_some() {
                if n.count > 0 {
                    let start = n.left_first as usize;
                    for &idx in &self.tlas.prim_indices[start..start + n.count as usize] {
                        let inst = &self.instances[idx as usize];
                        let blas = &self.blas[inst.blas as usize];
                        let (o, d) = inst.transform_ray(ray.origin, ray.dir);
                        let local = crate::scene::Ray {
                            origin: o,
                            dir: d,
                        };
                        // The BLAS owns a slice of the shared triangle array;
                        // its leaves index within that slice.
                        let tri_lo = blas.triangle_offset as usize;
                        let tri_hi = tri_lo + blas.triangle_count as usize;
                        if let Some(h) = blas.bvh.intersect_with_stats(
                            &triangles[tri_lo..tri_hi],
                            positions,
                            &local,
                            t_min,
                            closest,
                            stats,
                        ) {
                            closest = h.t;
                            best = Some(InstanceHit {
                                tri: crate::bvh::TriHit {
                                    triangle: h.triangle + blas.triangle_offset,
                                    ..h
                                },
                                instance: idx,
                            });
                        }
                    }
                } else {
                    let l = n.left_first;
                    if sp + 1 < stack.len() {
                        stack[sp] = l + 1;
                        sp += 1;
                    }
                    node = l;
                    continue;
                }
            }
            if sp == 0 {
                break;
            }
            sp -= 1;
            node = stack[sp];
        }
        best
    }
}

/// A hit inside an instance, with the instance it came from.
#[derive(Clone, Copy, Debug)]
pub struct InstanceHit {
    /// Triangle index is **absolute** into the shared array, not BLAS-relative.
    pub tri: crate::bvh::TriHit,
    pub instance: u32,
}
