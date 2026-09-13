//! Scene representation and brute-force intersection.
//!
//! The CPU tracer intersects the **GPU structs directly** ([`GpuSphere`],
//! [`GpuQuad`]) rather than keeping a separate, prettier CPU-side scene type.
//! That is deliberate: a parallel CPU representation would be one more thing
//! that can disagree with what the GPU actually sees, and "the reference tracer
//! was rendering a subtly different scene" is a miserable bug to chase. One
//! representation, two intersectors, written to match line for line.
//!
//! There is no acceleration structure here — it is brute force over every
//! primitive. That is intentional for build steps 1–5: this code is the
//! ground truth the BVH gets validated *against* (build step 7 of the test plan),
//! so it must stay obviously correct rather than fast.

use crate::gpu_layout::{GpuMaterial, GpuPrimitive, SceneBlob};
use glam::Vec3;

#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Vec3,
    /// Normalised. The intersectors assume `dot(dir, dir) == 1`.
    pub dir: Vec3,
}

#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub t: f32,
    /// BVH nodes visited finding this hit, for the traversal heatmap.
    pub steps: u32,
    pub position: Vec3,
    /// **Shading** normal, oriented against the incoming ray. For a smooth mesh
    /// this is the barycentric blend of the vertex normals, so a coarsely
    /// tessellated sphere still shades as a sphere.
    pub normal: Vec3,
    /// **Geometric** normal — the actual facet, oriented against the ray.
    ///
    /// Kept separately because the two have different jobs. Ray offsetting must
    /// use the geometric normal: offsetting along an interpolated normal can
    /// push the new origin *below* the actual surface at a grazing angle, which
    /// reintroduces the self-intersection the offset exists to prevent. Shading
    /// must use the interpolated one, or every facet edge is visible.
    pub geometric_normal: Vec3,
    /// True if the ray struck the side the stored normal points toward.
    /// Emission is one-sided and keys off this.
    pub front_face: bool,
    pub material: u32,
    /// Surface area of the primitive that was hit, when it is something the
    /// light sampler could have chosen; **zero otherwise**.
    ///
    /// Multiple importance sampling needs this: having arrived at an emitter by
    /// BSDF sampling, the weight depends on how likely light sampling was to
    /// find the same point, and that density is `1 / (num_lights * area)`
    /// converted to solid angle.
    ///
    /// Zero for spheres, which `light::build_lights` does not put in the light
    /// list — so the BSDF strategy correctly takes full credit for finding them.
    pub light_area: f32,
}

/// Upper bound used as "no hit yet". Finite rather than `f32::INFINITY` so that
/// the same constant works in WGSL without worrying about how a given backend
/// handles infinities in comparisons.
pub const T_MAX: f32 = 1.0e30;
/// Lower bound on `t`. Self-intersection is handled by offsetting the ray
/// *origin* (see [`crate::math::offset_ray_origin`]), not by a large `t_min`;
/// this is only here to reject the degenerate `t == 0` root.
pub const T_MIN: f32 = 1.0e-4;

/// Ray/sphere intersection.
///
/// Solves `|o + t*d - c|^2 = r^2`. Written with the half-`b` form
/// (`h = dot(oc, d)`, discriminant `h^2 - a*c`) rather than the textbook
/// `b^2 - 4ac`: it is one fewer multiply, and avoids the factor-of-4 growth in
/// the discriminant that costs precision for large radii.
#[inline]
pub fn intersect_sphere(s: &GpuPrimitive, ray: &Ray, t_min: f32, t_max: f32) -> Option<f32> {
    let center = Vec3::from_array(s.position);
    let oc = ray.origin - center;
    let a = ray.dir.dot(ray.dir);
    let h = oc.dot(ray.dir);
    let c = oc.dot(oc) - s.radius * s.radius;
    let disc = h * h - a * c;
    if disc < 0.0 {
        return None;
    }
    let sqrt_d = disc.sqrt();
    // Near root first; fall back to the far root when the near one is behind
    // `t_min` (which is the case when the origin is inside the sphere).
    let mut t = (-h - sqrt_d) / a;
    if t < t_min || t > t_max {
        t = (-h + sqrt_d) / a;
        if t < t_min || t > t_max {
            return None;
        }
    }
    Some(t)
}

/// Ray/parallelogram intersection.
///
/// Two stages: hit the quad's infinite plane, then test whether the hit point
/// lies inside the unit square of the `(edge_u, edge_v)` parameterisation.
///
/// The inside-test uses the "reciprocal basis" trick. Given
/// `p - origin = a*edge_u + b*edge_v` (which holds exactly, since `p` is in the
/// plane), we want `a` and `b`. With `n_raw = cross(edge_u, edge_v)` and
/// `w = n_raw / dot(n_raw, n_raw)`:
///
/// ```text
///   a = dot(w, cross(d, edge_v))
///   b = dot(w, cross(edge_u, d))
/// ```
///
/// because `cross(edge_u, edge_v)` is orthogonal to both edges, so crossing `d`
/// against one edge annihilates that edge's contribution and leaves the other
/// coefficient scaled by `|n_raw|^2`. Dividing that out is exactly the `w`
/// factor. Two crosses and two dots, no matrix inverse, no divisions in the
/// inner test.
#[inline]
pub fn intersect_quad(q: &GpuPrimitive, ray: &Ray, t_min: f32, t_max: f32) -> Option<f32> {
    let origin = Vec3::from_array(q.position);
    let edge_u = Vec3::from_array(q.edge_u);
    let edge_v = Vec3::from_array(q.edge_v);
    let normal = Vec3::from_array(q.normal);

    let denom = normal.dot(ray.dir);
    // Reject rays parallel to the plane. The threshold is on the *normalised*
    // normal dotted with a *normalised* direction, so it is a true angular
    // threshold and independent of scene scale.
    if denom.abs() < 1.0e-8 {
        return None;
    }
    let t = (normal.dot(origin) - normal.dot(ray.origin)) / denom;
    if t < t_min || t > t_max {
        return None;
    }

    let p = ray.origin + t * ray.dir;
    let d = p - origin;
    let n_raw = edge_u.cross(edge_v);
    let w = n_raw / n_raw.dot(n_raw);
    let a = w.dot(d.cross(edge_v));
    let b = w.dot(edge_u.cross(d));
    if !(0.0..=1.0).contains(&a) || !(0.0..=1.0).contains(&b) {
        return None;
    }
    Some(t)
}

#[derive(Clone, Debug, Default)]
pub struct Scene {
    pub blob: SceneBlob,
    /// BVH over `blob.triangles`. Empty until [`Scene::build_bvh`] runs, in
    /// which case triangle intersection falls back to brute force — which is
    /// exactly what the equivalence tests want to compare against.
    pub bvh: crate::bvh::Bvh,
    /// First node of the TLAS within `blob.bvh_nodes`. Meaningless when there
    /// are no instances.
    pub tlas_root: u32,
    /// Instancing. Empty means the scene takes the single-level path.
    pub instances: crate::instance::InstanceSet,
    /// Environment lighting. Empty means the scene uses the constant
    /// `SceneDef::background` instead.
    ///
    /// Not part of `blob`: the blob is the byte-for-byte GPU upload, and the map
    /// goes to the GPU as *textures* rather than storage buffers. That is forced
    /// as well as preferred — the wavefront's SHADE stage already binds exactly
    /// eight storage buffers, which is WebGPU's guaranteed per-stage limit, so
    /// there is no room for another. Textures come from a separate budget.
    pub env: crate::envmap::EnvMap,
}

/// Which acceleration-structure builder to use.
///
/// Both are kept for the same reason both path-tracing architectures are: they
/// answer different questions. The binned-SAH builder produces the better tree
/// and is inherently sequential; the linear builder produces a tree about 25%
/// more expensive to traverse out of steps that are all maps and sorts, which is
/// what makes it a candidate for rebuilding geometry every frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BvhBuilder {
    #[default]
    BinnedSah,
    /// Morton codes, a sort, and Karras's hierarchy. See [`crate::lbvh`].
    Linear,
}

impl BvhBuilder {
    pub fn name(self) -> &'static str {
        match self {
            BvhBuilder::BinnedSah => "sah",
            BvhBuilder::Linear => "lbvh",
        }
    }

    pub fn parse(s: &str) -> Option<BvhBuilder> {
        match s {
            "sah" | "binned-sah" => Some(BvhBuilder::BinnedSah),
            "lbvh" | "linear" => Some(BvhBuilder::Linear),
            _ => None,
        }
    }
}

impl Scene {
    /// How many strategies the light sampler chooses between.
    ///
    /// The area lights plus the environment map, when there is one. Every
    /// light-sampling density is divided by this, and so is the density the
    /// BSDF side uses for its MIS weight — if the two disagree the weights stop
    /// summing to one and the image is uniformly wrong by a factor nobody can
    /// see.
    pub fn light_strategy_count(&self) -> u32 {
        self.blob.lights.len() as u32 + u32::from(!self.env.is_empty())
    }

    pub fn material(&self, index: u32) -> &GpuMaterial {
        &self.blob.materials[index as usize]
    }

    /// Prepare a scene for rendering: build the acceleration structure and
    /// collect the emitters.
    ///
    /// Must be called after the last geometry change. Light collection has to
    /// come *after* BVH compaction, because compaction permutes the triangle
    /// array and the light list stores flattened copies of emissive triangles.
    pub fn finalize(&mut self) {
        self.build_bvh();
        self.blob.lights = crate::light::build_lights(&self.blob);
    }

    /// Build the acceleration structure and copy it into the blob for upload.
    ///
    /// The triangle array is compacted into traversal order, so BVH leaves index
    /// it directly and no separate primitive-index buffer is needed — see
    /// [`crate::bvh::Bvh::compact_primitives`].
    pub fn build_bvh(&mut self) {
        self.build_bvh_with(BvhBuilder::BinnedSah);
    }

    /// Build a two-level hierarchy: one BLAS per mesh range, a TLAS over the
    /// placements.
    ///
    /// `mesh_ranges` are `(first_triangle, count)` slices of `blob.triangles`;
    /// `placements` are `(object_to_world, mesh, material_override)`.
    ///
    /// The per-BLAS trees are kept as ordinary [`crate::bvh::Bvh`] objects for
    /// the CPU, and *also* flattened into `blob.bvh_nodes` for upload, with
    /// child and leaf indices rebased. Two representations of one thing is a
    /// drift risk, which is why they are produced here together and why the
    /// CPU/GPU agreement test covers an instanced scene.
    pub fn build_instances(
        &mut self,
        mesh_ranges: &[(u32, u32)],
        placements: &[(glam::Mat4, u32, u32)],
    ) {
        use crate::gpu_layout::GpuInstance;
        use crate::instance::{Blas, Instance, InstanceSet};

        let mut set = InstanceSet::default();
        let mut flat_nodes: Vec<crate::gpu_layout::GpuBvhNode> = Vec::new();

        for &(first, count) in mesh_ranges {
            let (lo, hi) = (first as usize, (first + count) as usize);
            let mut tris = self.blob.triangles[lo..hi].to_vec();
            let mut bvh = crate::bvh::Bvh::build(&tris, &self.blob.positions);
            // Compaction permutes the slice, so the blob has to receive the
            // permuted order — the BLAS's leaves index it directly.
            bvh.compact_primitives(&mut tris);
            self.blob.triangles[lo..hi].copy_from_slice(&tris);

            let mut bounds = crate::bvh::Aabb::default();
            for t in &tris {
                bounds.grow(&crate::bvh::triangle_bounds(t, &self.blob.positions));
            }

            let node_offset = flat_nodes.len() as u32;
            // Rebase: child indices are node-relative and leaf `left_first` is
            // triangle-relative, and both become absolute in the shared arrays.
            for n in &bvh.nodes {
                let mut n = *n;
                if n.count == 0 {
                    n.left_first += node_offset;
                } else {
                    n.left_first += first;
                }
                flat_nodes.push(n);
            }

            set.blas.push(Blas {
                node_offset,
                node_count: bvh.nodes.len() as u32,
                triangle_offset: first,
                triangle_count: count,
                bounds,
                bvh,
            });
        }

        for &(to_world, mesh, material) in placements {
            set.instances.push(Instance::new(to_world, mesh, material));
        }

        // The TLAS is an ordinary BVH, built over each instance's world bounds
        // by presenting them as degenerate "triangles" — the builder only ever
        // asks for bounds and centroids, so this reuses the binned-SAH split
        // rather than duplicating it.
        let world_bounds: Vec<crate::bvh::Aabb> = set
            .instances
            .iter()
            .map(|i| i.world_bounds(&set.blas[i.blas as usize].bounds))
            .collect();
        set.tlas = crate::bvh::Bvh::build_over_bounds(&world_bounds);

        // Flatten the TLAS after the BLASes; its leaves index `instances`, which
        // the shader finds appended to `primitives`.
        let tlas_offset = flat_nodes.len() as u32;
        let prim_base = self.blob.primitives.len() as u32;
        for n in &set.tlas.nodes {
            let mut n = *n;
            if n.count == 0 {
                n.left_first += tlas_offset;
            } else {
                // Leaves hold a range of `tlas.prim_indices`, which the shader
                // does not have — so the instances are *reordered* into
                // traversal order and leaves index them directly.
                n.left_first += prim_base;
            }
            flat_nodes.push(n);
        }

        self.blob.instances = set
            .tlas
            .prim_indices
            .iter()
            .map(|&i| {
                let inst = &set.instances[i as usize];
                GpuInstance::new(inst, set.blas[inst.blas as usize].node_offset)
            })
            .collect();

        self.blob.bvh_nodes = flat_nodes;
        self.tlas_root = tlas_offset;
        self.instances = set;
    }

    /// As [`Scene::build_bvh`], choosing the builder.
    ///
    /// An acceleration structure is meant to be invisible: which tree is built
    /// must change how long a render takes and nothing else. Having both
    /// reachable from one call is what lets a test assert exactly that.
    pub fn build_bvh_with(&mut self, builder: BvhBuilder) {
        self.bvh = match builder {
            BvhBuilder::BinnedSah => {
                crate::bvh::Bvh::build(&self.blob.triangles, &self.blob.positions)
            }
            BvhBuilder::Linear => {
                crate::lbvh::Lbvh::build(&self.blob.triangles, &self.blob.positions).as_bvh()
            }
        };
        self.bvh.compact_primitives(&mut self.blob.triangles);
        debug_assert!(self.bvh.is_compacted());
        self.blob.bvh_nodes = self.bvh.nodes.clone();
        self.blob.bvh_prim_indices = self.bvh.prim_indices.clone();
    }

    /// As [`Scene::shade_triangle`], for a hit found inside an instance.
    ///
    /// Everything is computed in **object space** and only the normals are
    /// brought back, because that is all that needs bringing back: the hit
    /// position follows from the world-space ray and `t`, which the transform
    /// preserves. Recomputing the position from the object-space hit and the
    /// forward matrix would be equivalent and would need the forward matrix,
    /// which is deliberately not stored.
    fn shade_instanced_triangle(&self, hit: &crate::instance::InstanceHit, ray: &Ray) -> Hit {
        let inst = &self.instances.instances[hit.instance as usize];
        let (o, d) = inst.transform_ray(ray.origin, ray.dir);
        let local_ray = Ray { origin: o, dir: d };
        let mut h = self.shade_triangle(&hit.tri, &local_ray);

        h.normal = inst.transform_normal(h.normal);
        h.geometric_normal = inst.transform_normal(h.geometric_normal);
        // `t` is the same in both spaces, so the world position is just the
        // world ray evaluated there.
        h.position = ray.origin + hit.tri.t * ray.dir;
        // Front/back is decided in world space, after the normal has been taken
        // there: a mirroring transform (negative determinant) flips the winding,
        // and deciding in object space would then light the wrong face.
        h.front_face = h.geometric_normal.dot(ray.dir) < 0.0;
        let flip = if h.front_face { 1.0 } else { -1.0 };
        h.normal *= flip;
        h.geometric_normal *= flip;
        if inst.material_override != u32::MAX {
            h.material = inst.material_override;
        }
        // Instanced triangles are not in the light list — `build_lights`
        // flattens emissive geometry in world space and an instance's triangles
        // live in object space — so there is no light-sampling density for MIS
        // to balance against.
        h.light_area = 0.0;
        h
    }

    /// Interpolate vertex attributes at a triangle hit and fill in a [`Hit`].
    fn shade_triangle(&self, hit: &crate::bvh::TriHit, ray: &Ray) -> Hit {
        let tri = &self.blob.triangles[hit.triangle as usize];
        let (p0, p1, p2) = crate::bvh::tri_positions(tri, &self.blob.positions);

        // Barycentric weights. Möller–Trumbore returns (u, v) for vertices 1
        // and 2; vertex 0 gets the remainder.
        let w = 1.0 - hit.u - hit.v;
        let attr = |i: u32| Vec3::from_array(self.blob.vertex_attrs[i as usize].normal);
        let shading =
            (w * attr(tri.i0) + hit.u * attr(tri.i1) + hit.v * attr(tri.i2)).normalize_or(Vec3::Z);

        let cross = (p1 - p0).cross(p2 - p0);
        let geometric = cross.normalize_or(Vec3::Z);
        // Front/back is decided by the *geometric* normal: the interpolated one
        // can disagree near a silhouette, and letting it decide would flip
        // one-sided emission on and off across a smooth surface.
        let front_face = geometric.dot(ray.dir) < 0.0;
        let flip = if front_face { 1.0 } else { -1.0 };

        Hit {
            t: hit.t,
            steps: 0,
            position: ray.origin + hit.t * ray.dir,
            normal: shading * flip,
            geometric_normal: geometric * flip,
            front_face,
            material: tri.material,
            // Half the parallelogram spanned by the two edges.
            light_area: 0.5 * cross.length(),
        }
    }

    /// Closest-hit over every primitive.
    pub fn intersect(&self, ray: &Ray) -> Option<Hit> {
        let mut stats = crate::bvh::TraversalStats::default();
        let hit = self.intersect_counting(ray, &mut stats);
        hit.map(|mut h| {
            h.steps = stats.node_visits;
            h
        })
    }

    /// As [`Scene::intersect`], reporting how much traversal it took.
    ///
    /// Separate rather than always-on because the counter is only wanted by the
    /// heatmap, and threading a `&mut` through the hot path of the CPU tracer
    /// costs more than the shader's single increment does.
    pub fn intersect_counting(
        &self,
        ray: &Ray,
        stats: &mut crate::bvh::TraversalStats,
    ) -> Option<Hit> {
        let mut best_t = T_MAX;
        let mut best: Option<usize> = None;

        for (i, prim) in self.blob.primitives.iter().enumerate() {
            let t = if prim.is_quad() {
                intersect_quad(prim, ray, T_MIN, best_t)
            } else {
                intersect_sphere(prim, ray, T_MIN, best_t)
            };
            if let Some(t) = t {
                best_t = t;
                best = Some(i);
            }
        }

        // Instances, through the two-level hierarchy.
        //
        // And **instead of** the single-level triangle path, not before it.
        // `build_instances` assigns every triangle in the blob to some BLAS, so
        // those triangles live in their meshes' object spaces — traversing them
        // directly would render an extra, untransformed copy of every mesh at
        // the origin. That is exactly what happened here, and it was invisible
        // for the identity transform because the ghost coincided with the
        // instance.
        //
        // A scene with both instanced and free-standing triangles would need the
        // blob to record which range belongs to which, and nothing needs that
        // yet.
        //
        // Analytic primitives stay brute force either way: there are a handful
        // of them, and putting heterogeneous types in one BVH would mean a tag
        // in every leaf.
        if !self.instances.is_empty() {
            return match self.instances.intersect(
                &self.blob.triangles,
                &self.blob.positions,
                ray,
                T_MIN,
                best_t,
                stats,
            ) {
                Some(ih) => Some(self.shade_instanced_triangle(&ih, ray)),
                None => self.shade_analytic(best, best_t, ray),
            };
        }

        // Triangles, through the BVH when one has been built.
        let tri_hit = if self.bvh.is_empty() {
            crate::bvh::brute_force_intersect(
                &self.blob.triangles,
                &self.blob.positions,
                ray,
                T_MIN,
                best_t,
            )
        } else {
            self.bvh.intersect_with_stats(
                &self.blob.triangles,
                &self.blob.positions,
                ray,
                T_MIN,
                best_t,
                stats,
            )
        };
        if let Some(th) = tri_hit {
            return Some(self.shade_triangle(&th, ray));
        }

        self.shade_analytic(best, best_t, ray)
    }

    /// Fill in a [`Hit`] for the closest analytic primitive, if any.
    ///
    /// Factored out because both the single-level and the instanced paths end
    /// here — the analytic primitives are outside the hierarchy either way.
    fn shade_analytic(&self, best: Option<usize>, best_t: f32, ray: &Ray) -> Option<Hit> {
        let index = best?;
        let prim = &self.blob.primitives[index];
        let position = ray.origin + best_t * ray.dir;
        let light_area = if prim.is_quad() {
            Vec3::from_array(prim.edge_u)
                .cross(Vec3::from_array(prim.edge_v))
                .length()
        } else {
            // Spheres are not in the light list, so there is no light-sampling
            // density for the MIS weight to balance against.
            0.0
        };
        let geom_normal = if prim.is_quad() {
            Vec3::from_array(prim.normal)
        } else {
            // Divide by the stored radius rather than normalising: exact for a
            // point that is genuinely on the sphere, and cheaper.
            (position - Vec3::from_array(prim.position)) / prim.radius
        };
        let material = prim.material;

        let front_face = geom_normal.dot(ray.dir) < 0.0;
        let oriented = if front_face {
            geom_normal
        } else {
            -geom_normal
        };
        Some(Hit {
            t: best_t,
            steps: 0,
            position,
            // Analytic primitives have an exact normal, so the shading and
            // geometric normals coincide.
            normal: oriented,
            geometric_normal: oriented,
            front_face,
            material,
            light_area,
        })
    }

    /// Any-hit within `[T_MIN, t_max)`. Used by shadow rays from build step 8;
    /// present now so the CPU and GPU traversal code stay structurally paired.
    pub fn occluded(&self, ray: &Ray, t_max: f32) -> bool {
        for prim in &self.blob.primitives {
            let hit = if prim.is_quad() {
                intersect_quad(prim, ray, T_MIN, t_max)
            } else {
                intersect_sphere(prim, ray, T_MIN, t_max)
            };
            if hit.is_some() {
                return true;
            }
        }
        if self.bvh.is_empty() {
            crate::bvh::brute_force_intersect(
                &self.blob.triangles,
                &self.blob.positions,
                ray,
                T_MIN,
                t_max,
            )
            .is_some()
        } else {
            self.bvh.occluded(
                &self.blob.triangles,
                &self.blob.positions,
                ray,
                T_MIN,
                t_max,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_sphere() -> GpuPrimitive {
        GpuPrimitive::sphere(Vec3::ZERO, 1.0, 0)
    }

    #[test]
    fn sphere_hit_from_outside_takes_near_root() {
        let s = unit_sphere();
        let ray = Ray {
            origin: Vec3::new(0.0, 0.0, -5.0),
            dir: Vec3::Z,
        };
        let t = intersect_sphere(&s, &ray, T_MIN, T_MAX).expect("should hit");
        assert!((t - 4.0).abs() < 1e-5, "t = {t}");
    }

    #[test]
    fn sphere_hit_from_inside_takes_far_root() {
        let s = unit_sphere();
        let ray = Ray {
            origin: Vec3::ZERO,
            dir: Vec3::Z,
        };
        let t = intersect_sphere(&s, &ray, T_MIN, T_MAX).expect("should hit");
        assert!((t - 1.0).abs() < 1e-5, "t = {t}");
    }

    #[test]
    fn sphere_miss_behind_ray() {
        let s = unit_sphere();
        let ray = Ray {
            origin: Vec3::new(0.0, 0.0, 5.0),
            dir: Vec3::Z,
        };
        assert!(intersect_sphere(&s, &ray, T_MIN, T_MAX).is_none());
    }

    fn unit_quad() -> GpuPrimitive {
        // Unit square in the z = 0 plane spanning [0,1] x [0,1], facing +Z.
        GpuPrimitive::quad(Vec3::ZERO, Vec3::X, Vec3::Y, 0)
    }

    #[test]
    fn quad_inside_and_outside() {
        let q = unit_quad();
        // Straight through the middle.
        let hit = Ray {
            origin: Vec3::new(0.5, 0.5, -1.0),
            dir: Vec3::Z,
        };
        let t = intersect_quad(&q, &hit, T_MIN, T_MAX).expect("centre should hit");
        assert!((t - 1.0).abs() < 1e-5);

        // Just outside each edge of the parameter square.
        for p in [
            Vec3::new(-0.01, 0.5, -1.0),
            Vec3::new(1.01, 0.5, -1.0),
            Vec3::new(0.5, -0.01, -1.0),
            Vec3::new(0.5, 1.01, -1.0),
        ] {
            let r = Ray {
                origin: p,
                dir: Vec3::Z,
            };
            assert!(
                intersect_quad(&q, &r, T_MIN, T_MAX).is_none(),
                "should miss at {p}"
            );
        }
    }

    #[test]
    fn quad_parallel_ray_misses() {
        let q = unit_quad();
        let r = Ray {
            origin: Vec3::new(0.5, 0.5, -1.0),
            dir: Vec3::X,
        };
        assert!(intersect_quad(&q, &r, T_MIN, T_MAX).is_none());
    }

    /// Closest-hit must actually pick the closest primitive regardless of the
    /// order they appear in the arrays.
    #[test]
    fn closest_hit_wins() {
        let near = GpuPrimitive::sphere(Vec3::ZERO, 1.0, 1);
        let far = GpuPrimitive::sphere(Vec3::new(0.0, 0.0, 10.0), 1.0, 2);
        for (a, b, expect) in [(near, far, 1u32), (far, near, 1u32)] {
            let scene = Scene {
                blob: SceneBlob {
                    materials: vec![GpuMaterial::default(); 3],
                    primitives: vec![a, b],
                    ..Default::default()
                },
                ..Default::default()
            };
            let hit = scene
                .intersect(&Ray {
                    origin: Vec3::new(0.0, 0.0, -5.0),
                    dir: Vec3::Z,
                })
                .expect("should hit");
            assert_eq!(hit.material, expect);
            assert!((hit.t - 4.0).abs() < 1e-5);
        }
    }

    #[test]
    fn normal_faces_the_ray() {
        let scene = Scene {
            blob: SceneBlob {
                materials: vec![GpuMaterial::default()],
                primitives: vec![unit_sphere()],
                ..Default::default()
            },
            ..Default::default()
        };
        // From outside: front face, normal points back at us.
        let h = scene
            .intersect(&Ray {
                origin: Vec3::new(0.0, 0.0, -5.0),
                dir: Vec3::Z,
            })
            .unwrap();
        assert!(h.front_face);
        assert!(h.normal.dot(Vec3::Z) < 0.0);

        // From inside: back face, normal flipped to still oppose the ray.
        let h = scene
            .intersect(&Ray {
                origin: Vec3::ZERO,
                dir: Vec3::Z,
            })
            .unwrap();
        assert!(!h.front_face);
        assert!(h.normal.dot(Vec3::Z) < 0.0);
    }
}
