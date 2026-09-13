//! CPU bounding volume hierarchy: binned-SAH build and iterative traversal.
//!
//! This is the **quality reference**. The GPU LBVH at build step 11 is the fast
//! builder; this one exists to say how good a BVH could have been, and to be the
//! thing the GPU traversal is validated against.
//!
//! Two properties are non-negotiable and both are tested:
//!
//!   * traversal returns *exactly* what brute force over every triangle returns;
//!   * the structure is sound — every child box inside its parent, every
//!     primitive referenced exactly once.
//!
//! A BVH that is merely *fast* and slightly wrong produces images with
//! occasional missing geometry that look like noise, and chasing that is
//! miserable. Hence the invariants.

use crate::gpu_layout::{GpuBvhNode, GpuTriangle};
use crate::scene::Ray;
use glam::Vec3;

/// Number of bins per axis for the SAH sweep.
///
/// Full SAH evaluates every possible split (sorting by centroid on each axis,
/// O(n log n) per node); binning approximates it with a fixed number of
/// candidate planes, which is O(n) per node and gives up a few percent of
/// quality. 12 is the usual sweet spot — going to 32 measurably improves nothing
/// on real geometry while making the build slower.
const BINS: usize = 12;

/// Relative cost of one traversal step versus one triangle test.
///
/// These are the only tuning knobs in the SAH. Raising `TRAVERSAL_COST` biases
/// toward fatter leaves (fewer nodes, more triangle tests); lowering it produces
/// a deeper tree. 1:1 is the classic default and is close to right for a GPU,
/// where a traversal step and a Möller–Trumbore test are comparable work.
const TRAVERSAL_COST: f32 = 1.0;
const INTERSECT_COST: f32 = 1.0;

/// Stop subdividing at or below this many primitives regardless of what the SAH
/// says. Guards against pathological splits that separate one primitive at a
/// time and build a linked list.
const MIN_LEAF: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Default for Aabb {
    /// The empty box: inverted, so it is the identity for union.
    fn default() -> Self {
        Self {
            min: Vec3::splat(f32::INFINITY),
            max: Vec3::splat(f32::NEG_INFINITY),
        }
    }
}

impl Aabb {
    #[inline]
    pub fn grow_point(&mut self, p: Vec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    #[inline]
    pub fn grow(&mut self, other: &Aabb) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.min.x > self.max.x
    }

    /// Surface area, the "A" in the surface area heuristic.
    ///
    /// The SAH's whole premise: for a ray with uniformly distributed origin and
    /// direction, the probability of hitting a convex box *given* that it hits
    /// an enclosing box is the ratio of their surface areas. So expected cost is
    /// area-weighted, not volume-weighted.
    #[inline]
    pub fn surface_area(&self) -> f32 {
        if self.is_empty() {
            return 0.0;
        }
        let d = self.max - self.min;
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    #[inline]
    pub fn centroid(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Expand outward by a magnitude-relative epsilon. See [`triangle_bounds`].
    #[inline]
    pub fn pad(&mut self) {
        if self.is_empty() {
            return;
        }
        const RELATIVE: f32 = 1.0e-6;
        for a in 0..3 {
            let scale = 1.0 + self.min[a].abs().max(self.max[a].abs());
            let p = RELATIVE * scale;
            self.min[a] -= p;
            self.max[a] += p;
        }
    }

    /// Does `other` fit entirely inside this box?
    #[inline]
    pub fn contains(&self, other: &Aabb, epsilon: f32) -> bool {
        other.is_empty()
            || (self.min.cmple(other.min + Vec3::splat(epsilon)).all()
                && self.max.cmpge(other.max - Vec3::splat(epsilon)).all())
    }

    /// Slab test. Returns the entry distance if the ray overlaps the box within
    /// `[t_min, t_max]`, otherwise `None`.
    ///
    /// `inv_dir` must come from [`safe_inv_dir`] — see the warning there. A ray
    /// tests thousands of boxes, so the reciprocal is computed once per ray.
    #[inline]
    pub fn hit(&self, origin: Vec3, inv_dir: Vec3, t_min: f32, t_max: f32) -> Option<f32> {
        let t0 = (self.min - origin) * inv_dir;
        let t1 = (self.max - origin) * inv_dir;
        let near = t0.min(t1);
        let far = t0.max(t1);
        let entry = near.max_element().max(t_min);
        let exit = far.min_element().min(t_max);
        if entry <= exit {
            Some(entry)
        } else {
            None
        }
    }
}

/// Reciprocal of a ray direction, guaranteed finite.
///
/// # Why not just `1.0 / dir`
///
/// Two problems, and the second is the nasty one.
///
/// 1. WGSL leaves floating-point division by zero **implementation-defined**.
///    An axis-aligned ray is not exotic — it is what a camera looking down an
///    axis produces — so relying on it yielding infinity is relying on luck.
///
/// 2. Even where it does yield infinity, the slab test then computes
///    `(plane - origin) * inf`. When the origin lies *exactly* on that slab
///    plane the term is `0 * inf = NaN`, and a NaN in the comparison silently
///    turns a hit into a miss. This is not a rare case: every triangle of an
///    axis-aligned surface has a **flat** AABB, and the Cornell box is made
///    entirely of axis-aligned surfaces.
///
/// Clamping the reciprocal to a large *finite* magnitude removes both problems
/// branchlessly. `0 * 1e16` is 0, not NaN, and no infinity ever enters the
/// arithmetic. The cost is that a ray within 1e-16 of parallel to an axis gets
/// a slightly conservative slab interval — which can only ever produce a false
/// *positive*, and those are rejected by the triangle test a moment later.
#[inline]
pub fn safe_inv_dir(dir: Vec3) -> Vec3 {
    const BIG: f32 = 1.0e16;
    const SMALL: f32 = 1.0 / BIG;
    #[inline]
    fn guard(d: f32) -> f32 {
        if d.abs() > SMALL {
            1.0 / d
        } else {
            // copysign keeps the slab ordering correct for -0.0.
            BIG.copysign(d)
        }
    }
    Vec3::new(guard(dir.x), guard(dir.y), guard(dir.z))
}

/// Bounds of one triangle, padded outward by a hair.
///
/// # Why pad
///
/// An axis-aligned triangle — a Cornell box wall, a floor, anything
/// architectural — produces a **flat** AABB with zero extent on one axis. A ray
/// travelling within that plane then enters and exits the slab at the same
/// instant, and floating-point rounding decides arbitrarily whether that counts
/// as an overlap. The failure mode is a *false negative*: the traversal skips a
/// node whose triangles a brute-force test would have hit, and geometry goes
/// intermittently missing in a way that reads as noise.
///
/// Padding makes every box conservative. A false positive costs one wasted
/// triangle test and is corrected immediately; a false negative costs a
/// debugging afternoon. The padding is relative to coordinate magnitude, for the
/// same reason ray offsets are (see [`crate::math::offset_ray_origin`]): float
/// spacing scales with magnitude, so a fixed epsilon is simultaneously too large
/// near the origin and too small far from it.
///
/// At 1e-6 relative this is far below anything the SAH can notice — at Cornell
/// box scale it is half a thousandth of a unit.
#[inline]
pub fn triangle_bounds(tri: &GpuTriangle, positions: &[[f32; 4]]) -> Aabb {
    let mut b = Aabb::default();
    for i in [tri.i0, tri.i1, tri.i2] {
        let p = positions[i as usize];
        b.grow_point(Vec3::new(p[0], p[1], p[2]));
    }
    b.pad();
    b
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BuildStats {
    pub triangles: usize,
    pub nodes: usize,
    pub leaves: usize,
    pub max_depth: usize,
    pub max_leaf_size: usize,
    pub mean_leaf_size: f32,
    pub build_seconds: f64,
    /// Expected traversal cost per ray under the SAH model, in units of one
    /// triangle test. The single number that says how good the tree is.
    pub sah_cost: f32,
}

#[derive(Clone, Debug, Default)]
pub struct Bvh {
    pub nodes: Vec<GpuBvhNode>,
    /// Permutation of triangle indices. Leaves address ranges of this, so the
    /// build never moves triangle data — only 4-byte indices.
    pub prim_indices: Vec<u32>,
    pub stats: BuildStats,
}

/// One primitive's precomputed bounds and centroid.
struct PrimInfo {
    bounds: Aabb,
    centroid: Vec3,
}

#[derive(Clone, Copy, Default)]
struct Bin {
    bounds: Aabb,
    count: u32,
}

impl Bvh {
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Build a binned-SAH BVH over `triangles`.
    pub fn build(triangles: &[GpuTriangle], positions: &[[f32; 4]]) -> Bvh {
        let bounds: Vec<Aabb> = triangles
            .iter()
            .map(|t| triangle_bounds(t, positions))
            .collect();
        Bvh::build_over_bounds(&bounds)
    }

    /// Build over arbitrary boxes.
    ///
    /// The split heuristic only ever asks a primitive for its bounds and its
    /// centroid, so nothing about it is triangle-specific. Exposing that is what
    /// lets the **TLAS** — a hierarchy over instance world bounds — reuse the
    /// binned-SAH builder rather than carry a second copy of it.
    pub fn build_over_bounds(bounds: &[Aabb]) -> Bvh {
        let start = std::time::Instant::now();
        if bounds.is_empty() {
            return Bvh::default();
        }
        let triangles_len = bounds.len();

        let prims: Vec<PrimInfo> = bounds
            .iter()
            .map(|b| PrimInfo {
                centroid: b.centroid(),
                bounds: *b,
            })
            .collect();

        let mut prim_indices: Vec<u32> = (0..triangles_len as u32).collect();
        // Upper bound for a binary tree with at least MIN_LEAF prims per leaf.
        let mut nodes: Vec<GpuBvhNode> = Vec::with_capacity(2 * triangles_len);
        nodes.push(GpuBvhNode::default());

        // Explicit work stack rather than recursion: a pathological scene can
        // drive the tree deep enough to overflow the call stack, and this also
        // makes the build trivially convertible to a parallel one later.
        let mut stack: Vec<(usize, usize, usize, usize)> = vec![(0, 0, triangles_len, 0)];
        let mut stats = BuildStats {
            triangles: triangles_len,
            ..Default::default()
        };
        let mut leaf_prims = 0usize;

        while let Some((node_idx, start_i, count, depth)) = stack.pop() {
            stats.max_depth = stats.max_depth.max(depth);

            let mut bounds = Aabb::default();
            let mut centroid_bounds = Aabb::default();
            for &pi in &prim_indices[start_i..start_i + count] {
                bounds.grow(&prims[pi as usize].bounds);
                centroid_bounds.grow_point(prims[pi as usize].centroid);
            }
            nodes[node_idx].bounds_min = bounds.min.to_array();
            nodes[node_idx].bounds_max = bounds.max.to_array();

            let make_leaf = |nodes: &mut Vec<GpuBvhNode>| {
                nodes[node_idx].left_first = start_i as u32;
                nodes[node_idx].count = count as u32;
            };

            if count <= MIN_LEAF {
                make_leaf(&mut nodes);
                stats.leaves += 1;
                leaf_prims += count;
                stats.max_leaf_size = stats.max_leaf_size.max(count);
                continue;
            }

            let extent = centroid_bounds.max - centroid_bounds.min;
            let axis = if extent.x > extent.y && extent.x > extent.z {
                0
            } else if extent.y > extent.z {
                1
            } else {
                2
            };
            // All centroids coincident: no split can separate them, and trying
            // would recurse forever on the same set.
            if extent[axis] <= 0.0 {
                make_leaf(&mut nodes);
                stats.leaves += 1;
                leaf_prims += count;
                stats.max_leaf_size = stats.max_leaf_size.max(count);
                continue;
            }

            // --- binned SAH sweep on the widest axis -------------------------
            //
            // Bin by centroid, then sweep once forward accumulating the left
            // bounds and once backward for the right, so evaluating all BINS-1
            // candidate planes costs O(n + BINS) rather than O(n * BINS).
            let scale = BINS as f32 / extent[axis];
            let mut bins = [Bin::default(); BINS];
            for &pi in &prim_indices[start_i..start_i + count] {
                let p = &prims[pi as usize];
                let b = bin_index(p.centroid[axis], centroid_bounds.min[axis], scale);
                bins[b].count += 1;
                bins[b].bounds.grow(&p.bounds);
            }

            let mut left_area = [0.0f32; BINS - 1];
            let mut left_count = [0u32; BINS - 1];
            let mut acc = Aabb::default();
            let mut n = 0u32;
            for i in 0..BINS - 1 {
                acc.grow(&bins[i].bounds);
                n += bins[i].count;
                left_area[i] = acc.surface_area();
                left_count[i] = n;
            }

            let mut right_area = [0.0f32; BINS - 1];
            let mut right_count = [0u32; BINS - 1];
            acc = Aabb::default();
            n = 0;
            for i in (1..BINS).rev() {
                acc.grow(&bins[i].bounds);
                n += bins[i].count;
                right_area[i - 1] = acc.surface_area();
                right_count[i - 1] = n;
            }

            let parent_area = bounds.surface_area();
            let mut best_cost = f32::INFINITY;
            let mut best_split = usize::MAX;
            for i in 0..BINS - 1 {
                if left_count[i] == 0 || right_count[i] == 0 {
                    continue;
                }
                // The surface area heuristic:
                //
                //   C = C_trav + (A_L/A_P) * N_L * C_isect + (A_R/A_P) * N_R * C_isect
                //
                // A_L/A_P is the conditional probability that a ray already
                // inside the parent box also enters the left child. So the two
                // terms are the *expected* number of triangle tests, and
                // C_trav is what the extra node visit costs. Minimising this
                // minimises expected work per ray, which is why SAH beats
                // splitting at the spatial median.
                let cost = TRAVERSAL_COST
                    + (left_area[i] * left_count[i] as f32 + right_area[i] * right_count[i] as f32)
                        / parent_area
                        * INTERSECT_COST;
                if cost < best_cost {
                    best_cost = cost;
                    best_split = i;
                }
            }

            // Splitting has to actually beat just testing everything here.
            let leaf_cost = count as f32 * INTERSECT_COST;
            if best_split == usize::MAX || best_cost >= leaf_cost {
                make_leaf(&mut nodes);
                stats.leaves += 1;
                leaf_prims += count;
                stats.max_leaf_size = stats.max_leaf_size.max(count);
                continue;
            }

            // Partition in place (Hoare-style) by bin index.
            let split_bin = best_split + 1;
            let mut i = start_i;
            let mut j = start_i + count;
            while i < j {
                let pi = prim_indices[i] as usize;
                let b = bin_index(prims[pi].centroid[axis], centroid_bounds.min[axis], scale);
                if b < split_bin {
                    i += 1;
                } else {
                    j -= 1;
                    prim_indices.swap(i, j);
                }
            }
            let left_n = i - start_i;
            // The binning and the partition use the same predicate, so this
            // cannot happen — but if a future change makes them disagree, an
            // empty side would produce infinite recursion rather than a wrong
            // image, so it is worth catching here.
            debug_assert!(
                left_n > 0 && left_n < count,
                "SAH partition produced an empty side"
            );

            // Children are allocated adjacently so one index addresses both.
            let left_idx = nodes.len();
            nodes.push(GpuBvhNode::default());
            nodes.push(GpuBvhNode::default());
            nodes[node_idx].left_first = left_idx as u32;
            nodes[node_idx].count = 0;

            stack.push((left_idx, start_i, left_n, depth + 1));
            stack.push((left_idx + 1, start_i + left_n, count - left_n, depth + 1));
        }

        stats.nodes = nodes.len();
        stats.mean_leaf_size = leaf_prims as f32 / stats.leaves.max(1) as f32;
        stats.build_seconds = start.elapsed().as_secs_f64();

        let mut bvh = Bvh {
            nodes,
            prim_indices,
            stats,
        };
        bvh.stats.sah_cost = bvh.sah_cost();
        bvh
    }

    /// Expected traversal cost per ray under the SAH model, in units of one
    /// triangle test.
    ///
    /// Sum over nodes of (node area / root area) times the node's own cost —
    /// `TRAVERSAL_COST` for an internal node, `count * INTERSECT_COST` for a
    /// leaf. This is the number to compare builders by: it is what the SAH is
    /// trying to minimise, and unlike a frame time it does not depend on the
    /// machine, the camera, or the resolution.
    pub fn sah_cost(&self) -> f32 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        let root_area = self.node_bounds(0).surface_area();
        if root_area <= 0.0 {
            return 0.0;
        }
        let mut total = 0.0;
        for node in &self.nodes {
            let area = Aabb {
                min: Vec3::from_array(node.bounds_min),
                max: Vec3::from_array(node.bounds_max),
            }
            .surface_area();
            let own = if node.count == 0 {
                TRAVERSAL_COST
            } else {
                node.count as f32 * INTERSECT_COST
            };
            total += area / root_area * own;
        }
        total
    }

    pub fn node_bounds(&self, i: usize) -> Aabb {
        Aabb {
            min: Vec3::from_array(self.nodes[i].bounds_min),
            max: Vec3::from_array(self.nodes[i].bounds_max),
        }
    }

    /// Reorder `triangles` into traversal order so leaves index the array
    /// **directly**, and reduce `prim_indices` to the identity.
    ///
    /// Two payoffs, one of which is not obvious:
    ///
    /// * The indirection through `prim_indices` disappears. On the GPU that is
    ///   one fewer storage buffer binding — which matters concretely, because
    ///   WebGPU guarantees only 8 per shader stage and this renderer needs
    ///   exactly 8 without it.
    /// * The triangles of a leaf become **contiguous in memory**. A leaf holding
    ///   two triangles now touches one cache line instead of two random ones,
    ///   and that is a real traversal win independent of the binding count.
    ///
    /// Safe to call once, after building. The CPU traversal keeps going through
    /// `prim_indices` so it works either way, which is what lets the
    /// equivalence tests run against an uncompacted tree.
    pub fn compact_primitives(&mut self, triangles: &mut [GpuTriangle]) {
        debug_assert_eq!(self.prim_indices.len(), triangles.len());
        let reordered: Vec<GpuTriangle> = self
            .prim_indices
            .iter()
            .map(|&i| triangles[i as usize])
            .collect();
        triangles.copy_from_slice(&reordered);
        for (i, p) in self.prim_indices.iter_mut().enumerate() {
            *p = i as u32;
        }
    }

    /// True when leaves index the triangle array directly (see
    /// [`Bvh::compact_primitives`]). The shader relies on this.
    pub fn is_compacted(&self) -> bool {
        self.prim_indices
            .iter()
            .enumerate()
            .all(|(i, &p)| p as usize == i)
    }

    /// Structural invariants. Returns a list of problems; empty means sound.
    ///
    /// Checked rather than assumed because the failure mode of a subtly broken
    /// BVH is *intermittently missing geometry*, which reads as noise and is
    /// exceptionally hard to attribute.
    pub fn validate(&self, triangle_count: usize) -> Vec<String> {
        let mut problems = Vec::new();
        if self.nodes.is_empty() {
            if triangle_count > 0 {
                problems.push(format!("{triangle_count} triangles but no nodes"));
            }
            return problems;
        }

        // Every primitive referenced exactly once.
        let mut seen = vec![0u32; triangle_count];
        for (i, node) in self.nodes.iter().enumerate() {
            if node.count == 0 {
                let l = node.left_first as usize;
                if l + 1 >= self.nodes.len() {
                    problems.push(format!("node {i} has out-of-range children at {l}"));
                    continue;
                }
                // Child boxes must be inside the parent's. A tolerance is needed
                // because the parent box is computed from the same floats but
                // accumulated in a different order.
                let parent = self.node_bounds(i);
                let eps = 1e-3 * parent.surface_area().sqrt().max(1.0);
                for c in [l, l + 1] {
                    if !parent.contains(&self.node_bounds(c), eps) {
                        problems.push(format!(
                            "node {i} bounds {:?}..{:?} do not contain child {c} {:?}..{:?}",
                            parent.min,
                            parent.max,
                            self.node_bounds(c).min,
                            self.node_bounds(c).max
                        ));
                    }
                }
            } else {
                let start = node.left_first as usize;
                let end = start + node.count as usize;
                if end > self.prim_indices.len() {
                    problems.push(format!("leaf {i} range {start}..{end} is out of bounds"));
                    continue;
                }
                for &pi in &self.prim_indices[start..end] {
                    if (pi as usize) >= triangle_count {
                        problems.push(format!("leaf {i} references triangle {pi}, out of range"));
                    } else {
                        seen[pi as usize] += 1;
                    }
                }
            }
        }

        let missing = seen.iter().filter(|&&c| c == 0).count();
        let duplicated = seen.iter().filter(|&&c| c > 1).count();
        if missing > 0 {
            problems.push(format!("{missing} triangles are in no leaf"));
        }
        if duplicated > 0 {
            problems.push(format!(
                "{duplicated} triangles appear in more than one leaf"
            ));
        }
        problems
    }
}

#[inline]
fn bin_index(centroid_coord: f32, axis_min: f32, scale: f32) -> usize {
    // Clamp: the highest centroid maps exactly to BINS and must fold into the
    // last bin, and f32 rounding can push a value marginally past the end.
    (((centroid_coord - axis_min) * scale) as usize).min(BINS - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_aabb_has_no_area_and_absorbs_anything() {
        let mut b = Aabb::default();
        assert!(b.is_empty());
        assert_eq!(b.surface_area(), 0.0);
        b.grow_point(Vec3::ZERO);
        b.grow_point(Vec3::ONE);
        assert!((b.surface_area() - 6.0).abs() < 1e-6);
        assert_eq!(b.centroid(), Vec3::splat(0.5));
    }

    #[test]
    fn slab_test_basic_cases() {
        let b = Aabb {
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
        };
        let inv = safe_inv_dir;

        // Straight through the middle, from outside.
        let t = b.hit(Vec3::new(0.0, 0.0, -5.0), inv(Vec3::Z), 0.0, f32::MAX);
        assert!((t.unwrap() - 4.0).abs() < 1e-5);

        // Starting inside: entry distance is clamped to t_min.
        assert_eq!(b.hit(Vec3::ZERO, inv(Vec3::Z), 0.0, f32::MAX), Some(0.0));

        // Pointing away.
        assert!(b
            .hit(Vec3::new(0.0, 0.0, -5.0), inv(-Vec3::Z), 0.0, f32::MAX)
            .is_none());

        // Missing to the side.
        assert!(b
            .hit(Vec3::new(5.0, 0.0, -5.0), inv(Vec3::Z), 0.0, f32::MAX)
            .is_none());

        // Beyond t_max.
        assert!(b
            .hit(Vec3::new(0.0, 0.0, -5.0), inv(Vec3::Z), 0.0, 1.0)
            .is_none());
    }

    /// A ray lying exactly in a face plane is the case that produces `0 * inf`
    /// with a naive reciprocal. It must be reported as a hit.
    #[test]
    fn slab_test_survives_axis_aligned_degenerate_rays() {
        let b = Aabb {
            min: Vec3::ZERO,
            max: Vec3::ONE,
        };

        // Origin exactly on the y = 0 face, travelling along +x within it.
        // Without the guarded reciprocal this computes 0 * inf = NaN.
        let hit = b.hit(
            Vec3::new(-1.0, 0.0, 0.5),
            safe_inv_dir(Vec3::X),
            0.0,
            f32::MAX,
        );
        assert!(
            hit.is_some(),
            "ray lying in the y = 0 face plane was rejected"
        );

        // The case that actually matters: a *flat* box, which is what every
        // axis-aligned triangle produces, with a ray travelling inside its
        // plane. Padded bounds make this a hit.
        let mut flat = Aabb {
            min: Vec3::ZERO,
            max: Vec3::new(1.0, 0.0, 1.0),
        };
        flat.pad();
        let hit = flat.hit(
            Vec3::new(-1.0, 0.0, 0.5),
            safe_inv_dir(Vec3::X),
            0.0,
            f32::MAX,
        );
        assert!(hit.is_some(), "ray inside a flat box's plane was rejected");

        // A flat box at a large coordinate, where a fixed epsilon would be far
        // too small to clear float spacing.
        let mut far = Aabb {
            min: Vec3::new(0.0, 5000.0, 0.0),
            max: Vec3::new(1.0, 5000.0, 1.0),
        };
        far.pad();
        let hit = far.hit(
            Vec3::new(-1.0, 5000.0, 0.5),
            safe_inv_dir(Vec3::X),
            0.0,
            f32::MAX,
        );
        assert!(
            hit.is_some(),
            "ray inside a distant flat box's plane was rejected"
        );

        // ...but a parallel ray genuinely outside the slab must still miss.
        // Padding must not be so generous that it swallows real separation.
        let miss = b.hit(
            Vec3::new(-1.0, 2.0, 0.5),
            safe_inv_dir(Vec3::X),
            0.0,
            f32::MAX,
        );
        assert!(
            miss.is_none(),
            "ray parallel to and outside the box reported a hit"
        );

        // Nothing above may produce a NaN, whatever the verdict.
        for origin in [
            Vec3::new(-1.0, 0.0, 0.5),
            Vec3::new(-1.0, 1.0, 0.5),
            Vec3::ZERO,
        ] {
            for dir in [Vec3::X, Vec3::Y, Vec3::Z, Vec3::new(1.0, 0.0, 0.0)] {
                if let Some(t) = b.hit(origin, safe_inv_dir(dir), 0.0, f32::MAX) {
                    assert!(t.is_finite() && !t.is_nan(), "slab test produced {t}");
                }
            }
        }
    }

    /// Padding must be conservative — strictly outward — and negligible.
    #[test]
    fn padding_only_grows_and_stays_tiny() {
        for (min, max) in [
            (Vec3::ZERO, Vec3::ONE),
            (Vec3::splat(-555.0), Vec3::splat(555.0)),
            (Vec3::new(0.0, 3.0, 0.0), Vec3::new(1.0, 3.0, 1.0)), // flat
        ] {
            let mut b = Aabb { min, max };
            let before = b;
            b.pad();
            assert!(b.min.cmple(before.min).all(), "padding moved min inward");
            assert!(b.max.cmpge(before.max).all(), "padding moved max inward");
            let grew = (b.max - b.min) - (before.max - before.min);
            let scale = min.abs().max(max.abs()).max_element().max(1.0);
            assert!(
                grew.max_element() < 1e-5 * scale,
                "padding grew the box by {grew} at scale {scale}"
            );
            // A flat axis must end up genuinely non-degenerate.
            assert!(
                b.max.cmpgt(b.min).all(),
                "padded box still has a zero-extent axis"
            );
        }
    }

    /// The guarded reciprocal must stay finite for every direction, including
    /// zero components and denormals.
    #[test]
    fn safe_inv_dir_is_always_finite() {
        for d in [
            Vec3::X,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(-0.0, 1.0, 0.0),
            Vec3::new(1e-30, 1.0, -1e-30),
            Vec3::new(f32::MIN_POSITIVE, -f32::MIN_POSITIVE, 1.0),
        ] {
            let inv = safe_inv_dir(d);
            assert!(inv.is_finite(), "safe_inv_dir({d}) = {inv}");
            // Sign must be preserved, including through negative zero, or the
            // slab ordering flips.
            for i in 0..3 {
                if d[i] != 0.0 {
                    assert_eq!(inv[i].signum(), d[i].signum(), "sign lost on axis {i}");
                }
            }
        }
    }

    fn test_mesh(levels: u32) -> (Vec<GpuTriangle>, Vec<[f32; 4]>) {
        let m = crate::mesh::icosphere(Vec3::ZERO, 1.0, levels);
        let mut blob = crate::gpu_layout::SceneBlob::default();
        m.append_to(&mut blob, 0);
        (blob.triangles, blob.positions)
    }

    #[test]
    fn build_is_structurally_sound() {
        for levels in 0..5 {
            let (tris, pos) = test_mesh(levels);
            let bvh = Bvh::build(&tris, &pos);
            let problems = bvh.validate(tris.len());
            assert!(problems.is_empty(), "level {levels}: {problems:#?}");
            assert_eq!(bvh.prim_indices.len(), tris.len());
        }
    }

    #[test]
    fn root_bounds_enclose_every_vertex() {
        let (tris, pos) = test_mesh(3);
        let bvh = Bvh::build(&tris, &pos);
        let root = bvh.node_bounds(0);
        for p in &pos {
            let v = Vec3::new(p[0], p[1], p[2]);
            assert!(
                v.cmpge(root.min - Vec3::splat(1e-5)).all()
                    && v.cmple(root.max + Vec3::splat(1e-5)).all(),
                "vertex {v} outside root bounds {:?}..{:?}",
                root.min,
                root.max
            );
        }
    }

    #[test]
    fn empty_input_produces_an_empty_bvh() {
        let bvh = Bvh::build(&[], &[]);
        assert!(bvh.is_empty());
        assert!(bvh.validate(0).is_empty());
    }

    /// A degenerate mesh where every triangle shares one centroid must still
    /// terminate — the centroid-extent guard is what stops it recursing forever.
    #[test]
    fn coincident_centroids_terminate() {
        let positions = vec![
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ];
        let tris = vec![
            GpuTriangle {
                i0: 0,
                i1: 1,
                i2: 2,
                material: 0
            };
            64
        ];
        let bvh = Bvh::build(&tris, &positions);
        assert!(bvh.validate(tris.len()).is_empty());
        // All identical, so there is nothing to separate: one leaf.
        assert_eq!(bvh.stats.leaves, 1);
    }

    /// SAH cost must beat the trivial alternative of testing every triangle,
    /// and it must scale sub-linearly. This is the headline quality number.
    #[test]
    fn sah_cost_is_far_below_brute_force() {
        for levels in 2..6 {
            let (tris, pos) = test_mesh(levels);
            let bvh = Bvh::build(&tris, &pos);
            let brute = tris.len() as f32;
            assert!(
                bvh.stats.sah_cost < brute * 0.1,
                "level {levels}: SAH cost {:.1} vs brute force {brute}",
                bvh.stats.sah_cost
            );
        }
    }

    /// Depth must stay logarithmic. A linear-depth tree still renders correctly
    /// but would overflow the fixed traversal stack in the shader.
    #[test]
    fn depth_stays_logarithmic() {
        let (tris, pos) = test_mesh(5);
        let bvh = Bvh::build(&tris, &pos);
        let ideal = (tris.len() as f32).log2();
        assert!(
            (bvh.stats.max_depth as f32) < ideal * 3.0,
            "{} triangles reached depth {}, ideal is about {ideal:.0}",
            tris.len(),
            bvh.stats.max_depth
        );
    }
}

// ---------------------------------------------------------------------------
// Triangle intersection and traversal
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct TriHit {
    pub t: f32,
    /// Barycentric coordinates of the hit, for interpolating vertex attributes.
    /// The third weight is `1 - u - v`, associated with vertex 0.
    pub u: f32,
    pub v: f32,
    pub triangle: u32,
}

/// Counters for the BVH traversal heatmap and for comparing builders.
///
/// Node visits per ray is the number that actually says whether a BVH is any
/// good — far more diagnostic than a frame time, which mixes in shading,
/// occupancy and memory traffic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TraversalStats {
    pub node_visits: u32,
    pub triangle_tests: u32,
}

/// Möller–Trumbore ray/triangle intersection.
///
/// Solves `O + tD = (1-u-v)V0 + uV1 + vV2` for `(t, u, v)` by Cramer's rule,
/// which lets the shared sub-expressions be reused: the determinant that decides
/// whether the ray is parallel is the same quantity that normalises the
/// barycentrics, so no separate plane test is needed and no plane equation has
/// to be stored per triangle.
///
/// **Double-sided**: the test is on `|det|`, not `det > 0`. Back-face culling
/// would be faster, but a path tracer has to see the inside of surfaces — that
/// is how a ray inside a glass object finds its way out, at build step 12.
#[inline]
pub fn intersect_triangle(
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    ray: &Ray,
    t_min: f32,
    t_max: f32,
) -> Option<(f32, f32, f32)> {
    let e1 = p1 - p0;
    let e2 = p2 - p0;
    let pv = ray.dir.cross(e2);
    let det = e1.dot(pv);

    // Ray parallel to the triangle's plane. The threshold is deliberately tiny
    // and absolute: `det` is already a ratio of volumes to lengths, and a larger
    // epsilon here punches holes in geometry viewed at grazing angles.
    if det.abs() < 1.0e-12 {
        return None;
    }
    let inv_det = 1.0 / det;

    let tv = ray.origin - p0;
    let u = tv.dot(pv) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }

    let qv = tv.cross(e1);
    let v = ray.dir.dot(qv) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    let t = e2.dot(qv) * inv_det;
    if t < t_min || t > t_max {
        return None;
    }
    Some((t, u, v))
}

/// Closest hit by testing every triangle.
///
/// This is the oracle the BVH is checked against. It is O(n) and never used for
/// rendering, but it is the definition of the right answer.
pub fn brute_force_intersect(
    triangles: &[GpuTriangle],
    positions: &[[f32; 4]],
    ray: &Ray,
    t_min: f32,
    t_max: f32,
) -> Option<TriHit> {
    let mut best: Option<TriHit> = None;
    let mut closest = t_max;
    for (i, tri) in triangles.iter().enumerate() {
        let (p0, p1, p2) = tri_positions(tri, positions);
        if let Some((t, u, v)) = intersect_triangle(p0, p1, p2, ray, t_min, closest) {
            closest = t;
            best = Some(TriHit {
                t,
                u,
                v,
                triangle: i as u32,
            });
        }
    }
    best
}

#[inline]
pub fn tri_positions(tri: &GpuTriangle, positions: &[[f32; 4]]) -> (Vec3, Vec3, Vec3) {
    let g = |i: u32| {
        let p = positions[i as usize];
        Vec3::new(p[0], p[1], p[2])
    };
    (g(tri.i0), g(tri.i1), g(tri.i2))
}

/// Maximum traversal stack depth.
///
/// A binary BVH over N primitives has depth O(log N) in the good case, but the
/// stack only ever holds *deferred siblings* — one per level actually descended
/// — so 64 covers trees far deeper than anything the SAH builder produces (the
/// depth test pins it at ~3x log2(N)). The shader uses a smaller stack for
/// register-pressure reasons; see the note there.
pub const MAX_STACK: usize = 64;

impl Bvh {
    /// Closest hit, with an explicit stack and near-child-first ordering.
    ///
    /// # Why the ordering matters
    ///
    /// Descending into the nearer child first means that by the time the farther
    /// child is popped, `closest` is often already smaller than that child's
    /// entry distance, and the whole subtree is culled without being entered.
    /// Without ordering, roughly half the traversals do the work in the useless
    /// order and the node-visit count rises substantially. It costs one compare
    /// and a swap.
    pub fn intersect(
        &self,
        triangles: &[GpuTriangle],
        positions: &[[f32; 4]],
        ray: &Ray,
        t_min: f32,
        t_max: f32,
    ) -> Option<TriHit> {
        self.intersect_with_stats(
            triangles,
            positions,
            ray,
            t_min,
            t_max,
            &mut TraversalStats::default(),
        )
    }

    pub fn intersect_with_stats(
        &self,
        triangles: &[GpuTriangle],
        positions: &[[f32; 4]],
        ray: &Ray,
        t_min: f32,
        t_max: f32,
        stats: &mut TraversalStats,
    ) -> Option<TriHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let inv_dir = safe_inv_dir(ray.dir);
        let mut best: Option<TriHit> = None;
        let mut closest = t_max;

        let mut stack = [0u32; MAX_STACK];
        let mut sp = 0usize;
        let mut node = 0u32;

        loop {
            let n = &self.nodes[node as usize];
            stats.node_visits += 1;

            if n.count > 0 {
                let start = n.left_first as usize;
                for &pi in &self.prim_indices[start..start + n.count as usize] {
                    let tri = &triangles[pi as usize];
                    let (p0, p1, p2) = tri_positions(tri, positions);
                    stats.triangle_tests += 1;
                    if let Some((t, u, v)) = intersect_triangle(p0, p1, p2, ray, t_min, closest) {
                        closest = t;
                        best = Some(TriHit {
                            t,
                            u,
                            v,
                            triangle: pi,
                        });
                    }
                }
            } else {
                let c0 = n.left_first;
                let c1 = c0 + 1;
                let d0 = self
                    .node_bounds(c0 as usize)
                    .hit(ray.origin, inv_dir, t_min, closest);
                let d1 = self
                    .node_bounds(c1 as usize)
                    .hit(ray.origin, inv_dir, t_min, closest);

                match (d0, d1) {
                    (Some(a), Some(b)) => {
                        let (near, far) = if a <= b { (c0, c1) } else { (c1, c0) };
                        // The stack cannot overflow for any tree the builder
                        // produces, but silently corrupting memory if it ever
                        // did would be far worse than dropping a subtree.
                        if sp < MAX_STACK {
                            stack[sp] = far;
                            sp += 1;
                        } else {
                            debug_assert!(false, "BVH traversal stack overflow");
                        }
                        node = near;
                        continue;
                    }
                    (Some(_), None) => {
                        node = c0;
                        continue;
                    }
                    (None, Some(_)) => {
                        node = c1;
                        continue;
                    }
                    (None, None) => {}
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

    /// Any-hit within `[t_min, t_max)`, for shadow rays.
    ///
    /// Returns on the first hit rather than finding the closest, and does not
    /// narrow `t_max` as it goes — a shadow ray only needs to know *whether*
    /// something is in the way.
    pub fn occluded(
        &self,
        triangles: &[GpuTriangle],
        positions: &[[f32; 4]],
        ray: &Ray,
        t_min: f32,
        t_max: f32,
    ) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let inv_dir = safe_inv_dir(ray.dir);
        let mut stack = [0u32; MAX_STACK];
        let mut sp = 0usize;
        let mut node = 0u32;

        loop {
            let n = &self.nodes[node as usize];
            if n.count > 0 {
                let start = n.left_first as usize;
                for &pi in &self.prim_indices[start..start + n.count as usize] {
                    let (p0, p1, p2) = tri_positions(&triangles[pi as usize], positions);
                    if intersect_triangle(p0, p1, p2, ray, t_min, t_max).is_some() {
                        return true;
                    }
                }
            } else {
                let c0 = n.left_first;
                let c1 = c0 + 1;
                let h0 = self
                    .node_bounds(c0 as usize)
                    .hit(ray.origin, inv_dir, t_min, t_max)
                    .is_some();
                let h1 = self
                    .node_bounds(c1 as usize)
                    .hit(ray.origin, inv_dir, t_min, t_max)
                    .is_some();
                if h0 {
                    if h1 && sp < MAX_STACK {
                        stack[sp] = c1;
                        sp += 1;
                    }
                    node = c0;
                    continue;
                }
                if h1 {
                    node = c1;
                    continue;
                }
            }
            if sp == 0 {
                return false;
            }
            sp -= 1;
            node = stack[sp];
        }
    }
}

#[cfg(test)]
mod traversal_tests {
    use super::*;
    use crate::rng::Rng;

    fn mesh(levels: u32) -> (Vec<GpuTriangle>, Vec<[f32; 4]>) {
        let m = crate::mesh::icosphere(Vec3::ZERO, 1.0, levels);
        let mut blob = crate::gpu_layout::SceneBlob::default();
        m.append_to(&mut blob, 0);
        (blob.triangles, blob.positions)
    }

    #[test]
    fn moller_trumbore_basics() {
        let (p0, p1, p2) = (Vec3::ZERO, Vec3::X, Vec3::Y);
        let hit = Ray {
            origin: Vec3::new(0.25, 0.25, -1.0),
            dir: Vec3::Z,
        };
        let (t, u, v) = intersect_triangle(p0, p1, p2, &hit, 0.0, f32::MAX).expect("should hit");
        assert!((t - 1.0).abs() < 1e-6);
        assert!(
            (u - 0.25).abs() < 1e-6 && (v - 0.25).abs() < 1e-6,
            "barycentrics {u}, {v}"
        );

        // Outside the triangle but inside its plane.
        let miss = Ray {
            origin: Vec3::new(0.9, 0.9, -1.0),
            dir: Vec3::Z,
        };
        assert!(intersect_triangle(p0, p1, p2, &miss, 0.0, f32::MAX).is_none());

        // Behind the ray.
        let behind = Ray {
            origin: Vec3::new(0.25, 0.25, 1.0),
            dir: Vec3::Z,
        };
        assert!(intersect_triangle(p0, p1, p2, &behind, 0.0, f32::MAX).is_none());

        // Parallel.
        let parallel = Ray {
            origin: Vec3::new(0.25, 0.25, -1.0),
            dir: Vec3::X,
        };
        assert!(intersect_triangle(p0, p1, p2, &parallel, 0.0, f32::MAX).is_none());
    }

    /// Double-sided: a ray from behind must still hit.
    #[test]
    fn triangles_are_two_sided() {
        let (p0, p1, p2) = (Vec3::ZERO, Vec3::X, Vec3::Y);
        let front = Ray {
            origin: Vec3::new(0.25, 0.25, -1.0),
            dir: Vec3::Z,
        };
        let back = Ray {
            origin: Vec3::new(0.25, 0.25, 1.0),
            dir: -Vec3::Z,
        };
        assert!(intersect_triangle(p0, p1, p2, &front, 0.0, f32::MAX).is_some());
        assert!(intersect_triangle(p0, p1, p2, &back, 0.0, f32::MAX).is_some());
    }

    /// Interpolating positions with the returned barycentrics must reproduce the
    /// hit point. This is what proves `u` and `v` are actually usable for
    /// shading normals, rather than merely being in range.
    #[test]
    fn barycentrics_reconstruct_the_hit_point() {
        let (p0, p1, p2) = (
            Vec3::new(-3.0, 1.0, 2.0),
            Vec3::new(4.0, -2.0, 0.5),
            Vec3::new(1.0, 5.0, -3.0),
        );
        let mut rng = Rng::new(4, 4, 4);
        for _ in 0..1000 {
            let origin = Vec3::new(
                rng.next_f32() * 20.0 - 10.0,
                rng.next_f32() * 20.0 - 10.0,
                rng.next_f32() * 20.0 - 10.0,
            );
            let target = Vec3::new(
                rng.next_f32() * 8.0 - 4.0,
                rng.next_f32() * 8.0 - 4.0,
                rng.next_f32() * 8.0 - 4.0,
            );
            let ray = Ray {
                origin,
                dir: (target - origin).normalize(),
            };
            if let Some((t, u, v)) = intersect_triangle(p0, p1, p2, &ray, 1e-4, 1e4) {
                let from_ray = ray.origin + t * ray.dir;
                let from_bary = (1.0 - u - v) * p0 + u * p1 + v * p2;
                assert!(
                    (from_ray - from_bary).length() < 1e-3,
                    "barycentric point {from_bary} != ray point {from_ray}"
                );
            }
        }
    }

    /// **The** BVH test: traversal must return exactly what brute force returns,
    /// for every ray. Not "almost", not "usually" — the same triangle and the
    /// same distance.
    #[test]
    fn traversal_matches_brute_force() {
        let mut rng = Rng::new(0xB74, 1, 1);
        for levels in 0..5u32 {
            let (tris, pos) = mesh(levels);
            let bvh = Bvh::build(&tris, &pos);

            let mut hits = 0;
            let trials = 20_000;
            for _ in 0..trials {
                // A mix of rays: some aimed at the sphere, some random, and
                // some axis-aligned — the last are the ones that exercise flat
                // AABBs and the degenerate slab cases.
                let origin = Vec3::new(
                    rng.next_f32() * 6.0 - 3.0,
                    rng.next_f32() * 6.0 - 3.0,
                    rng.next_f32() * 6.0 - 3.0,
                );
                let dir = match rng.next_u32() % 4 {
                    0 => (Vec3::ZERO - origin).normalize(),
                    1 => [Vec3::X, Vec3::Y, Vec3::Z][(rng.next_u32() % 3) as usize],
                    2 => -[Vec3::X, Vec3::Y, Vec3::Z][(rng.next_u32() % 3) as usize],
                    _ => Vec3::new(
                        rng.next_f32() * 2.0 - 1.0,
                        rng.next_f32() * 2.0 - 1.0,
                        rng.next_f32() * 2.0 - 1.0,
                    )
                    .normalize_or_zero(),
                };
                if dir.length_squared() < 0.5 {
                    continue;
                }
                let ray = Ray { origin, dir };

                let a = bvh.intersect(&tris, &pos, &ray, 1e-4, 1e4);
                let b = brute_force_intersect(&tris, &pos, &ray, 1e-4, 1e4);

                match (a, b) {
                    (None, None) => {}
                    (Some(x), Some(y)) => {
                        hits += 1;
                        // Distances must agree to the last few ULP — both took
                        // the identical code path through the same triangle.
                        assert!(
                            (x.t - y.t).abs() <= 1e-5 * y.t.abs().max(1.0),
                            "level {levels}: t differs, bvh {} vs brute {}",
                            x.t,
                            y.t
                        );
                        // The triangle index may legitimately differ only when
                        // two triangles are at exactly the same distance.
                        if x.triangle != y.triangle {
                            assert!(
                                (x.t - y.t).abs() < 1e-6,
                                "level {levels}: different triangle at different distance"
                            );
                        }
                    }
                    (a, b) => panic!(
                        "level {levels}: BVH and brute force disagree about whether there is a hit \
                         ({}, {}) for ray {origin} -> {dir}",
                        a.map_or("miss".into(), |h| format!("hit t={}", h.t)),
                        b.map_or("miss".into(), |h| format!("hit t={}", h.t)),
                    ),
                }
            }
            // A test where nothing ever hits would pass vacuously.
            assert!(
                hits > trials / 20,
                "level {levels}: only {hits} hits, test is not exercising anything"
            );
        }
    }

    /// The same equivalence for shadow rays, which take a different code path.
    #[test]
    fn occlusion_matches_brute_force() {
        let mut rng = Rng::new(0x5AD, 2, 2);
        let (tris, pos) = mesh(3);
        let bvh = Bvh::build(&tris, &pos);
        let mut blocked = 0;
        for _ in 0..20_000 {
            let origin = Vec3::new(
                rng.next_f32() * 6.0 - 3.0,
                rng.next_f32() * 6.0 - 3.0,
                rng.next_f32() * 6.0 - 3.0,
            );
            let target = Vec3::new(
                rng.next_f32() * 6.0 - 3.0,
                rng.next_f32() * 6.0 - 3.0,
                rng.next_f32() * 6.0 - 3.0,
            );
            let d = target - origin;
            let len = d.length();
            if len < 1e-3 {
                continue;
            }
            let ray = Ray {
                origin,
                dir: d / len,
            };
            let a = bvh.occluded(&tris, &pos, &ray, 1e-4, len - 1e-4);
            let b = brute_force_intersect(&tris, &pos, &ray, 1e-4, len - 1e-4).is_some();
            assert_eq!(a, b, "occlusion disagrees for {origin} -> {target}");
            if a {
                blocked += 1;
            }
        }
        assert!(
            blocked > 1000,
            "only {blocked} occluded rays; test is too weak"
        );
    }

    /// Node visits must grow far slower than the triangle count. This is the
    /// property the whole data structure exists for, so it is asserted rather
    /// than assumed.
    #[test]
    fn traversal_cost_scales_sublinearly() {
        let mut measurements = Vec::new();
        for levels in 2..6u32 {
            let (tris, pos) = mesh(levels);
            let bvh = Bvh::build(&tris, &pos);
            let mut rng = Rng::new(7, levels, 0);
            let mut stats = TraversalStats::default();
            let rays = 2000;
            for _ in 0..rays {
                let origin =
                    Vec3::new(rng.next_f32() * 4.0 - 2.0, rng.next_f32() * 4.0 - 2.0, -3.0);
                let ray = Ray {
                    origin,
                    dir: Vec3::Z,
                };
                bvh.intersect_with_stats(&tris, &pos, &ray, 1e-4, 1e4, &mut stats);
            }
            measurements.push((
                tris.len(),
                stats.node_visits as f32 / rays as f32,
                stats.triangle_tests as f32 / rays as f32,
            ));
        }
        for (n, nodes, tris_tested) in &measurements {
            eprintln!("{n:>7} triangles: {nodes:6.1} node visits, {tris_tested:6.1} triangle tests per ray");
        }
        // 64x more triangles must not cost anywhere near 64x more work.
        let first = measurements.first().unwrap();
        let last = measurements.last().unwrap();
        let triangle_growth = last.0 as f32 / first.0 as f32;
        let visit_growth = last.1 / first.1;
        assert!(
            visit_growth < triangle_growth.log2(),
            "node visits grew {visit_growth:.1}x for {triangle_growth:.0}x more triangles"
        );
    }
}
