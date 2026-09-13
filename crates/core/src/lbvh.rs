//! Linear BVH: a bounding volume hierarchy built by sorting, not by recursion.
//!
//! # Why a second builder at all
//!
//! `bvh.rs` builds a binned-SAH tree. It produces a better tree than anything
//! here will — the SAH is an actual cost model, and greedy top-down splitting
//! against it is hard to beat — but it is inherently sequential: each split
//! depends on the partition its parent chose. That is fine for a scene loaded
//! once and it is useless for geometry that changes every frame, which is what
//! animation, skinning and instancing all need.
//!
//! The LBVH (Lauterbach et al. 2009, Karras 2012) trades tree quality for
//! parallelism. Every step is a map or a sort over independent elements:
//!
//! ```text
//!   1. Morton code     one per primitive, independent          O(n) parallel
//!   2. radix sort      by code                                 O(n) parallel
//!   3. hierarchy       Karras: each internal node found alone   O(1) per node
//!   4. AABB fit        bottom-up, one atomic per node          O(n) parallel
//! ```
//!
//! Step 3 is the interesting one. The naive parallel build needs to know where
//! its children are, which is a recursive question; Karras's insight is that for
//! a *sorted* Morton array the tree is implicit in the codes' common prefixes,
//! so node `i` can determine its own range and split point from the array alone,
//! reading nothing any other node wrote.
//!
//! # Why the codes are 30-bit
//!
//! Ten bits per axis, interleaved, packed into one `u32`.
//!
//! The alternative is 21 bits per axis in a 63-bit key, which resolves far finer
//! and is what a production builder uses on large scenes. WGSL has no 64-bit
//! integer type, so that key would have to be a pair of `u32`s: two words to
//! sort, two words to compare, and a longest-common-prefix that straddles a word
//! boundary. Thirty bits keeps every one of those single-word, and a 2^30 grid
//! is 1.07e9 cells — enough that duplicate codes stay rare even at the scale of
//! `bvh-stress`. Duplicates are handled rather than assumed away (see
//! [`longest_common_prefix`]), so the failure mode when they do occur is a
//! slightly worse tree, never a wrong one.

use crate::bvh::{Aabb, BuildStats, Bvh};
use crate::gpu_layout::{GpuBvhNode, GpuTriangle};
use glam::Vec3;

/// Bits per axis in a Morton code. Three of these must fit in a `u32`.
pub const MORTON_BITS: u32 = 10;

/// Largest value an axis can take after quantisation.
pub const MORTON_SCALE: f32 = ((1u32 << MORTON_BITS) - 1) as f32; // 1023

/// Spread the low 10 bits of `v` so that each occupies every third bit.
///
/// `0000000000_0000000000_abcdefghij` becomes `a..b..c..d..e..f..g..h..i..j`.
///
/// The shift-and-mask ladder is the standard trick: each step splits the bits
/// into two halves and pushes the upper half far enough left that the gap it
/// leaves is exactly what the next step needs. Doing it this way rather than
/// with a ten-iteration loop matters because this function is also written in
/// WGSL, where a loop per primitive would be ten dependent iterations of
/// scalar work in the middle of an otherwise trivially parallel kernel.
///
/// The masks are the bit patterns that survive each step, and they are worth
/// reading as base-4/base-8 groupings rather than as magic numbers:
///
/// ```text
///   0x030000FF   0000_0011 ... 1111_1111   16-bit groups of 8
///   0x0300F00F   ... 1111 0000 0000 1111   8-bit groups of 4
///   0x030C30C3   ... 11 0000 11 0000 11    4-bit groups of 2
///   0x09249249   ... 1 00 1 00 1 00 1      the final every-third-bit pattern
/// ```
pub fn expand_bits(v: u32) -> u32 {
    let mut x = v & 0x0000_03FF; // keep 10 bits; higher bits would collide
    x = (x | (x << 16)) & 0x030000FF;
    x = (x | (x << 8)) & 0x0300F00F;
    x = (x | (x << 4)) & 0x030C30C3;
    x = (x | (x << 2)) & 0x09249249;
    x
}

/// Interleave three unit-interval coordinates into a 30-bit Morton code.
///
/// `p` must already be normalised to the scene's bounding box. Values are
/// clamped rather than wrapped: a coordinate landing exactly on the upper bound
/// would otherwise quantise to 1024 and its top bit would collide with the next
/// axis, producing a code that sorts nowhere near its neighbours.
///
/// The x bits are placed highest, so sorting by the code walks the volume in
/// z-order (the "Z" of Z-curve). Any consistent assignment works; what matters
/// is that nearby points get nearby codes, which is what makes the sorted array
/// a usable spatial ordering.
pub fn morton3d(p: Vec3) -> u32 {
    let q = |v: f32| -> u32 {
        // NaN maps to 0 through the failed comparison in `clamp`'s max, which is
        // the right answer: a degenerate primitive should sort somewhere stable
        // rather than poison the ordering.
        let s = (v * MORTON_SCALE).clamp(0.0, MORTON_SCALE);
        s as u32
    };
    (expand_bits(q(p.x)) << 2) | (expand_bits(q(p.y)) << 1) | expand_bits(q(p.z))
}

/// Map a point into `bounds`'s unit cube.
///
/// A zero-extent axis maps to 0.5 rather than dividing by zero. Flat scenes are
/// not exotic — a ground plane, a single axis-aligned quad — and an axis with no
/// extent carries no ordering information anyway, so putting every primitive at
/// the middle of it is exactly right.
pub fn normalise(p: Vec3, bounds: &Aabb) -> Vec3 {
    let extent = bounds.max - bounds.min;
    let f = |v: f32, lo: f32, e: f32| if e > 0.0 { (v - lo) / e } else { 0.5 };
    Vec3::new(
        f(p.x, bounds.min.x, extent.x),
        f(p.y, bounds.min.y, extent.y),
        f(p.z, bounds.min.z, extent.z),
    )
}

/// Morton code for every triangle, in input order, keyed on centroid.
///
/// The centroid rather than the bounds: a Morton code is a point ordering, and
/// using a corner would bias every primitive toward one side of its own extent.
pub fn morton_codes(triangles: &[GpuTriangle], positions: &[[f32; 4]], bounds: &Aabb) -> Vec<u32> {
    triangles
        .iter()
        .map(|t| {
            let b = crate::bvh::triangle_bounds(t, positions);
            morton3d(normalise(b.centroid(), bounds))
        })
        .collect()
}

/// Length of the common prefix of two codes, with the index as a tiebreak.
///
/// Karras's `delta`. It is the whole of the hierarchy: the tree over a sorted
/// Morton array is exactly the tree of common prefixes, so "how much do these
/// two codes agree" is the only question the build ever asks.
///
/// **Duplicate codes are the subtle case.** Two identical codes have a prefix of
/// 32, and the construction would then have no way to split them — the range
/// search never terminates and the tree is malformed. Appending the index makes
/// every key distinct, which is why the tiebreak is `i ^ j` leading zeros added
/// to a full 32: indices are distinct by construction, so the extended keys are
/// too. The resulting node is a poor one (it splits primitives that are
/// spatially coincident by array position) but it is a *valid* one.
///
/// Out-of-range indices return -1, which is how the range search at the array's
/// edges terminates without a bounds test in the hot loop.
pub fn longest_common_prefix(codes: &[u32], i: i32, j: i32) -> i32 {
    if j < 0 || j >= codes.len() as i32 {
        return -1;
    }
    let (a, b) = (codes[i as usize], codes[j as usize]);
    if a == b {
        // Identical codes: fall back to the indices, which cannot be identical.
        return 32 + (i as u32 ^ j as u32).leading_zeros() as i32;
    }
    (a ^ b).leading_zeros() as i32
}

/// Collapse any subtree holding at most this many primitives into one leaf.
///
/// Karras's construction produces exactly one primitive per leaf, which is a
/// tree of depth ~log2(n) whose every leaf test is a single triangle. That is a
/// great deal of box testing for very little primitive testing — the SAH cost of
/// the uncollapsed tree is dominated by traversal steps. Collapsing trades a few
/// extra triangle tests for far fewer node visits.
///
/// Four, because the traversal-cost minimum and the memory minimum pull in
/// opposite directions and four is where they meet. Measured on `bvh-stress`
/// (368k triangles), sweeping the threshold:
///
/// ```text
///   threshold     1       2       4       8      16      32
///   SAH cost   99.30   98.54  101.87  113.25  144.90  199.74
///   nodes     737279  456703  254857  140395   71071   38843
/// ```
///
/// The cost minimum is at 2. Four gives that up for 3.4% more traversal cost and
/// takes 44% fewer nodes for it — 8.2 MB against 14.6 MB at 32 bytes a node,
/// which is bandwidth every ray pays on a GPU the cost model knows nothing
/// about. Past 8 the tree stops discriminating and the curve turns sharply
/// upward. `collapse_threshold_is_near_optimal` pins the default to within 10%
/// of the measured minimum, so this stays honest if the scenes change.
pub const LBVH_MAX_LEAF: usize = 4;

/// A node of the Karras tree, in its natural layout.
///
/// Child ids live in one space of `2n - 1` slots: `[0, n-1)` are internal nodes
/// and `[n-1, 2n-1)` are leaves, so "is this child a leaf" is a comparison
/// rather than a flag. That is the layout the construction produces directly,
/// and it is *not* the layout the renderer traverses — see [`relayout`].
#[derive(Clone, Copy, Debug, Default)]
struct KarrasNode {
    left: u32,
    right: u32,
}

/// Build the internal hierarchy over sorted Morton codes (Karras 2012, §3).
///
/// Every internal node is determined independently, reading only the sorted code
/// array. There is no shared state and no ordering between nodes, which is the
/// entire reason this is worth doing: the same function runs unchanged as one
/// GPU thread per node.
///
/// The construction rests on one observation. In a sorted Morton array, the
/// range of primitives under a node is exactly the maximal run of codes sharing
/// some prefix, and the node's split is where that prefix gets one bit longer.
/// So a node needs two things — the extent of its range, and the split inside it
/// — and both are answered by binary searches over [`longest_common_prefix`].
fn build_hierarchy(codes: &[u32]) -> (Vec<KarrasNode>, Vec<u32>) {
    let n = codes.len();
    let mut nodes = vec![KarrasNode::default(); n.saturating_sub(1)];
    // Parent of every node in the unified id space, for the bottom-up fit.
    let mut parent = vec![u32::MAX; 2 * n - 1];
    let leaf_base = (n - 1) as u32;

    let delta = |i: i32, j: i32| longest_common_prefix(codes, i, j);

    for i in 0..n - 1 {
        let i = i as i32;

        // --- Which way does this node's range extend? ------------------------
        //
        // A node is anchored at `i` and runs in whichever direction its
        // neighbours agree with it more. If the code at i+1 shares a longer
        // prefix than the one at i-1, this node opens to the right.
        let d = (delta(i, i + 1) - delta(i, i - 1)).signum();

        // Anything in this node's range must share *more* prefix with i than
        // the neighbour on the far side does. That neighbour's agreement is
        // therefore the lower bound the search tests against.
        let delta_min = delta(i, i - d);

        // --- How far does it extend? -----------------------------------------
        //
        // Doubling to bracket, then a binary search to land exactly. Doubling
        // first because the range length is unbounded — a single node can cover
        // the whole array — and starting a binary search from n would cost
        // log2(n) steps for every node regardless of how small most of them are.
        let mut l_max = 2i32;
        while delta(i, i + l_max * d) > delta_min {
            l_max *= 2;
        }
        let mut l = 0i32;
        let mut t = l_max / 2;
        while t >= 1 {
            if delta(i, i + (l + t) * d) > delta_min {
                l += t;
            }
            t /= 2;
        }
        let j = i + l * d;
        let (first, last) = (i.min(j), i.max(j));

        // --- Where does it split? ---------------------------------------------
        //
        // At the last position still sharing the node's own prefix. Same binary
        // search, now bounded by the node's agreement with its far end rather
        // than by its neighbour's.
        let delta_node = delta(i, j);
        let mut s = 0i32;
        let mut t = l;
        // Ceiling division: the search must be able to reach `l` itself, and
        // halving a 1 to 0 would stop one short on odd lengths.
        while t > 1 {
            t = (t + 1) / 2;
            if delta(i, i + (s + t) * d) > delta_node {
                s += t;
            }
        }
        let split = i + s * d + d.min(0);

        // A child is a leaf exactly when it is a single element of the range.
        let left = if split == first {
            leaf_base + split as u32
        } else {
            split as u32
        };
        let right = if split + 1 == last {
            leaf_base + (split + 1) as u32
        } else {
            (split + 1) as u32
        };

        nodes[i as usize] = KarrasNode { left, right };
        parent[left as usize] = i as u32;
        parent[right as usize] = i as u32;
    }

    (nodes, parent)
}

/// A tree in the renderer's own node format, built by sorting rather than by
/// recursive splitting.
pub struct Lbvh {
    pub nodes: Vec<GpuBvhNode>,
    pub prim_indices: Vec<u32>,
    pub stats: BuildStats,
}

/// Relayout the Karras tree into the renderer's format.
///
/// Two things change. Children become **adjacent** — the traversal shader
/// derives the right child as `left + 1`, which halves the child indices it has
/// to load — and subtrees small enough to be worth it collapse into single
/// leaves (see [`LBVH_MAX_LEAF`]).
///
/// Done depth-first from the root, which is what makes siblings adjacent: a node
/// emits both of its children before either child emits anything of its own.
///
/// This pass is the one part of the build that is not embarrassingly parallel,
/// and it is why the GPU builder needs a separate ordering step rather than
/// writing Karras's output straight out.
fn relayout(
    karras: &[KarrasNode],
    subtree_size: &[u32],
    sorted_indices: &[u32],
    bounds: &[Aabb],
    n: usize,
    max_leaf: usize,
) -> (Vec<GpuBvhNode>, Vec<u32>, BuildStats) {
    let leaf_base = (n - 1) as u32;
    let mut nodes: Vec<GpuBvhNode> = Vec::with_capacity(2 * n);
    let mut prim_indices: Vec<u32> = Vec::with_capacity(n);
    let mut stats = BuildStats {
        triangles: n,
        ..Default::default()
    };

    // Gather every primitive beneath a node, in sorted order, into the leaf
    // range. Iterative rather than recursive: a degenerate Morton distribution
    // can make this tree thousands deep, and a recursive gather would overflow
    // the stack on exactly the input that most needs to work.
    let gather = |root: u32, out: &mut Vec<u32>| {
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            if id >= leaf_base {
                out.push(sorted_indices[(id - leaf_base) as usize]);
            } else {
                let k = karras[id as usize];
                // Right first so the left subtree pops first, preserving the
                // sorted order that makes leaves spatially coherent.
                stack.push(k.right);
                stack.push(k.left);
            }
        }
    };

    let bounds_of = |id: u32| bounds[id as usize];
    let size_of = |id: u32| {
        if id >= leaf_base {
            1
        } else {
            subtree_size[id as usize]
        }
    };

    // (karras id, slot in `nodes` to write, depth)
    nodes.push(GpuBvhNode::default());
    let mut stack = vec![(0u32, 0usize, 1usize)];
    while let Some((id, slot, depth)) = stack.pop() {
        stats.max_depth = stats.max_depth.max(depth);
        let b = bounds_of(id);
        let count = size_of(id) as usize;

        if id >= leaf_base || count <= max_leaf {
            let first = prim_indices.len() as u32;
            gather(id, &mut prim_indices);
            let got = prim_indices.len() as u32 - first;
            nodes[slot] = GpuBvhNode {
                bounds_min: b.min.to_array(),
                bounds_max: b.max.to_array(),
                left_first: first,
                count: got,
            };
            stats.leaves += 1;
            stats.max_leaf_size = stats.max_leaf_size.max(got as usize);
            continue;
        }

        let k = karras[id as usize];
        let left_slot = nodes.len();
        nodes.push(GpuBvhNode::default());
        nodes.push(GpuBvhNode::default());
        nodes[slot] = GpuBvhNode {
            bounds_min: b.min.to_array(),
            bounds_max: b.max.to_array(),
            left_first: left_slot as u32,
            count: 0,
        };
        // Right pushed first so left is processed first — cosmetic for
        // correctness, but it keeps node order close to primitive order, which
        // is what makes the array cache-friendly during traversal.
        stack.push((k.right, left_slot + 1, depth + 1));
        stack.push((k.left, left_slot, depth + 1));
    }

    stats.nodes = nodes.len();
    stats.mean_leaf_size = if stats.leaves > 0 {
        n as f32 / stats.leaves as f32
    } else {
        0.0
    };
    (nodes, prim_indices, stats)
}

impl Lbvh {
    /// Build a linear BVH over `triangles`.
    ///
    /// This is the CPU reference for the GPU builder, in the same sense that the
    /// CPU path tracer is the reference for the GPU one: every stage here has a
    /// WGSL twin, and the two are diffed rather than eyeballed.
    pub fn build(triangles: &[GpuTriangle], positions: &[[f32; 4]]) -> Lbvh {
        Self::build_with_leaf_size(triangles, positions, LBVH_MAX_LEAF)
    }

    /// As [`Lbvh::build`], with the collapse threshold exposed.
    ///
    /// Exists so the threshold can be swept rather than asserted — the default
    /// is justified by `collapse_threshold_is_near_optimal`, not by taste.
    pub fn build_with_leaf_size(
        triangles: &[GpuTriangle],
        positions: &[[f32; 4]],
        max_leaf: usize,
    ) -> Lbvh {
        let max_leaf = max_leaf.max(1);
        let start = std::time::Instant::now();
        let n = triangles.len();
        if n == 0 {
            return Lbvh {
                nodes: Vec::new(),
                prim_indices: Vec::new(),
                stats: BuildStats::default(),
            };
        }

        let tri_bounds: Vec<Aabb> = triangles
            .iter()
            .map(|t| crate::bvh::triangle_bounds(t, positions))
            .collect();

        // Codes are keyed on the *centroid* box, not the primitive box: the
        // ordering is a point ordering, and quantising the centroids into their
        // own extent uses the full 10 bits per axis even when the scene is thin.
        let mut centroid_bounds = Aabb::default();
        for b in &tri_bounds {
            centroid_bounds.grow_point(b.centroid());
        }

        let codes: Vec<u32> = tri_bounds
            .iter()
            .map(|b| morton3d(normalise(b.centroid(), &centroid_bounds)))
            .collect();

        // Sort by code, carrying the original index. This is the step the GPU
        // radix sort replaces, and the step its correctness is defined against.
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_by_key(|&i| (codes[i as usize], i));
        let sorted_codes: Vec<u32> = order.iter().map(|&i| codes[i as usize]).collect();

        // A single primitive has no internal nodes at all, so the general path
        // (which indexes `n - 1` internal slots) has nothing to build.
        if n == 1 {
            let stats = BuildStats {
                triangles: 1,
                nodes: 1,
                leaves: 1,
                max_depth: 1,
                max_leaf_size: 1,
                mean_leaf_size: 1.0,
                build_seconds: start.elapsed().as_secs_f64(),
                sah_cost: 1.0,
            };
            return Lbvh {
                nodes: vec![GpuBvhNode {
                    bounds_min: tri_bounds[0].min.to_array(),
                    bounds_max: tri_bounds[0].max.to_array(),
                    left_first: 0,
                    count: 1,
                }],
                prim_indices: vec![0],
                stats,
            };
        }

        let (karras, parent) = build_hierarchy(&sorted_codes);
        let leaf_base = (n - 1) as u32;

        // --- Bottom-up AABB fit ------------------------------------------------
        //
        // Walk up from every leaf. A node's bounds need *both* children, so the
        // first of two arrivals stops and the second proceeds — which is why the
        // GPU version of this is one atomic counter per internal node, and why
        // this CPU version counts visits rather than recursing.
        let mut bounds = vec![Aabb::default(); 2 * n - 1];
        let mut subtree_size = vec![0u32; n - 1];
        let mut visits = vec![0u32; n - 1];
        for (leaf, &prim) in order.iter().enumerate() {
            bounds[leaf_base as usize + leaf] = tri_bounds[prim as usize];
        }
        for leaf in 0..n {
            let mut node = parent[leaf_base as usize + leaf];
            while node != u32::MAX {
                visits[node as usize] += 1;
                if visits[node as usize] == 1 {
                    // First child to arrive: the sibling has not been fitted yet,
                    // so this node's bounds are not yet knowable.
                    break;
                }
                let k = karras[node as usize];
                let mut b = bounds[k.left as usize];
                b.grow(&bounds[k.right as usize]);
                bounds[node as usize] = b;
                let sz = |id: u32| {
                    if id >= leaf_base {
                        1
                    } else {
                        subtree_size[id as usize]
                    }
                };
                subtree_size[node as usize] = sz(k.left) + sz(k.right);
                node = parent[node as usize];
            }
        }

        let (nodes, prim_indices, mut stats) =
            relayout(&karras, &subtree_size, &order, &bounds, n, max_leaf);
        stats.build_seconds = start.elapsed().as_secs_f64();

        let mut out = Lbvh {
            nodes,
            prim_indices,
            stats,
        };
        out.stats.sah_cost = out.as_bvh().sah_cost();
        out
    }

    /// Assemble a tree from a hierarchy and fit computed elsewhere.
    ///
    /// The seam between the GPU builder and this one. Everything the GPU
    /// produces — Karras's children, the subtree sizes, the sorted order, the
    /// fitted boxes — is exactly what [`relayout`] consumes, so the GPU path
    /// reuses the collapse and the ordering rather than reimplementing them in
    /// WGSL. That also means the two builders cannot drift in leaf policy,
    /// which is the sort of difference that would otherwise show up only as an
    /// unexplained performance gap.
    pub fn from_parts(
        karras: &[[u32; 2]],
        subtree_size: &[u32],
        sorted_indices: &[u32],
        node_bounds: &[Aabb],
        n: usize,
    ) -> Lbvh {
        if n == 0 {
            return Lbvh {
                nodes: Vec::new(),
                prim_indices: Vec::new(),
                stats: BuildStats::default(),
            };
        }
        if n == 1 {
            let b = node_bounds[0];
            return Lbvh {
                nodes: vec![GpuBvhNode {
                    bounds_min: b.min.to_array(),
                    bounds_max: b.max.to_array(),
                    left_first: 0,
                    count: 1,
                }],
                prim_indices: vec![0],
                stats: BuildStats {
                    triangles: 1,
                    nodes: 1,
                    leaves: 1,
                    max_depth: 1,
                    max_leaf_size: 1,
                    mean_leaf_size: 1.0,
                    sah_cost: 1.0,
                    ..Default::default()
                },
            };
        }
        let nodes: Vec<KarrasNode> = karras
            .iter()
            .map(|c| KarrasNode {
                left: c[0],
                right: c[1],
            })
            .collect();
        let (nodes, prim_indices, mut stats) = relayout(
            &nodes,
            subtree_size,
            sorted_indices,
            node_bounds,
            n,
            LBVH_MAX_LEAF,
        );
        stats.triangles = n;
        let mut out = Lbvh {
            nodes,
            prim_indices,
            stats,
        };
        out.stats.sah_cost = out.as_bvh().sah_cost();
        out
    }

    /// View this tree as a [`Bvh`], so the existing validator, SAH cost model
    /// and traversal all apply unchanged.
    ///
    /// The formats are identical by construction — that is the point of the
    /// relayout pass — so this is a cheap clone rather than a conversion.
    pub fn as_bvh(&self) -> Bvh {
        Bvh {
            nodes: self.nodes.clone(),
            prim_indices: self.prim_indices.clone(),
            stats: self.stats,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_bits_spreads_every_third_bit() {
        assert_eq!(expand_bits(0), 0);
        // A single low bit stays put.
        assert_eq!(expand_bits(1), 1);
        // Bit 1 moves to bit 3, bit 2 to bit 6, bit k to bit 3k.
        for k in 0..MORTON_BITS {
            assert_eq!(
                expand_bits(1 << k),
                1 << (3 * k),
                "bit {k} should land at {}",
                3 * k
            );
        }
        // All ten bits set gives the every-third-bit pattern over 30 bits.
        assert_eq!(expand_bits(0x3FF), 0x09249249);
        // Bits above the tenth are discarded rather than aliasing into the
        // neighbouring axis.
        assert_eq!(expand_bits(0xFFFF_FC00), 0);
    }

    #[test]
    fn morton_interleaves_axes_without_collision() {
        // Each axis alone must occupy its own bit lane and nothing else.
        let x = morton3d(Vec3::new(1.0, 0.0, 0.0));
        let y = morton3d(Vec3::new(0.0, 1.0, 0.0));
        let z = morton3d(Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(x & y, 0, "x and y lanes overlap");
        assert_eq!(x & z, 0, "x and z lanes overlap");
        assert_eq!(y & z, 0, "y and z lanes overlap");
        assert_eq!(x | y | z, 0x3FFF_FFFF, "the three lanes must cover 30 bits");
        // And the whole code must fit in 30 bits.
        assert_eq!(morton3d(Vec3::splat(1.0)) >> 30, 0);
    }

    #[test]
    fn morton_is_monotonic_along_an_axis() {
        // Holding two axes fixed, increasing the third must increase the code.
        // This is the property that makes a sort a spatial ordering.
        let mut last = 0;
        for i in 0..=1023u32 {
            let c = morton3d(Vec3::new(0.0, 0.0, i as f32 / MORTON_SCALE));
            assert!(c > last || i == 0, "code went backwards at {i}");
            last = c;
        }
    }

    #[test]
    fn morton_clamps_rather_than_wrapping() {
        // Out-of-range input must not alias into another axis's bits.
        let hi = morton3d(Vec3::splat(1.0));
        assert_eq!(morton3d(Vec3::splat(2.0)), hi);
        assert_eq!(morton3d(Vec3::splat(-1.0)), 0);
        assert_eq!(morton3d(Vec3::splat(f32::NAN)), 0);
    }

    #[test]
    fn nearby_points_get_nearby_codes_on_average() {
        // The point of the curve, and the reason sorting by it is a spatial
        // ordering at all.
        //
        // This is deliberately an *average*, not a per-pair claim. The Z-curve
        // has seams: crossing a power-of-two boundary flips a high bit, and at
        // the very worst case — the exact centre, where 511 steps to 512 on all
        // three axes at once — a one-cell move is a maximal jump in code space.
        // An earlier version of this test asserted the per-pair version at
        // exactly that point and failed, which is the curve behaving correctly.
        //
        // What must hold is that neighbours are closer *on average* than distant
        // points, by a wide margin.
        let mut rng_state = 0x1234_5678u32;
        let mut next = || {
            // xorshift32, local to the test: no dependency on the renderer's RNG.
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 17;
            rng_state ^= rng_state << 5;
            rng_state as f32 / u32::MAX as f32
        };

        let step = 2.0 / MORTON_SCALE;
        let (mut near_sum, mut far_sum) = (0.0f64, 0.0f64);
        const N: usize = 2000;
        for _ in 0..N {
            let p = Vec3::new(next(), next(), next()) * 0.9 + Vec3::splat(0.05);
            let q = p + Vec3::new(next() - 0.5, next() - 0.5, next() - 0.5) * step;
            let r = Vec3::new(next(), next(), next());
            let c = morton3d(p) as f64;
            near_sum += (c - morton3d(q) as f64).abs();
            far_sum += (c - morton3d(r) as f64).abs();
        }
        let (near, far) = (near_sum / N as f64, far_sum / N as f64);
        eprintln!("mean code distance: adjacent {near:.3e}, random {far:.3e}");
        assert!(
            near * 10.0 < far,
            "adjacent cells ({near:.3e}) should be far closer in code space than \
             random ones ({far:.3e}); locality is what makes the sort meaningful"
        );
    }

    #[test]
    fn normalise_handles_a_flat_axis() {
        let mut b = Aabb::default();
        b.grow_point(Vec3::new(0.0, 5.0, 0.0));
        b.grow_point(Vec3::new(10.0, 5.0, 10.0));
        let n = normalise(Vec3::new(5.0, 5.0, 5.0), &b);
        assert_eq!(n.x, 0.5);
        assert_eq!(n.y, 0.5, "a zero-extent axis must not divide by zero");
        assert_eq!(n.z, 0.5);
        assert!(n.is_finite());
    }

    #[test]
    fn lcp_counts_shared_leading_bits() {
        let codes = [0b1100u32, 0b1101, 0b1000];
        // 1100 vs 1101 share all but the last bit: 32 - 1 = 31 leading bits.
        assert_eq!(longest_common_prefix(&codes, 0, 1), 31);
        // 1100 vs 1000 differ at bit 2, so they share 32 - 3 = 29.
        assert_eq!(longest_common_prefix(&codes, 0, 2), 29);
        // Out of range terminates the range search.
        assert_eq!(longest_common_prefix(&codes, 0, -1), -1);
        assert_eq!(longest_common_prefix(&codes, 0, 3), -1);
    }

    #[test]
    fn lcp_breaks_ties_on_duplicate_codes() {
        // Without the index tiebreak this returns 32 for every pair and the
        // hierarchy build cannot split them.
        let codes = [7u32, 7, 7, 7];
        for i in 0..4i32 {
            for j in 0..4i32 {
                if i == j {
                    continue;
                }
                let d = longest_common_prefix(&codes, i, j);
                assert!(d > 32, "duplicates must still be distinguishable, got {d}");
                assert!(d < 64);
            }
        }
        // And the tiebreak must still be a *prefix* measure: indices that agree
        // in more high bits must score higher.
        assert!(
            longest_common_prefix(&codes, 0, 1) > longest_common_prefix(&codes, 0, 2),
            "0 and 1 share more index bits than 0 and 2"
        );
    }
}
