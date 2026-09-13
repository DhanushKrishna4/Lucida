//! A 4-wide BVH, built by collapsing the binary one.
//!
//! # Why width is the lever
//!
//! Measured on `bvh-stress` (368k triangles), a ray costs about **31 node visits
//! and 3.7 triangle tests**. Traversal, not intersection, is where the time
//! goes — so the thing worth attacking is how many nodes a ray has to fetch, and
//! that is set by the tree's depth and branching factor.
//!
//! A 4-wide tree halves the depth. Each visit fetches four boxes instead of two,
//! but it covers two binary levels, so the number of *fetches* roughly halves
//! while the number of box *tests* stays about the same. On a GPU that is a good
//! trade twice over: the four box tests are independent and vectorise, and the
//! traversal stack — which lives in scarce private memory and sets the register
//! pressure of the whole megakernel — halves in depth.
//!
//! # Why this exists on the CPU first
//!
//! Because the trade above is an argument, not a measurement. Porting it to the
//! GPU means a new node struct through the codegen, a rewritten traversal
//! shader, and a collapse pass in both builders. This module is the cheap
//! version of that experiment: collapse on the CPU, traverse on the CPU, and see
//! whether the node-visit count actually drops before paying for the rest.

use crate::bvh::{intersect_triangle, Aabb, Bvh, TraversalStats, TriHit};
use crate::gpu_layout::GpuTriangle;
use crate::scene::Ray;
use glam::Vec3;

/// Marks a child slot that holds nothing.
pub const EMPTY: u32 = u32::MAX;

/// A 4-wide node: 128 bytes, two cache lines.
///
/// Bounds are stored **structure-of-arrays** — all four minimum x values
/// adjacent, then all four y, and so on — rather than as four separate boxes.
/// That is the layout a SIMD slab test wants: one load per axis gives the four
/// values that get compared together. It matters more on the GPU, where the
/// natural formulation is four lanes of a `vec4<f32>`, than it does in this
/// scalar reference.
#[derive(Clone, Copy, Debug)]
pub struct Bvh4Node {
    pub min: [[f32; 4]; 3],
    pub max: [[f32; 4]; 3],
    /// Child node index (internal) or first primitive (leaf).
    pub child: [u32; 4],
    /// 0 = internal child, > 0 = leaf holding that many primitives,
    /// [`EMPTY`] = unused slot.
    pub count: [u32; 4],
}

impl Default for Bvh4Node {
    fn default() -> Self {
        Self {
            // An inverted box never hits, so an unused slot costs a test and
            // never produces a false positive.
            min: [[f32::INFINITY; 4]; 3],
            max: [[f32::NEG_INFINITY; 4]; 3],
            child: [0; 4],
            count: [EMPTY; 4],
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Bvh4 {
    pub nodes: Vec<Bvh4Node>,
    pub prim_indices: Vec<u32>,
    pub max_depth: usize,
}

/// One child of a node being assembled: an index into the *binary* tree.
#[derive(Clone, Copy)]
struct Slot {
    binary: u32,
    area: f32,
}

impl Bvh4 {
    /// Collapse a binary tree into a 4-wide one.
    ///
    /// Each 4-wide node starts as a binary node's two children and then
    /// repeatedly replaces whichever child is *internal and largest* by that
    /// child's own two children, until it holds four or nothing is left to
    /// expand.
    ///
    /// Largest by surface area, because the SAH says a node's cost is
    /// proportional to the probability a random ray hits it, which is
    /// proportional to its surface area. Promoting the biggest child is
    /// therefore promoting the one a ray is most likely to have to open
    /// anyway — flattening the part of the tree that gets traversed most.
    pub fn from_binary(binary: &Bvh) -> Bvh4 {
        if binary.nodes.is_empty() {
            return Bvh4::default();
        }
        let mut out = Bvh4 {
            nodes: vec![Bvh4Node::default()],
            prim_indices: binary.prim_indices.clone(),
            max_depth: 0,
        };

        let is_leaf = |i: u32| binary.nodes[i as usize].count > 0;
        let area = |i: u32| binary.node_bounds(i as usize).surface_area();

        // (binary node to expand, slot in `out.nodes` to fill, depth)
        let mut stack = vec![(0u32, 0usize, 1usize)];
        while let Some((root, slot, depth)) = stack.pop() {
            out.max_depth = out.max_depth.max(depth);

            // Gather up to four children.
            let mut kids: Vec<Slot> = Vec::with_capacity(4);
            if is_leaf(root) {
                // A leaf at the root of this group: nothing to widen.
                kids.push(Slot {
                    binary: root,
                    area: area(root),
                });
            } else {
                let l = binary.nodes[root as usize].left_first;
                for c in [l, l + 1] {
                    kids.push(Slot {
                        binary: c,
                        area: area(c),
                    });
                }
                while kids.len() < 4 {
                    // Widen the largest internal child, if any.
                    let pick = kids
                        .iter()
                        .enumerate()
                        .filter(|(_, s)| !is_leaf(s.binary))
                        .max_by(|a, b| a.1.area.partial_cmp(&b.1.area).unwrap());
                    let Some((idx, _)) = pick else { break };
                    let s = kids.swap_remove(idx);
                    let l = binary.nodes[s.binary as usize].left_first;
                    for c in [l, l + 1] {
                        kids.push(Slot {
                            binary: c,
                            area: area(c),
                        });
                    }
                }
            }

            let mut node = Bvh4Node::default();
            for (k, s) in kids.iter().enumerate() {
                let b = binary.node_bounds(s.binary as usize);
                for axis in 0..3 {
                    node.min[axis][k] = b.min[axis];
                    node.max[axis][k] = b.max[axis];
                }
                let bn = &binary.nodes[s.binary as usize];
                if bn.count > 0 {
                    node.child[k] = bn.left_first;
                    node.count[k] = bn.count;
                } else {
                    // Reserve a slot now; fill it when the child is processed.
                    let reserved = out.nodes.len();
                    out.nodes.push(Bvh4Node::default());
                    node.child[k] = reserved as u32;
                    node.count[k] = 0;
                    stack.push((s.binary, reserved, depth + 1));
                }
            }
            out.nodes[slot] = node;
        }
        out
    }

    pub fn node_bounds(&self, node: usize, slot: usize) -> Aabb {
        let n = &self.nodes[node];
        Aabb {
            min: Vec3::new(n.min[0][slot], n.min[1][slot], n.min[2][slot]),
            max: Vec3::new(n.max[0][slot], n.max[1][slot], n.max[2][slot]),
        }
    }

    /// Closest hit, counting node visits so the width experiment has a number.
    ///
    /// Children are tested **front to back**, which is not optional. The first
    /// version of this walked slots in array order, and on `bvh-stress` that
    /// raised triangle tests per ray from 3.7 to 9.0 and made the 4-wide tree
    /// twice as slow as the binary one — because `closest` only shrinks when a
    /// near hit is found, and every leaf opened before that happens is work the
    /// ordering would have culled. It was measuring the absence of ordering, not
    /// the effect of width.
    ///
    /// Four children are sorted by entry distance with an insertion sort — at
    /// this size a sorting network buys nothing — and pushed far-to-near, so the
    /// nearest is on top of the stack.
    pub fn intersect(
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
        let inv_dir = crate::bvh::safe_inv_dir(ray.dir);
        let mut best: Option<TriHit> = None;
        let mut closest = t_max;

        // Entries are (index, count): count > 0 means a leaf holding that many
        // primitives, 0 means an internal node. Half the depth of the binary
        // tree's stack, which is the other half of the argument for widening —
        // this array is private memory on a GPU and its size sets the register
        // pressure of every thread.
        let mut stack = [(0u32, 0u32); 32];
        let mut sp = 0usize;
        let mut current = Some((0u32, 0u32));

        loop {
            let (index, count) = match current.take() {
                Some(x) => x,
                None => {
                    if sp == 0 {
                        break;
                    }
                    sp -= 1;
                    stack[sp]
                }
            };

            if count > 0 {
                let start = index as usize;
                for &p in &self.prim_indices[start..start + count as usize] {
                    stats.triangle_tests += 1;
                    let (p0, p1, p2) = crate::bvh::tri_positions(&triangles[p as usize], positions);
                    if let Some((t, u, v)) = intersect_triangle(p0, p1, p2, ray, t_min, closest) {
                        closest = t;
                        best = Some(TriHit {
                            t,
                            u,
                            v,
                            triangle: p,
                        });
                    }
                }
                continue;
            }

            stats.node_visits += 1;
            let n = &self.nodes[index as usize];

            // Entry distance, child index, child count.
            let mut hits: [(f32, u32, u32); 4] = [(0.0, 0, 0); 4];
            let mut found = 0usize;
            for k in 0..4 {
                if n.count[k] == EMPTY {
                    continue;
                }
                let b = Aabb {
                    min: Vec3::new(n.min[0][k], n.min[1][k], n.min[2][k]),
                    max: Vec3::new(n.max[0][k], n.max[1][k], n.max[2][k]),
                };
                if let Some(t) = b.hit(ray.origin, inv_dir, t_min, closest) {
                    hits[found] = (t, n.child[k], n.count[k]);
                    found += 1;
                }
            }
            if found == 0 {
                continue;
            }

            // Insertion sort, ascending by entry distance.
            for i in 1..found {
                let v = hits[i];
                let mut j = i;
                while j > 0 && hits[j - 1].0 > v.0 {
                    hits[j] = hits[j - 1];
                    j -= 1;
                }
                hits[j] = v;
            }

            // Push far-to-near so the nearest is popped first. The nearest is
            // carried in `current` rather than pushed, which is the usual trick
            // to avoid a push/pop pair on the hot path.
            for i in (1..found).rev() {
                if sp < stack.len() {
                    stack[sp] = (hits[i].1, hits[i].2);
                    sp += 1;
                } else {
                    debug_assert!(false, "BVH4 traversal stack overflow");
                }
            }
            current = Some((hits[0].1, hits[0].2));
        }
        best
    }
}
