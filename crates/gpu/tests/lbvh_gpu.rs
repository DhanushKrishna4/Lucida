//! Build step 11: the GPU builder, stage by stage against the CPU builder.
//!
//! Checked per stage rather than only on the finished tree. A tree can be valid,
//! render correctly, and still be the product of an unstable sort or a
//! mis-parenthesised range search — the symptom is a tree that is quietly worse
//! than it should be, with no image to look at and nothing failing. Diffing each
//! stage is what makes those visible.

use pt_core::bvh::Bvh;
use pt_core::lbvh::{morton_codes, normalise, Lbvh};
use pt_core::scenes;
use pt_gpu::lbvh::{build_lbvh_on_gpu, build_on_gpu};
use pt_gpu::Gpu;

fn gpu() -> Option<Gpu> {
    match Gpu::new() {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("\n*** SKIPPING GPU TEST: {e} ***\n");
            None
        }
    }
}

fn mesh_scenes() -> Vec<pt_core::scenes::SceneDef> {
    scenes::all()
        .into_iter()
        .filter(|s| !s.scene.blob.triangles.is_empty())
        .collect()
}

/// Extract one axis back out of an interleaved Morton code.
fn deinterleave(code: u32, shift: u32) -> u32 {
    let mut v = 0u32;
    for k in 0..10 {
        v |= ((code >> (3 * k + shift)) & 1) << k;
    }
    v
}

/// Morton codes must agree with the CPU's, to within one quantisation step —
/// and the ordering they induce must agree exactly.
///
/// Bit-exactness is deliberately *not* the assertion here, and it took a
/// measurement to establish why. Normalising a centroid into the unit cube is a
/// float division, and Metal's default fast-math turns `a / b` into a reciprocal
/// multiply, so the GPU and the CPU need not round it identically. Measured on
/// `bvh-stress`: 192 of 368640 codes differ, every one of them a coordinate
/// landing within an ULP of an integer boundary — the CPU gets 257.99997 and
/// truncates to 257, the GPU gets 258.0 and truncates to 258.
///
/// Asserting equality would therefore be asserting something false about
/// floating point, and the test would be permanently fragile. What is both true
/// and load-bearing is that the disagreement is bounded by one cell and that the
/// *sort order* — the only thing the rest of the build consumes — is identical.
#[test]
fn morton_codes_agree_with_the_cpu() {
    let Some(gpu) = gpu() else { return };
    for def in mesh_scenes() {
        let blob = &def.scene.blob;
        let n = blob.triangles.len();
        let out = build_on_gpu(&gpu, &blob.triangles, &blob.positions).expect("gpu build");

        // The GPU reduced the scene bounds itself; the CPU reference has to use
        // the same box or every code differs for an uninteresting reason.
        let cpu_codes = morton_codes(&blob.triangles, &blob.positions, &out.scene_bounds);

        let mut differing = 0usize;
        let mut max_axis_delta = 0i64;
        for (slot, &prim) in out.order.iter().enumerate() {
            let (g, c) = (out.codes[slot], cpu_codes[prim as usize]);
            if g == c {
                continue;
            }
            differing += 1;
            for shift in 0..3 {
                let d = deinterleave(g, shift) as i64 - deinterleave(c, shift) as i64;
                max_axis_delta = max_axis_delta.max(d.abs());
            }
        }
        eprintln!(
            "{:<14} {differing:>6} of {n} codes differ, max axis delta {max_axis_delta}",
            def.name
        );

        assert!(
            max_axis_delta <= 1,
            "{}: a code differs by {max_axis_delta} cells on some axis. Rounding \
             explains one; more than that means the WGSL and Rust `expand_bits` \
             ladders have actually diverged — check the mask constants, the clamp, \
             and the AABB padding that feeds the centroid.",
            def.name
        );
        assert!(
            differing * 1000 < n.max(1000),
            "{}: {differing} of {n} codes differ ({:.3}%). Boundary rounding is \
             rare; this is systematic.",
            def.name,
            100.0 * differing as f64 / n as f64
        );

        // The claim that actually matters. Nothing downstream reads a code —
        // only the order it produces.
        let mut cpu_order: Vec<u32> = (0..n as u32).collect();
        cpu_order.sort_by_key(|&i| (cpu_codes[i as usize], i));
        assert_eq!(
            cpu_order, out.order,
            "{}: the GPU and CPU sorts disagree on the final ordering",
            def.name
        );
    }
}

/// The GPU-reduced scene bounds must contain every centroid.
#[test]
fn scene_bounds_reduction_is_correct() {
    let Some(gpu) = gpu() else { return };
    for def in mesh_scenes() {
        let blob = &def.scene.blob;
        let out = build_on_gpu(&gpu, &blob.triangles, &blob.positions).expect("gpu build");
        let mut cpu = pt_core::bvh::Aabb::default();
        for t in &blob.triangles {
            cpu.grow_point(pt_core::bvh::triangle_bounds(t, &blob.positions).centroid());
        }
        // Exact: both reduce the same floats with min/max, which is associative
        // and exact regardless of the order the two implementations use.
        assert_eq!(
            out.scene_bounds.min.to_array(),
            cpu.min.to_array(),
            "{}: scene bounds min",
            def.name
        );
        assert_eq!(
            out.scene_bounds.max.to_array(),
            cpu.max.to_array(),
            "{}: scene bounds max",
            def.name
        );
        // And normalisation must land inside the unit cube.
        for t in blob.triangles.iter().take(1000) {
            let c = pt_core::bvh::triangle_bounds(t, &blob.positions).centroid();
            let n = normalise(c, &out.scene_bounds);
            assert!(
                n.min_element() >= 0.0 && n.max_element() <= 1.0,
                "{}: centroid normalised to {n:?}, outside the unit cube",
                def.name
            );
        }
    }
}

/// The codes the sort emits must be non-decreasing, and carry the right payload.
#[test]
fn sorted_order_is_a_permutation_in_code_order() {
    let Some(gpu) = gpu() else { return };
    for def in mesh_scenes() {
        let blob = &def.scene.blob;
        let n = blob.triangles.len();
        let out = build_on_gpu(&gpu, &blob.triangles, &blob.positions).expect("gpu build");

        assert_eq!(out.order.len(), n, "{}: order length", def.name);
        let mut seen = vec![false; n];
        for &i in &out.order {
            assert!(!seen[i as usize], "{}: index {i} appears twice", def.name);
            seen[i as usize] = true;
        }
        for w in out.codes.windows(2) {
            assert!(w[0] <= w[1], "{}: codes are not sorted", def.name);
        }
    }
}

/// Karras's hierarchy: every node reachable, every leaf used exactly once.
#[test]
fn hierarchy_is_a_well_formed_binary_tree() {
    let Some(gpu) = gpu() else { return };
    for def in mesh_scenes() {
        let blob = &def.scene.blob;
        let n = blob.triangles.len();
        let out = build_on_gpu(&gpu, &blob.triangles, &blob.positions).expect("gpu build");
        let leaf_base = (n - 1) as u32;

        // Exactly one node has no parent, and it is node 0.
        let roots: Vec<usize> = (0..n - 1).filter(|&i| out.parent[i] == u32::MAX).collect();
        assert_eq!(
            roots,
            vec![0],
            "{}: expected exactly one root at index 0, found {roots:?}",
            def.name
        );

        // Every child id is in range, and every id below the root is claimed by
        // exactly one parent.
        let mut claimed = vec![0u32; 2 * n - 1];
        for (i, c) in out.karras.iter().enumerate() {
            for &child in c {
                assert!(
                    (child as usize) < 2 * n - 1,
                    "{}: node {i} has out-of-range child {child}",
                    def.name
                );
                claimed[child as usize] += 1;
                assert_eq!(
                    out.parent[child as usize], i as u32,
                    "{}: parent of {child} disagrees with node {i}'s child list",
                    def.name
                );
            }
        }
        assert!(
            claimed[1..].iter().all(|&c| c == 1),
            "{}: {} nodes have a parent count other than 1",
            def.name,
            claimed[1..].iter().filter(|&&c| c != 1).count()
        );
        // Every leaf is claimed.
        assert!(
            (leaf_base as usize..2 * n - 1).all(|i| claimed[i] == 1),
            "{}: some leaves are unreachable",
            def.name
        );
    }
}

/// The bottom-up fit: a parent's box must contain both children's.
#[test]
fn fitted_bounds_contain_their_children() {
    let Some(gpu) = gpu() else { return };
    for def in mesh_scenes() {
        let blob = &def.scene.blob;
        let out = build_on_gpu(&gpu, &blob.triangles, &blob.positions).expect("gpu build");
        for (i, c) in out.karras.iter().enumerate() {
            let p = out.node_bounds[i];
            for &child in c {
                let b = out.node_bounds[child as usize];
                // Exact containment: both sides are min/max of the same floats,
                // so there is no accumulation error to tolerate here.
                assert!(
                    p.min.cmple(b.min).all() && p.max.cmpge(b.max).all(),
                    "{}: node {i} box {:?}..{:?} does not contain child {child} {:?}..{:?}",
                    def.name,
                    p.min,
                    p.max,
                    b.min,
                    b.max
                );
            }
        }
        // The root must contain everything.
        let root = out.node_bounds[0];
        let mut all = pt_core::bvh::Aabb::default();
        for t in &blob.triangles {
            all.grow(&pt_core::bvh::triangle_bounds(t, &blob.positions));
        }
        assert!(
            root.min.cmple(all.min).all() && root.max.cmpge(all.max).all(),
            "{}: the root box does not contain the scene",
            def.name
        );
    }
}

/// Subtree sizes must sum to the primitive count at the root.
#[test]
fn subtree_sizes_account_for_every_primitive() {
    let Some(gpu) = gpu() else { return };
    for def in mesh_scenes() {
        let blob = &def.scene.blob;
        let n = blob.triangles.len();
        let out = build_on_gpu(&gpu, &blob.triangles, &blob.positions).expect("gpu build");
        assert_eq!(
            out.subtree_size[0] as usize, n,
            "{}: the root covers {} of {n} primitives — the bottom-up walk is \
             not reaching it, which usually means the atomic gate let both \
             children through or neither",
            def.name, out.subtree_size[0]
        );
    }
}

/// The finished GPU tree must be valid, and no worse than the CPU one.
#[test]
fn gpu_tree_matches_the_cpu_tree() {
    let Some(gpu) = gpu() else { return };
    eprintln!(
        "\n{:<14} {:>9} {:>11} {:>11} {:>9}",
        "scene", "tris", "cpu cost", "gpu cost", "nodes"
    );
    for def in mesh_scenes() {
        let blob = &def.scene.blob;
        let cpu = Lbvh::build(&blob.triangles, &blob.positions);
        let gpu_tree = build_lbvh_on_gpu(&gpu, &blob.triangles, &blob.positions).expect("build");

        let problems = gpu_tree.as_bvh().validate(blob.triangles.len());
        assert!(
            problems.is_empty(),
            "{}: {:?}",
            def.name,
            &problems[..problems.len().min(5)]
        );

        eprintln!(
            "{:<14} {:>9} {:>11.2} {:>11.2} {:>9}",
            def.name,
            blob.triangles.len(),
            cpu.stats.sah_cost,
            gpu_tree.stats.sah_cost,
            gpu_tree.stats.nodes
        );

        // The two builders run the same algorithm on the same input, so the
        // trees should be identical — not merely similar. Any difference means
        // one of the stages diverged.
        assert_eq!(
            gpu_tree.nodes.len(),
            cpu.nodes.len(),
            "{}: node count differs",
            def.name
        );
        assert_eq!(
            gpu_tree.prim_indices, cpu.prim_indices,
            "{}: primitive order differs — the GPU sort and the CPU sort \
             disagree on ties",
            def.name
        );
        assert!(
            (gpu_tree.stats.sah_cost - cpu.stats.sah_cost).abs() < 1e-3,
            "{}: SAH cost {} vs {}",
            def.name,
            gpu_tree.stats.sah_cost,
            cpu.stats.sah_cost
        );
    }
}

/// And the tree the GPU built must render the same image.
#[test]
fn gpu_built_tree_renders_correctly() {
    use pt_core::image::compare;
    use pt_core::integrator::{self, RenderParams};

    let Some(gpu) = gpu() else { return };
    let params = RenderParams {
        width: 96,
        height: 96,
        samples: 24,
        max_depth: 6,
        ..Default::default()
    };

    let mut sah = scenes::cornell_mesh();
    sah.scene.build_bvh();
    let a = integrator::render(&sah, &params);

    let mut lin = scenes::cornell_mesh();
    let built = {
        let blob = &lin.scene.blob;
        build_lbvh_on_gpu(&gpu, &blob.triangles, &blob.positions).expect("build")
    };
    let mut bvh: Bvh = built.as_bvh();
    bvh.compact_primitives(&mut lin.scene.blob.triangles);
    lin.scene.blob.bvh_nodes = bvh.nodes.clone();
    lin.scene.blob.bvh_prim_indices = bvh.prim_indices.clone();
    lin.scene.bvh = bvh;
    lin.scene.blob.lights = pt_core::light::build_lights(&lin.scene.blob);
    let b = integrator::render(&lin, &params);

    let d = compare(&a, &b).expect("same size");
    eprintln!("gpu-built lbvh vs sah: mean rel {:.3e}", d.mean_rel);
    assert!(
        d.mean_rel < 2.0e-3,
        "a GPU-built tree changes the image by {:.3e} — it is missing geometry",
        d.mean_rel
    );
}
