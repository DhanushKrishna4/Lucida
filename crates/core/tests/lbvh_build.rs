//! Build step 11: the linear BVH must be a *valid* tree, and a usable one.
//!
//! Validity and quality are separate claims and are tested separately. A tree
//! can be perfectly valid and useless (every primitive in one leaf), and the
//! SAH cost is the number that says which it is.

use pt_core::bvh::Bvh;
use pt_core::lbvh::{Lbvh, LBVH_MAX_LEAF};
use pt_core::scenes;

/// Every scene with triangles, so the builder meets real Morton distributions
/// rather than only synthetic ones.
fn mesh_scenes() -> Vec<pt_core::scenes::SceneDef> {
    scenes::all()
        .into_iter()
        .filter(|s| !s.scene.blob.triangles.is_empty())
        .collect()
}

#[test]
fn lbvh_is_a_valid_tree() {
    for def in mesh_scenes() {
        let blob = &def.scene.blob;
        let lbvh = Lbvh::build(&blob.triangles, &blob.positions);
        let problems = lbvh.as_bvh().validate(blob.triangles.len());
        assert!(
            problems.is_empty(),
            "{}: {} structural problems, first few: {:?}",
            def.name,
            problems.len(),
            &problems[..problems.len().min(5)]
        );
    }
}

#[test]
fn every_primitive_appears_exactly_once() {
    // The validator checks this too, but failing it here says *what* went wrong
    // rather than only that something did — a dropped primitive and a duplicated
    // one have very different causes (a bad range search versus a bad relayout).
    for def in mesh_scenes() {
        let blob = &def.scene.blob;
        let lbvh = Lbvh::build(&blob.triangles, &blob.positions);
        let mut seen = vec![0u32; blob.triangles.len()];
        for &i in &lbvh.prim_indices {
            seen[i as usize] += 1;
        }
        assert_eq!(
            lbvh.prim_indices.len(),
            blob.triangles.len(),
            "{}: primitive count changed",
            def.name
        );
        assert!(
            seen.iter().all(|&c| c == 1),
            "{}: {} missing, {} duplicated",
            def.name,
            seen.iter().filter(|&&c| c == 0).count(),
            seen.iter().filter(|&&c| c > 1).count()
        );
    }
}

#[test]
fn leaves_respect_the_collapse_threshold() {
    for def in mesh_scenes() {
        let blob = &def.scene.blob;
        let lbvh = Lbvh::build(&blob.triangles, &blob.positions);
        assert!(
            lbvh.stats.max_leaf_size <= LBVH_MAX_LEAF,
            "{}: leaf of {} exceeds the collapse threshold {}",
            def.name,
            lbvh.stats.max_leaf_size,
            LBVH_MAX_LEAF
        );
    }
}

/// The quality comparison, and the reason to keep both builders.
///
/// The LBVH sorts; the binned-SAH builder optimises an actual cost model. The
/// SAH tree should win, and by how much is the price of parallel construction.
/// Pinning it means a regression in the Morton path shows up as a number rather
/// than as a slow render nobody attributes to the builder.
#[test]
fn lbvh_quality_against_binned_sah() {
    eprintln!(
        "\n{:<14} {:>9} {:>12} {:>12} {:>9} {:>10} {:>10}",
        "scene", "tris", "sah cost", "lbvh cost", "ratio", "sah nodes", "lbvh nodes"
    );
    for def in mesh_scenes() {
        let blob = &def.scene.blob;
        let sah = Bvh::build(&blob.triangles, &blob.positions);
        let lbvh = Lbvh::build(&blob.triangles, &blob.positions);
        let ratio = lbvh.stats.sah_cost / sah.stats.sah_cost;
        eprintln!(
            "{:<14} {:>9} {:>12.2} {:>12.2} {:>9.2}x {:>10} {:>10}",
            def.name,
            blob.triangles.len(),
            sah.stats.sah_cost,
            lbvh.stats.sah_cost,
            ratio,
            sah.stats.nodes,
            lbvh.stats.nodes
        );
        assert!(
            ratio < 3.0,
            "{}: the LBVH is {ratio:.2}x the SAH tree's traversal cost. \
             Sorting buys parallelism and costs quality, but past about 2x the \
             tree is not doing its job — suspect the collapse threshold or a \
             degenerate Morton distribution.",
            def.name
        );
    }
}

/// Degenerate inputs, which are where a range search goes wrong.
#[test]
fn handles_pathological_primitive_counts() {
    let blob = &scenes::cornell_mesh().scene.blob;
    for n in [1usize, 2, 3, 4, 5, 7, 8, 9, 33] {
        if n > blob.triangles.len() {
            break;
        }
        let tris = &blob.triangles[..n];
        let lbvh = Lbvh::build(tris, &blob.positions);
        let problems = lbvh.as_bvh().validate(n);
        assert!(problems.is_empty(), "n={n}: {problems:?}");
        assert_eq!(lbvh.prim_indices.len(), n, "n={n}: lost primitives");
    }
}

/// Coincident primitives give identical Morton codes.
///
/// This is the case the index tiebreak in `longest_common_prefix` exists for.
/// Without it the range search does not terminate; with it the tree is poor but
/// correct. Worth its own test because no natural scene reliably produces it.
#[test]
fn handles_duplicate_morton_codes() {
    let blob = &scenes::cornell_mesh().scene.blob;
    // Every triangle identical, so every centroid — and so every code — is too.
    let tris = vec![blob.triangles[0]; 64];
    let lbvh = Lbvh::build(&tris, &blob.positions);
    let problems = lbvh.as_bvh().validate(tris.len());
    assert!(problems.is_empty(), "duplicate codes: {problems:?}");
    assert_eq!(lbvh.prim_indices.len(), 64);
}

/// A build is a pure function of its input.
#[test]
fn build_is_deterministic() {
    let blob = &scenes::cornell_mesh().scene.blob;
    let a = Lbvh::build(&blob.triangles, &blob.positions);
    let b = Lbvh::build(&blob.triangles, &blob.positions);
    assert_eq!(a.nodes.len(), b.nodes.len());
    assert_eq!(a.prim_indices, b.prim_indices);
    for (x, y) in a.nodes.iter().zip(b.nodes.iter()) {
        assert_eq!(x.left_first, y.left_first);
        assert_eq!(x.count, y.count);
        assert_eq!(x.bounds_min, y.bounds_min);
        assert_eq!(x.bounds_max, y.bounds_max);
    }
}

/// The collapse threshold, measured rather than assumed.
///
/// Karras produces one primitive per leaf. That is a lot of box tests for one
/// triangle test each, so subtrees get collapsed — but collapse too far and the
/// tree stops discriminating and every ray tests primitives it should have
/// culled. The SAH cost model prices both sides, so the right threshold is
/// simply the minimum of that curve.
#[test]
fn collapse_threshold_is_near_optimal() {
    use pt_core::lbvh::Lbvh;
    let sizes = [1usize, 2, 4, 8, 16, 32];
    eprintln!("\nSAH cost versus collapse threshold (lower is better):");
    eprint!("{:<14}", "scene");
    for s in sizes {
        eprint!("{:>9}", format!("n<={s}"));
    }
    eprintln!("{:>9}", "best");

    for def in mesh_scenes() {
        let blob = &def.scene.blob;
        eprint!("{:<14}", def.name);
        let mut best = (f32::INFINITY, 0usize);
        let mut nodes = Vec::new();
        for s in sizes {
            let t = Lbvh::build_with_leaf_size(&blob.triangles, &blob.positions, s);
            eprint!("{:>9.2}", t.stats.sah_cost);
            nodes.push(t.stats.nodes);
            if t.stats.sah_cost < best.0 {
                best = (t.stats.sah_cost, s);
            }
        }
        eprintln!("{:>9}", best.1);
        // Node count is the other half of the trade: the cost model prices
        // traversal steps, not the memory they read.
        eprint!("{:<14}", "  nodes");
        for n in &nodes {
            eprint!("{n:>9}");
        }
        eprintln!();

        // The default must be within a few percent of the measured optimum. Not
        // equal to it: the optimum moves with the scene, and a threshold that
        // chases each one would be overfitting to this test's six scenes.
        let chosen = Lbvh::build_with_leaf_size(&blob.triangles, &blob.positions, LBVH_MAX_LEAF);
        assert!(
            chosen.stats.sah_cost <= best.0 * 1.10,
            "{}: the default threshold {} costs {:.2}, more than 10% above the \
             best measured ({:.2} at {})",
            def.name,
            LBVH_MAX_LEAF,
            chosen.stats.sah_cost,
            best.0,
            best.1
        );
    }
}

/// The claim an acceleration structure lives or dies by: it must not be visible.
///
/// Two different trees over the same geometry must return the same closest hit
/// for every ray. If they disagree, one of them is missing geometry — and a BVH
/// that misses geometry intermittently reads as *noise*, which is close to
/// impossible to attribute after the fact. So this is asserted on the image, at
/// a sample count high enough that a systematic miss cannot hide under variance.
#[test]
fn the_tree_does_not_change_the_image() {
    use pt_core::image::compare;
    use pt_core::integrator::{self, RenderParams};
    use pt_core::scene::BvhBuilder;

    // Constructed fresh per builder rather than cloned: `build_bvh_with`
    // compacts the triangle array into traversal order, so the two builders must
    // not share one.
    /// A scene to run both builders over, constructed fresh each time.
    type Case = (&'static str, fn() -> pt_core::scenes::SceneDef);
    let cases: [Case; 1] = [("cornell-mesh", scenes::cornell_mesh)];

    for (name, make) in cases {
        let params = RenderParams {
            width: 96,
            height: 96,
            samples: 32,
            max_depth: 6,
            ..Default::default()
        };

        let mut sah = make();
        sah.scene.build_bvh_with(BvhBuilder::BinnedSah);
        let a = integrator::render(&sah, &params);

        let mut lin = make();
        lin.scene.build_bvh_with(BvhBuilder::Linear);
        let b = integrator::render(&lin, &params);

        let d = compare(&a, &b).expect("same size");
        eprintln!(
            "{name}: sah vs lbvh  mean rel {:.3e}  max abs {:.3e}  energy {:.6}",
            d.mean_rel,
            d.max_abs,
            d.mean_b / d.mean_a
        );
        // Measured bit-exact on this scene, which is the strongest form of the
        // claim: the two trees visit nodes in completely different orders and
        // still agree to the last bit, so closest-hit really is order
        // independent here. The threshold is nonetheless a tolerance rather
        // than an equality, because a ray landing exactly on an edge shared by
        // two triangles may resolve to either one depending on visit order, and
        // a scene with more coplanar geometry would show those ULP-level ties.
        // What the threshold rules out is a *missed* hit.
        assert!(
            d.mean_rel < 2.0e-3,
            "{name}: the two trees disagree by {:.3e} mean relative error. \
             At this level one of them is missing geometry, not merely breaking \
             ties differently.",
            d.mean_rel
        );
        assert!(
            (d.mean_b / d.mean_a - 1.0).abs() < 1e-3,
            "{name}: energy ratio {:.6} — one tree is dropping hits",
            d.mean_b / d.mean_a
        );
    }
}
