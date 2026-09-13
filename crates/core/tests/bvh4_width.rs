//! Binary versus 4-wide BVH: the experiment, pinned.
//!
//! A wider tree is only interesting if it finds the *same* geometry. These tests
//! assert that first, and the structural properties second — a 4-wide tree that
//! halves the node visits by missing triangles would look like a triumph in the
//! counters.

use glam::Vec3;
use pt_core::bvh::{Bvh, TraversalStats};
use pt_core::bvh4::{Bvh4, EMPTY};
use pt_core::rng::Rng;
use pt_core::scene::Ray;
use pt_core::scenes;

fn mesh_scenes() -> Vec<pt_core::scenes::SceneDef> {
    scenes::all()
        .into_iter()
        .filter(|s| !s.scene.blob.triangles.is_empty())
        .collect()
}

/// Random rays through the scene's bounding sphere, so a measurement is not one
/// viewpoint's luck.
fn probe_rays(bounds: &pt_core::bvh::Aabb, n: usize) -> Vec<Ray> {
    let c = bounds.centroid();
    let r = (bounds.max - bounds.min).length();
    let mut rng = Rng::new(11, 0, 0);
    (0..n)
        .map(|_| {
            let dir = Vec3::new(
                rng.next_f32() * 2.0 - 1.0,
                rng.next_f32() * 2.0 - 1.0,
                rng.next_f32() * 2.0 - 1.0,
            )
            .normalize();
            Ray {
                origin: c - dir * r,
                dir,
            }
        })
        .collect()
}

/// The load-bearing claim: same rays, same hits.
#[test]
fn the_wide_tree_finds_the_same_geometry() {
    for def in mesh_scenes() {
        let blob = &def.scene.blob;
        let (t, p) = (&blob.triangles, &blob.positions);
        let bin = Bvh::build(t, p);
        let wide = Bvh4::from_binary(&bin);

        let mut s = TraversalStats::default();
        let mut disagreements = 0;
        let mut hits = 0;
        for ray in probe_rays(&bin.node_bounds(0), 20_000) {
            let a = bin.intersect_with_stats(t, p, &ray, 1e-4, 1e9, &mut s);
            let b = wide.intersect(t, p, &ray, 1e-4, 1e9, &mut s);
            match (a, b) {
                (None, None) => {}
                (Some(x), Some(y)) => {
                    hits += 1;
                    // The distance must agree exactly. The *triangle* need not:
                    // a ray landing on an edge shared by two coplanar triangles
                    // can resolve either way depending on visit order, and the
                    // two traversals visit in different orders by construction.
                    if x.t != y.t {
                        disagreements += 1;
                    }
                }
                _ => disagreements += 1,
            }
        }
        assert!(
            hits > 1000,
            "{}: only {hits} hits, test is too weak",
            def.name
        );
        assert_eq!(
            disagreements, 0,
            "{}: {disagreements} rays disagree between the binary and 4-wide trees",
            def.name
        );
    }
}

/// Structural invariants of the collapse.
#[test]
fn collapse_preserves_every_primitive() {
    for def in mesh_scenes() {
        let blob = &def.scene.blob;
        let bin = Bvh::build(&blob.triangles, &blob.positions);
        let wide = Bvh4::from_binary(&bin);

        let mut seen = vec![0u32; blob.triangles.len()];
        let mut internal_targets = vec![0u32; wide.nodes.len()];
        for n in &wide.nodes {
            for k in 0..4 {
                if n.count[k] == EMPTY {
                    continue;
                }
                if n.count[k] == 0 {
                    let c = n.child[k] as usize;
                    assert!(c < wide.nodes.len(), "{}: child out of range", def.name);
                    internal_targets[c] += 1;
                } else {
                    let start = n.child[k] as usize;
                    for &i in &wide.prim_indices[start..start + n.count[k] as usize] {
                        seen[i as usize] += 1;
                    }
                }
            }
        }
        assert!(
            seen.iter().all(|&c| c == 1),
            "{}: {} primitives missing, {} duplicated",
            def.name,
            seen.iter().filter(|&&c| c == 0).count(),
            seen.iter().filter(|&&c| c > 1).count()
        );
        // Every node but the root has exactly one parent.
        assert!(
            internal_targets[1..].iter().all(|&c| c == 1),
            "{}: the node graph is not a tree",
            def.name
        );
        assert_eq!(
            internal_targets[0], 0,
            "{}: the root has a parent",
            def.name
        );
    }
}

/// The measurement the width question turns on, kept as a test so it cannot
/// quietly regress.
///
/// Widening must halve node visits *without* costing triangle tests. The second
/// half is the real risk: the first version of the 4-wide traversal walked slots
/// in array order rather than front to back, and triangle tests per ray went
/// from 3.7 to 9.0 — a tree that looked wider and behaved worse.
#[test]
fn widening_halves_node_visits_at_equal_quality() {
    eprintln!(
        "\n{:<14} {:>10} {:>10} {:>9} {:>10} {:>10} {:>9}",
        "scene", "bin nodes", "4-wide", "depth", "visit/ray", "tri/ray", "bytes"
    );
    for def in mesh_scenes() {
        let blob = &def.scene.blob;
        let (t, p) = (&blob.triangles, &blob.positions);
        let bin = Bvh::build(t, p);
        let wide = Bvh4::from_binary(&bin);
        let rays = probe_rays(&bin.node_bounds(0), 20_000);

        let mut s2 = TraversalStats::default();
        for ray in &rays {
            bin.intersect_with_stats(t, p, ray, 1e-4, 1e9, &mut s2);
        }
        let mut s4 = TraversalStats::default();
        for ray in &rays {
            wide.intersect(t, p, ray, 1e-4, 1e9, &mut s4);
        }

        let visit_ratio = s4.node_visits as f64 / s2.node_visits as f64;
        let tri_ratio = s4.triangle_tests as f64 / s2.triangle_tests.max(1) as f64;
        // 32 bytes a binary node against 128 a 4-wide one.
        let byte_ratio = (wide.nodes.len() * 128) as f64 / (bin.stats.nodes * 32) as f64;
        eprintln!(
            "{:<14} {:>10} {:>10} {:>8.2}x {:>9.2}x {:>9.2}x {:>8.2}x",
            def.name,
            bin.stats.nodes,
            wide.nodes.len(),
            wide.max_depth as f64 / bin.stats.max_depth as f64,
            visit_ratio,
            tri_ratio,
            byte_ratio,
        );

        assert!(
            visit_ratio < 0.65,
            "{}: node visits only fell to {visit_ratio:.2}x. Widening should \
             roughly halve them; check that children are pushed far-to-near.",
            def.name
        );
        assert!(
            tri_ratio < 1.15,
            "{}: triangle tests rose to {tri_ratio:.2}x. The wide tree is \
             opening leaves the binary one culled, which means the front-to-back \
             ordering is not working.",
            def.name
        );
        assert!(
            byte_ratio < 1.05,
            "{}: the wide tree costs {byte_ratio:.2}x the memory",
            def.name
        );
    }
}
