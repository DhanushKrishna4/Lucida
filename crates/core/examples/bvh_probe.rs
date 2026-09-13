//! Measure BVH traversal cost with real camera rays.
//!
//! `cargo run --release -p pt-core --example bvh_probe`
//!
//! Node visits per ray is the number that says whether an acceleration structure
//! is any good. Unlike a frame time it does not depend on the machine, the
//! resolution, or how much of the budget went to shading — and it is exactly the
//! quantity a wider BVH trades against.

use glam::Vec2;
use pt_core::bvh::TraversalStats;
use pt_core::camera::generate_ray;
use pt_core::integrator::{build_uniforms, RenderParams};
use pt_core::scenes;

fn main() {
    let params = RenderParams {
        width: 256,
        height: 256,
        ..Default::default()
    };

    println!(
        "{:<14} {:>9} {:>8} {:>7} {:>8} {:>9} {:>9} {:>8}",
        "scene", "tris", "nodes", "depth", "SAH", "visits/ray", "tests/ray", "max vis"
    );
    for def in scenes::all() {
        let n_tris = def.scene.blob.triangles.len();
        if n_tris == 0 {
            println!("{:<14} {:>9} (analytic, no BVH)", def.name, 0);
            continue;
        }
        let u = build_uniforms(&def, &params);
        let mut total = TraversalStats::default();
        let mut max_visits = 0u32;
        let mut rays = 0u32;

        for y in 0..params.height {
            for x in 0..params.width {
                let ray = generate_ray(&u, x, y, Vec2::splat(0.5), Vec2::ZERO);
                let mut s = TraversalStats::default();
                def.scene.bvh.intersect_with_stats(
                    &def.scene.blob.triangles,
                    &def.scene.blob.positions,
                    &ray,
                    1e-4,
                    1e30,
                    &mut s,
                );
                total.node_visits += s.node_visits;
                total.triangle_tests += s.triangle_tests;
                max_visits = max_visits.max(s.node_visits);
                rays += 1;
            }
        }

        let st = def.scene.bvh.stats;
        println!(
            "{:<14} {:>9} {:>8} {:>7} {:>8.1} {:>9.1} {:>9.1} {:>8}",
            def.name,
            n_tris,
            st.nodes,
            st.max_depth,
            st.sah_cost,
            total.node_visits as f64 / rays as f64,
            total.triangle_tests as f64 / rays as f64,
            max_visits,
        );
    }

    println!();
    println!("Node memory, at 32 bytes per binary node:");
    for def in scenes::all() {
        if def.scene.blob.triangles.is_empty() {
            continue;
        }
        let bytes = def.scene.bvh.nodes.len() * 32;
        println!(
            "  {:<14} {:>8.2} MiB  ({:.1} bytes per triangle)",
            def.name,
            bytes as f64 / (1024.0 * 1024.0),
            bytes as f64 / def.scene.blob.triangles.len() as f64
        );
    }
}
