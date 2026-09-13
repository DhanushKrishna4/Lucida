//! Area lights and next event estimation.
//!
//! # The measure conversion, derived once
//!
//! This is the single most common source of energy errors in a path tracer, so
//! the derivation lives here and everything else refers to it.
//!
//! Light sampling picks a **point** on an emitter, so its density is naturally
//! with respect to **area**: choose a light with probability `p_select`, then a
//! point uniformly over its surface with density `1 / area`. So
//!
//! ```text
//!   p_A(x') = p_select / area
//! ```
//!
//! But the light transport integral is over **solid angle** at the shading
//! point, and the BSDF's pdf is a solid-angle density too. Adding an
//! area-measure density to a solid-angle one is adding apples to oranges — it
//! produces an image that looks entirely plausible and is wrong by a factor that
//! varies with distance, which is exactly why this bug is hard to spot.
//!
//! The conversion comes from how a patch of area projects to a solid angle seen
//! from distance `d`:
//!
//! ```text
//!   dw = dA * |cos(theta')| / d^2
//! ```
//!
//! where `theta'` is the angle between the light's normal and the direction back
//! to the shading point. The `cos` accounts for the patch being seen obliquely,
//! and the `1/d^2` for it subtending less angle further away. Densities
//! transform by the *inverse* Jacobian, so
//!
//! ```text
//!   p_w(w) = p_A(x') * dA/dw = p_A(x') * d^2 / |cos(theta')|
//! ```
//!
//! Two consequences worth keeping in mind:
//!
//! * the pdf **grows** with distance, so distant lights are correctly given
//!   less weight per sample;
//! * it blows up as the light is viewed edge-on (`cos(theta') -> 0`), which is
//!   the classic firefly source in NEE. Those samples contribute almost nothing
//!   anyway — the emitter is edge-on — so they are rejected rather than divided
//!   by a near-zero.

use crate::gpu_layout::{GpuLight, GpuMaterial, SceneBlob, LIGHT_KIND_QUAD, LIGHT_KIND_TRIANGLE};
use crate::scene::{Ray, Scene};
use glam::{Vec2, Vec3};

/// Below this, the emitter is edge-on and the solid-angle pdf diverges. The
/// sample carries essentially no energy, so discarding it costs nothing and
/// removes the firefly.
const MIN_COS_LIGHT: f32 = 1.0e-6;

/// Collect every emissive primitive in a scene into a flat list of area lights.
///
/// Quads contribute one entry; triangles one each. Analytic spheres are not
/// emitters yet — sampling a sphere's *visible cap* rather than its whole
/// surface is a different (and better) estimator, and mixing the two here would
/// obscure the measure conversion this module exists to get right.
pub fn build_lights(blob: &SceneBlob) -> Vec<GpuLight> {
    let mut lights = Vec::new();
    let emits = |m: u32| -> bool {
        let e = blob.materials[m as usize].emissive;
        e[0] > 0.0 || e[1] > 0.0 || e[2] > 0.0
    };

    for prim in &blob.primitives {
        // Only quads emit for now. Sampling a sphere's *visible cap* rather than
        // its whole surface is a different and better estimator, and mixing the
        // two here would obscure the measure conversion this module exists to
        // get right.
        if !prim.is_quad() || !emits(prim.material) {
            continue;
        }
        let eu = Vec3::from_array(prim.edge_u);
        let ev = Vec3::from_array(prim.edge_v);
        // The magnitude of the edge cross product is the parallelogram's area.
        let cross = eu.cross(ev);
        lights.push(GpuLight {
            origin: prim.position,
            area: cross.length(),
            edge_u: prim.edge_u,
            kind: LIGHT_KIND_QUAD,
            edge_v: prim.edge_v,
            material: prim.material,
            normal: prim.normal,
            _pad0: 0.0,
        });
    }

    for t in &blob.triangles {
        if !emits(t.material) {
            continue;
        }
        let p = |i: u32| {
            let v = blob.positions[i as usize];
            Vec3::new(v[0], v[1], v[2])
        };
        let (p0, p1, p2) = (p(t.i0), p(t.i1), p(t.i2));
        let e1 = p1 - p0;
        let e2 = p2 - p0;
        let cross = e1.cross(e2);
        let len = cross.length();
        if len <= 0.0 {
            continue; // degenerate triangle: zero area, cannot be sampled
        }
        lights.push(GpuLight {
            origin: p0.to_array(),
            // Half the parallelogram.
            area: 0.5 * len,
            edge_u: e1.to_array(),
            kind: LIGHT_KIND_TRIANGLE,
            edge_v: e2.to_array(),
            material: t.material,
            normal: (cross / len).to_array(),
            _pad0: 0.0,
        });
    }

    lights
}

/// A point sampled on an emitter, with its density already converted to solid
/// angle at the shading point.
#[derive(Clone, Copy, Debug)]
pub struct LightSample {
    /// Direction from the shading point toward the light, normalised.
    pub wi: Vec3,
    /// Distance to the sampled point.
    pub distance: f32,
    pub radiance: Vec3,
    /// **Solid-angle** density at the shading point.
    pub pdf: f32,
    /// Which light was chosen, for the MIS weight at build step 9.
    pub light_index: u32,
}

/// Uniformly sample a point on `light`'s surface.
///
/// Returns the point and the outward normal there.
#[inline]
pub fn sample_light_point(light: &GpuLight, u: Vec2) -> (Vec3, Vec3) {
    let origin = Vec3::from_array(light.origin);
    let eu = Vec3::from_array(light.edge_u);
    let ev = Vec3::from_array(light.edge_v);
    let normal = Vec3::from_array(light.normal);

    let p = if light.kind == LIGHT_KIND_TRIANGLE {
        // Uniform on a triangle. The square root warps the unit square onto the
        // triangle with constant Jacobian — without it, samples pile up toward
        // one corner, which biases the estimate toward whatever that corner
        // happens to illuminate.
        let su = u.x.sqrt();
        let b1 = 1.0 - su;
        let b2 = u.y * su;
        origin + b1 * eu + b2 * ev
    } else {
        // Uniform on a parallelogram is just the unit square, unwarped.
        origin + u.x * eu + u.y * ev
    };
    (p, normal)
}

/// Sample the light list and return a shadow-ray candidate, or `None` when the
/// sample cannot contribute.
///
/// Selection is **uniform over lights** at this stage. Power-weighted selection
/// and a light BVH are the documented upgrades; uniform is correct, just higher
/// variance when lights differ greatly in brightness, and keeping it simple here
/// means the measure conversion is the only thing under test.
pub fn sample_lights(
    lights: &[GpuLight],
    materials: &[GpuMaterial],
    shading_point: Vec3,
    u_select: f32,
    u_area: Vec2,
) -> Option<LightSample> {
    if lights.is_empty() {
        return None;
    }
    let n = lights.len();
    // `min` guards the u = 1.0 edge, which `next_f32` cannot produce but a
    // low-discrepancy sequence later might.
    let index = ((u_select * n as f32) as usize).min(n - 1);
    let light = &lights[index];

    let (point, light_normal) = sample_light_point(light, u_area);
    let to_light = point - shading_point;
    let dist_sq = to_light.length_squared();
    if dist_sq <= 0.0 {
        return None;
    }
    let distance = dist_sq.sqrt();
    let wi = to_light / distance;

    // The emitter is one-sided: it radiates from the face its normal points
    // from. `cos_light` is the angle at the *light*, between its normal and the
    // direction back toward the shading point.
    let cos_light = light_normal.dot(-wi);
    if cos_light <= MIN_COS_LIGHT {
        return None;
    }

    // Area measure -> solid angle. See the module documentation for the
    // derivation; this is the line where energy errors live.
    let pdf_area = 1.0 / (n as f32 * light.area);
    let pdf = pdf_area * dist_sq / cos_light;
    if !pdf.is_finite() || pdf <= 0.0 {
        return None;
    }

    Some(LightSample {
        wi,
        distance,
        radiance: Vec3::from_array(materials[light.material as usize].emissive),
        pdf,
        light_index: index as u32,
    })
}

/// Solid-angle density the light sampler *would have* assigned to a direction
/// that happened to land on an emitter, given only the geometry at the hit.
///
/// This is what multiple importance sampling needs from the BSDF side: having
/// walked into a light, how likely was the other strategy to have found the same
/// direction? Answering it needs no light index — the emitter's area, its normal
/// and the number of lights are enough, and all three are available at the hit.
///
/// **`area == 0` means "not sampleable as a light"**, and the function returns
/// zero. An emissive sphere is the case: `build_lights` does not include spheres
/// (sampling a sphere's visible cap is a different estimator), so light sampling
/// genuinely cannot produce that direction and the BSDF strategy must take full
/// credit for it. Returning a spurious non-zero density here would silently
/// darken emissive spheres.
///
/// The density is in **solid angle**, the same measure the BSDF's pdf uses.
/// Mixing measures in a MIS weight is the classic way to get an image that is
/// subtly wrong everywhere and obviously wrong nowhere.
pub fn light_pdf_from_geometry(
    num_lights: u32,
    area: f32,
    light_normal: Vec3,
    shading_point: Vec3,
    hit_point: Vec3,
) -> f32 {
    if num_lights == 0 || area <= 0.0 {
        return 0.0;
    }
    let to_light = hit_point - shading_point;
    let dist_sq = to_light.length_squared();
    if dist_sq <= 0.0 {
        return 0.0;
    }
    let wi = to_light / dist_sq.sqrt();
    let cos_light = light_normal.dot(-wi);
    if cos_light <= MIN_COS_LIGHT {
        return 0.0;
    }
    let pdf_area = 1.0 / (num_lights as f32 * area);
    pdf_area * dist_sq / cos_light
}

/// As [`light_pdf_from_geometry`], addressed by light index instead.
///
/// Kept next to the sampler so the two cannot drift apart; the tests assert they
/// agree with what `sample_lights` reported.
pub fn light_pdf(
    lights: &[GpuLight],
    light_index: u32,
    shading_point: Vec3,
    hit_point: Vec3,
    light_normal: Vec3,
) -> f32 {
    let Some(light) = lights.get(light_index as usize) else {
        return 0.0;
    };
    light_pdf_from_geometry(
        lights.len() as u32,
        light.area,
        light_normal,
        shading_point,
        hit_point,
    )
}

/// The **power heuristic** with beta = 2 (Veach 1995).
///
/// ```text
///   w_a = p_a^2 / (p_a^2 + p_b^2)
/// ```
///
/// Given two strategies that can both generate a direction, this decides how
/// much credit each gets. The weights sum to 1 for any pair of densities, which
/// is what keeps the combined estimator unbiased — no path is counted twice and
/// none is dropped.
///
/// Squaring is what makes it better than the balance heuristic (`beta = 1`): it
/// pushes weight more aggressively toward whichever strategy sampled the
/// direction densely, which suppresses the low-probability samples that turn
/// into fireflies. Veach found beta = 2 close to optimal across a range of
/// scenes, and it is the standard choice.
///
/// Computed as `1 / (1 + (p_b/p_a)^2)` rather than literally squaring both.
/// Algebraically identical, but it cannot overflow: a GGX pdf reaches 1e5 at low
/// roughness, so `p^2` is 1e10 — fine on its own, but the ratio form stays
/// well-conditioned however extreme the pair becomes, and degrades to exactly
/// 0 or 1 rather than to NaN.
#[inline]
pub fn power_heuristic(pdf_a: f32, pdf_b: f32) -> f32 {
    if pdf_a <= 0.0 {
        return 0.0;
    }
    let r = pdf_b / pdf_a;
    1.0 / (1.0 + r * r)
}

/// Is the straight path from `from` to the sampled light point unobstructed?
///
/// `t_max` stops just short of the light so the emitter itself does not count as
/// its own occluder. The shortening is relative to the distance for the same
/// reason ray offsets are relative to position magnitude: float spacing scales
/// with magnitude, so a fixed epsilon is simultaneously too small far away and
/// too large up close.
#[inline]
pub fn unoccluded(scene: &Scene, from: Vec3, wi: Vec3, distance: f32) -> bool {
    let ray = Ray {
        origin: from,
        dir: wi,
    };
    !scene.occluded(&ray, distance * (1.0 - 1.0e-3))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::scenes;

    #[test]
    fn cornell_box_has_one_light_with_the_right_area() {
        let def = scenes::cornell_box();
        let lights = build_lights(&def.scene.blob);
        assert_eq!(lights.len(), 1, "expected exactly one emissive quad");
        // The ceiling light spans x in [213, 343] and z in [227, 332].
        let expected = 130.0 * 105.0;
        assert!(
            (lights[0].area - expected).abs() < 1.0,
            "light area {} should be {expected}",
            lights[0].area
        );
        // It must face down into the room.
        assert!(
            lights[0].normal[1] < -0.99,
            "light normal {:?}",
            lights[0].normal
        );
    }

    /// A triangulated emitter must have the same total area as the quad it
    /// replaced, and sampling it must cover the same surface.
    #[test]
    fn triangulated_light_matches_the_quad_it_replaces() {
        let def = scenes::cornell_box();
        let quad_lights = build_lights(&def.scene.blob);

        let mut tri = scenes::cornell_box();
        let prims = std::mem::take(&mut tri.scene.blob.primitives);
        for q in prims.iter().filter(|p| p.is_quad()) {
            crate::mesh::quad(
                Vec3::from_array(q.position),
                Vec3::from_array(q.edge_u),
                Vec3::from_array(q.edge_v),
            )
            .append_to(&mut tri.scene.blob, q.material);
        }
        let tri_lights = build_lights(&tri.scene.blob);

        assert_eq!(tri_lights.len(), 2, "a quad light becomes two triangles");
        let quad_area = quad_lights[0].area;
        let tri_area: f32 = tri_lights.iter().map(|l| l.area).sum();
        assert!(
            (tri_area - quad_area).abs() < 1e-2,
            "triangulated light area {tri_area} != quad area {quad_area}"
        );
    }

    /// Sampled points must lie on the emitter, and be uniformly distributed over
    /// it. Non-uniformity biases the estimate toward whatever that region
    /// happens to illuminate, which is invisible in an image.
    #[test]
    fn light_points_are_uniform_over_the_surface() {
        for kind in [LIGHT_KIND_QUAD, LIGHT_KIND_TRIANGLE] {
            let eu = Vec3::new(4.0, 0.0, 0.0);
            let ev = Vec3::new(0.0, 0.0, 3.0);
            let cross = eu.cross(ev);
            let light = GpuLight {
                origin: [1.0, 5.0, -2.0],
                area: if kind == LIGHT_KIND_TRIANGLE {
                    0.5 * cross.length()
                } else {
                    cross.length()
                },
                edge_u: eu.to_array(),
                kind,
                edge_v: ev.to_array(),
                material: 0,
                normal: cross.normalize().to_array(),
                _pad0: 0.0,
            };

            let mut rng = Rng::new(31, kind, 0);
            let n = 200_000;
            // Split the surface into a grid in (a, b) coordinates and count.
            const G: usize = 8;
            let mut counts = [[0u32; G]; G];
            let mut inside = 0u32;
            for _ in 0..n {
                let (p, _) = sample_light_point(&light, rng.next_vec2());
                // Recover the parametric coordinates.
                let d = p - Vec3::from_array(light.origin);
                let a = d.dot(eu) / eu.length_squared();
                let b = d.dot(ev) / ev.length_squared();
                assert!(
                    (-1e-4..=1.0 + 1e-4).contains(&a) && (-1e-4..=1.0 + 1e-4).contains(&b),
                    "sampled point off the emitter: ({a}, {b})"
                );
                if kind == LIGHT_KIND_TRIANGLE {
                    assert!(
                        a + b <= 1.0 + 1e-4,
                        "point outside the triangle: ({a}, {b})"
                    );
                }
                // The point must be in the emitter's plane.
                let n_hat = Vec3::from_array(light.normal);
                assert!(
                    d.dot(n_hat).abs() < 1e-3,
                    "point off the plane by {}",
                    d.dot(n_hat)
                );

                let ia = ((a * G as f32) as usize).min(G - 1);
                let ib = ((b * G as f32) as usize).min(G - 1);
                counts[ia][ib] += 1;
                inside += 1;
            }

            // Uniform over the region means every cell that is *fully* inside it
            // gets the same expected count.
            let mut full_cells = Vec::new();
            for (ia, row) in counts.iter().enumerate() {
                for (ib, &count) in row.iter().enumerate() {
                    // A cell is fully inside the triangle when its far corner is.
                    let far = (ia + 1) as f32 / G as f32 + (ib + 1) as f32 / G as f32;
                    if kind != LIGHT_KIND_TRIANGLE || far <= 1.0 {
                        full_cells.push(count as f64);
                    }
                }
            }
            let mean = full_cells.iter().sum::<f64>() / full_cells.len() as f64;
            for (i, &c) in full_cells.iter().enumerate() {
                // Poisson: standard deviation is sqrt(mean); allow 5 sigma.
                let sigma = mean.sqrt();
                assert!(
                    (c - mean).abs() < 5.0 * sigma,
                    "kind {kind}: cell {i} has {c} samples, expected about {mean:.0} \
                     (+/- {:.0}) — the sampler is not uniform",
                    5.0 * sigma
                );
            }
            assert_eq!(inside, n as u32);
        }
    }

    /// **The measure conversion, tested directly.**
    ///
    /// If `p_w` is a correct solid-angle density, then integrating `1 / p_w`
    /// over the sampled directions must converge to the solid angle the light
    /// actually subtends. Computing that solid angle independently — by Monte
    /// Carlo over directions rather than over the light's area — gives two
    /// routes to the same number, and they only agree if the `d^2 / cos` factor
    /// is right.
    ///
    /// This is the check that catches an area-measure pdf used as a solid-angle
    /// one: that error scales with distance, so it is invisible at one distance
    /// and obvious across several.
    #[test]
    fn solid_angle_conversion_is_correct() {
        let eu = Vec3::new(2.0, 0.0, 0.0);
        let ev = Vec3::new(0.0, 0.0, 2.0);
        // cross(eu, ev) is already (0, -1, 0) here, i.e. facing down toward the
        // shading points below. Negating it would point the emitter away and
        // every sample would be (correctly) rejected as back-facing.
        let cross = eu.cross(ev);
        let normal = cross.normalize();
        let light = GpuLight {
            origin: [-1.0, 0.0, -1.0],
            area: cross.length(),
            edge_u: eu.to_array(),
            kind: LIGHT_KIND_QUAD,
            edge_v: ev.to_array(),
            material: 0,
            normal: normal.to_array(),
            _pad0: 0.0,
        };
        let materials = vec![GpuMaterial::emissive(Vec3::ONE)];
        let lights = [light];

        for &height in &[1.0f32, 3.0, 10.0, 40.0] {
            let shading_point = Vec3::new(0.0, -height, 0.0);

            // Route 1: E[1 / p_w] over light samples.
            let mut rng = Rng::new(41, height as u32, 0);
            let n = 400_000;
            let mut sum = 0.0f64;
            for _ in 0..n {
                let u_select = rng.next_f32();
                let u_area = rng.next_vec2();
                if let Some(s) = sample_lights(&lights, &materials, shading_point, u_select, u_area)
                {
                    sum += 1.0 / s.pdf as f64;
                }
            }
            let from_pdf = sum / n as f64;

            // Route 2: the exact solid angle a rectangle subtends from a point
            // on its central axis,
            //
            //   Omega = 4 * atan( a*b / (d * sqrt(a^2 + b^2 + d^2)) )
            //
            // for half-extents a and b at distance d. Closed form, so the
            // reference carries no noise of its own.
            //
            // Monte Carlo was tried first and had to be abandoned: at height 40
            // the light subtends 0.0024 sr, so four million hemisphere samples
            // produce only about 1500 hits and the *reference* has 2.6% error.
            // The test was failing on the noise in its own ground truth.
            let (a, b, d) = (1.0f64, 1.0f64, height as f64);
            let from_geometry = 4.0 * (a * b / (d * (a * a + b * b + d * d).sqrt())).atan();

            let rel = (from_pdf - from_geometry).abs() / from_geometry;
            eprintln!(
                "height {height:>5}: solid angle from pdf {from_pdf:.6}, from geometry \
                 {from_geometry:.6}, relative difference {rel:.4}"
            );
            assert!(
                rel < 0.01,
                "at height {height} the light-sampling pdf implies a solid angle of \
                 {from_pdf:.6} but the light actually subtends {from_geometry:.6}. \
                 The area-to-solid-angle conversion is wrong."
            );
        }
    }

    /// The light count must enter the pdf, and it must enter it the right way
    /// round.
    ///
    /// Adding a second light **halves** the density for the first, because it is
    /// now chosen half as often — and the estimator's `1 / pdf` compensates
    /// exactly. It is easy to talk oneself into the opposite; the assertion here
    /// was originally written backwards.
    #[test]
    fn light_count_scales_the_pdf() {
        let make = |origin: [f32; 3], scale: f32| {
            let eu = Vec3::new(scale, 0.0, 0.0);
            let ev = Vec3::new(0.0, 0.0, 1.0);
            let cross = eu.cross(ev);
            GpuLight {
                origin,
                area: cross.length(),
                edge_u: eu.to_array(),
                kind: LIGHT_KIND_QUAD,
                edge_v: ev.to_array(),
                material: 0,
                // Faces down (-y), toward the shading point below it.
                normal: cross.normalize().to_array(),
                _pad0: 0.0,
            }
        };
        let materials = vec![GpuMaterial::emissive(Vec3::ONE)];
        let p = Vec3::new(0.5, -2.0, 0.5);

        let one = [make([0.0, 0.0, 0.0], 1.0)];
        // A second, unrelated emitter far away.
        let two = [make([0.0, 0.0, 0.0], 1.0), make([10.0, 0.0, 0.0], 1.0)];

        let s1 = sample_lights(&one, &materials, p, 0.1, Vec2::splat(0.5)).unwrap();
        let s2 = sample_lights(&two, &materials, p, 0.1, Vec2::splat(0.5)).unwrap();
        assert!(
            (s2.pdf / s1.pdf - 0.5).abs() < 1e-4,
            "adding a second light should halve the first one's density, got {} vs {}",
            s2.pdf,
            s1.pdf
        );
    }

    /// **Splitting an emitter must change nothing.**
    ///
    /// One light of area `A` and two abutting lights of area `A/2` describe the
    /// same physical emitter, so the sampling density over its surface must be
    /// identical: `1 / (1 * A)` versus `1 / (2 * A/2)`.
    ///
    /// This is the invariant with physical meaning, and the reason the light
    /// count and the per-light area have to appear together in the pdf. Getting
    /// only one of them right produces a scene whose brightness depends on how
    /// its geometry happens to be tessellated — which is exactly what would
    /// happen to every quad light the moment it became two triangles.
    #[test]
    fn splitting_an_emitter_does_not_change_its_density() {
        let materials = vec![GpuMaterial::emissive(Vec3::ONE)];
        let quad = |origin: [f32; 3], width: f32| {
            let eu = Vec3::new(width, 0.0, 0.0);
            let ev = Vec3::new(0.0, 0.0, 2.0);
            let cross = eu.cross(ev);
            GpuLight {
                origin,
                area: cross.length(),
                edge_u: eu.to_array(),
                kind: LIGHT_KIND_QUAD,
                edge_v: ev.to_array(),
                material: 0,
                normal: cross.normalize().to_array(),
                _pad0: 0.0,
            }
        };
        // One emitter spanning x in [0, 2], versus two spanning [0, 1] and [1, 2].
        let whole = [quad([0.0, 0.0, 0.0], 2.0)];
        let halves = [quad([0.0, 0.0, 0.0], 1.0), quad([1.0, 0.0, 0.0], 1.0)];

        let p = Vec3::new(1.0, -3.0, 1.0);
        // Aim both at the same physical point, the centre of the left half.
        let a = sample_lights(&whole, &materials, p, 0.0, Vec2::new(0.25, 0.5)).unwrap();
        let b = sample_lights(&halves, &materials, p, 0.0, Vec2::new(0.5, 0.5)).unwrap();

        assert!(
            (a.wi - b.wi).length() < 1e-5,
            "the two sampling schemes did not reach the same point: {} vs {}",
            a.wi,
            b.wi
        );
        assert!(
            (a.pdf / b.pdf - 1.0).abs() < 1e-4,
            "splitting the emitter changed its sampling density: {} vs {}",
            a.pdf,
            b.pdf
        );
    }

    /// `light_pdf` must reproduce what `sample_lights` reported, or the MIS
    /// weights at build step 9 are computed against a different density than the
    /// one that generated the sample.
    #[test]
    fn light_pdf_agrees_with_the_sampler() {
        let def = scenes::cornell_box();
        let lights = build_lights(&def.scene.blob);
        let materials = &def.scene.blob.materials;
        let mut rng = Rng::new(51, 0, 0);

        for _ in 0..20_000 {
            let shading_point = Vec3::new(
                rng.next_f32() * 555.0,
                rng.next_f32() * 400.0,
                rng.next_f32() * 555.0,
            );
            let u_select = rng.next_f32();
            let u_area = rng.next_vec2();
            let Some(s) = sample_lights(&lights, materials, shading_point, u_select, u_area) else {
                continue;
            };
            let hit_point = shading_point + s.wi * s.distance;
            let normal = Vec3::from_array(lights[s.light_index as usize].normal);
            let p = light_pdf(&lights, s.light_index, shading_point, hit_point, normal);
            assert!(
                (p - s.pdf).abs() <= 1e-3 * s.pdf,
                "light_pdf {p} disagrees with the sampler's {}",
                s.pdf
            );
        }
    }
}
