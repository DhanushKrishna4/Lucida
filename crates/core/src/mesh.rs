//! Triangle meshes and procedural generators.
//!
//! Meshes are generated rather than loaded from files at this stage, on purpose:
//! no assets to commit, deterministic across machines, and the polygon count is
//! a dial — which is exactly what a BVH needs to be tested against. Loading real
//! geometry arrives with the asset pipeline; nothing here assumes procedural
//! origin.
//!
//! Geometry is **indexed**: a shared vertex list plus triangles referencing it.
//! A closed mesh shares each vertex between roughly six triangles, so indexing
//! is about a third the memory of storing corners inline, and it is what lets a
//! subdivided sphere have genuinely smooth normals.

use crate::gpu_layout::{GpuTriangle, GpuVertexAttr, SceneBlob};
use glam::{Vec2, Vec3};
use std::collections::HashMap;

/// A mesh under construction, before it is flattened into a [`SceneBlob`].
#[derive(Clone, Debug, Default)]
pub struct MeshBuilder {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<Vec2>,
    /// Triples of indices into the vertex arrays.
    pub indices: Vec<[u32; 3]>,
}

impl MeshBuilder {
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }
    pub fn triangle_count(&self) -> usize {
        self.indices.len()
    }

    pub fn push_vertex(&mut self, position: Vec3, normal: Vec3, uv: Vec2) -> u32 {
        self.positions.push(position);
        self.normals.push(normal);
        self.uvs.push(uv);
        (self.positions.len() - 1) as u32
    }

    /// Append into a scene blob, assigning `material` to every triangle.
    ///
    /// Vertex indices are rebased onto the blob's existing vertex array, so
    /// several meshes can share one buffer.
    pub fn append_to(&self, blob: &mut SceneBlob, material: u32) {
        let base = blob.positions.len() as u32;
        for i in 0..self.positions.len() {
            let p = self.positions[i];
            blob.positions.push([p.x, p.y, p.z, 0.0]);
            blob.vertex_attrs.push(GpuVertexAttr {
                normal: self.normals[i].to_array(),
                uv: self.uvs[i].to_array(),
                ..Default::default()
            });
        }
        for t in &self.indices {
            blob.triangles.push(GpuTriangle {
                i0: base + t[0],
                i1: base + t[1],
                i2: base + t[2],
                material,
            });
        }
    }

    /// Apply an affine transform, taking normals through the inverse transpose.
    ///
    /// Normals are covectors: under a non-uniform scale they do **not** transform
    /// like positions. Scaling x by 2 and applying that same scale to a normal
    /// tilts it the wrong way — it must be scaled by 1/2 in x instead, which is
    /// what the inverse transpose does. This is the same rule that will govern
    /// instance transforms at build step 15, so it is written once here.
    pub fn transformed(mut self, transform: glam::Mat4) -> Self {
        let normal_matrix = glam::Mat3::from_mat4(transform).inverse().transpose();
        for p in &mut self.positions {
            *p = transform.transform_point3(*p);
        }
        for n in &mut self.normals {
            *n = (normal_matrix * *n).normalize_or_zero();
        }
        self
    }
}

/// A parallelogram as two triangles.
///
/// Geometrically *identical* to [`crate::gpu_layout::GpuQuad`] with the same
/// corners — which makes it the sharpest available test of the triangle path:
/// a Cornell box built from these must render the same image as one built from
/// analytic quads, to within floating-point noise.
pub fn quad(origin: Vec3, edge_u: Vec3, edge_v: Vec3) -> MeshBuilder {
    let n = edge_u.cross(edge_v).normalize();
    let mut m = MeshBuilder::default();
    m.push_vertex(origin, n, Vec2::new(0.0, 0.0));
    m.push_vertex(origin + edge_u, n, Vec2::new(1.0, 0.0));
    m.push_vertex(origin + edge_u + edge_v, n, Vec2::new(1.0, 1.0));
    m.push_vertex(origin + edge_v, n, Vec2::new(0.0, 1.0));
    // Both triangles wound so that cross(e1, e2) agrees with `n`.
    m.indices.push([0, 1, 2]);
    m.indices.push([0, 2, 3]);
    m
}

/// An axis-aligned box with flat (per-face) normals.
///
/// Each face gets its own four vertices rather than sharing eight corners: a
/// shared corner would have to average three face normals and round the edges,
/// which is wrong for a box.
pub fn cuboid(min: Vec3, max: Vec3) -> MeshBuilder {
    let mut m = MeshBuilder::default();
    let faces: [(Vec3, Vec3, Vec3); 6] = [
        // origin, edge_u, edge_v — ordered so each normal points outward.
        (
            Vec3::new(min.x, min.y, min.z),
            Vec3::new(0.0, 0.0, max.z - min.z),
            Vec3::new(0.0, max.y - min.y, 0.0),
        ), // -x
        (
            Vec3::new(max.x, min.y, min.z),
            Vec3::new(0.0, max.y - min.y, 0.0),
            Vec3::new(0.0, 0.0, max.z - min.z),
        ), // +x
        (
            Vec3::new(min.x, min.y, min.z),
            Vec3::new(max.x - min.x, 0.0, 0.0),
            Vec3::new(0.0, 0.0, max.z - min.z),
        ), // -y
        (
            Vec3::new(min.x, max.y, min.z),
            Vec3::new(0.0, 0.0, max.z - min.z),
            Vec3::new(max.x - min.x, 0.0, 0.0),
        ), // +y
        (
            Vec3::new(min.x, min.y, min.z),
            Vec3::new(0.0, max.y - min.y, 0.0),
            Vec3::new(max.x - min.x, 0.0, 0.0),
        ), // -z
        (
            Vec3::new(min.x, min.y, max.z),
            Vec3::new(max.x - min.x, 0.0, 0.0),
            Vec3::new(0.0, max.y - min.y, 0.0),
        ), // +z
    ];
    for (o, eu, ev) in faces {
        let face = quad(o, eu, ev);
        let base = m.vertex_count() as u32;
        for i in 0..face.vertex_count() {
            m.push_vertex(face.positions[i], face.normals[i], face.uvs[i]);
        }
        for t in &face.indices {
            m.indices.push([base + t[0], base + t[1], base + t[2]]);
        }
    }
    m
}

/// A geodesic sphere: an icosahedron with each triangle subdivided `levels`
/// times and every vertex projected onto the sphere.
///
/// Triangle count is `20 * 4^levels`, so the knob spans 20 to ~1.3 M triangles —
/// which is what makes it useful as a BVH stress test as well as geometry.
///
/// An icosahedron base rather than a UV sphere: a UV sphere crowds vertices at
/// the poles and stretches them at the equator, giving wildly uneven triangle
/// areas. A BVH built over those is not representative of real geometry, and the
/// poles would dominate any traversal measurement.
pub fn icosphere(center: Vec3, radius: f32, levels: u32) -> MeshBuilder {
    // Golden-ratio icosahedron: 12 vertices at the corners of three orthogonal
    // golden rectangles.
    let t = (1.0 + 5.0f32.sqrt()) / 2.0;
    let base: [Vec3; 12] = [
        Vec3::new(-1.0, t, 0.0),
        Vec3::new(1.0, t, 0.0),
        Vec3::new(-1.0, -t, 0.0),
        Vec3::new(1.0, -t, 0.0),
        Vec3::new(0.0, -1.0, t),
        Vec3::new(0.0, 1.0, t),
        Vec3::new(0.0, -1.0, -t),
        Vec3::new(0.0, 1.0, -t),
        Vec3::new(t, 0.0, -1.0),
        Vec3::new(t, 0.0, 1.0),
        Vec3::new(-t, 0.0, -1.0),
        Vec3::new(-t, 0.0, 1.0),
    ];
    #[rustfmt::skip]
    let base_faces: [[u32; 3]; 20] = [
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ];

    let mut unit: Vec<Vec3> = base.iter().map(|v| v.normalize()).collect();
    let mut faces: Vec<[u32; 3]> = base_faces.to_vec();

    for _ in 0..levels {
        // Cache midpoints by edge so adjacent triangles share the new vertex.
        // Without this the mesh becomes a soup of unshared vertices, normals
        // stop being smooth, and the vertex count explodes by 4x per level.
        let mut midpoints: HashMap<(u32, u32), u32> = HashMap::new();
        let mut next = Vec::with_capacity(faces.len() * 4);

        let midpoint =
            |a: u32, b: u32, unit: &mut Vec<Vec3>, cache: &mut HashMap<(u32, u32), u32>| -> u32 {
                let key = if a < b { (a, b) } else { (b, a) };
                if let Some(&i) = cache.get(&key) {
                    return i;
                }
                let m = ((unit[a as usize] + unit[b as usize]) * 0.5).normalize();
                unit.push(m);
                let i = (unit.len() - 1) as u32;
                cache.insert(key, i);
                i
            };

        for f in &faces {
            let a = midpoint(f[0], f[1], &mut unit, &mut midpoints);
            let b = midpoint(f[1], f[2], &mut unit, &mut midpoints);
            let c = midpoint(f[2], f[0], &mut unit, &mut midpoints);
            next.push([f[0], a, c]);
            next.push([f[1], b, a]);
            next.push([f[2], c, b]);
            next.push([a, b, c]);
        }
        faces = next;
    }

    let mut m = MeshBuilder::default();
    for &u in &unit {
        // On a unit sphere the position *is* the normal, which is why the
        // shading normal here is exact rather than an average of face normals.
        // Equirectangular UVs; the seam at u = 0/1 is not stitched, which is
        // harmless until textures arrive.
        let uv = Vec2::new(
            0.5 + u.z.atan2(u.x) / std::f32::consts::TAU,
            0.5 - u.y.asin() / std::f32::consts::PI,
        );
        m.push_vertex(center + u * radius, u, uv);
    }
    m.indices = faces;
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quad_has_two_triangles_and_a_consistent_normal() {
        let m = quad(Vec3::ZERO, Vec3::X, Vec3::Y);
        assert_eq!(m.triangle_count(), 2);
        assert_eq!(m.vertex_count(), 4);
        for n in &m.normals {
            assert!((*n - Vec3::Z).length() < 1e-6, "normal {n} should be +Z");
        }
        // Both triangles must wind the same way as the stored normal.
        for t in &m.indices {
            let (a, b, c) = (
                m.positions[t[0] as usize],
                m.positions[t[1] as usize],
                m.positions[t[2] as usize],
            );
            let geo = (b - a).cross(c - a).normalize();
            assert!(geo.dot(Vec3::Z) > 0.99, "triangle {t:?} is wound backwards");
        }
    }

    #[test]
    fn cuboid_is_closed_and_outward_facing() {
        let m = cuboid(Vec3::splat(-1.0), Vec3::splat(1.0));
        assert_eq!(m.triangle_count(), 12);
        for t in &m.indices {
            let (a, b, c) = (
                m.positions[t[0] as usize],
                m.positions[t[1] as usize],
                m.positions[t[2] as usize],
            );
            let centroid = (a + b + c) / 3.0;
            let geo = (b - a).cross(c - a).normalize();
            // The box is centred on the origin, so "outward" is "away from it".
            assert!(
                geo.dot(centroid.normalize()) > 0.5,
                "face at {centroid} faces inward (normal {geo})"
            );
        }
    }

    #[test]
    fn icosphere_counts_are_exact() {
        for levels in 0..5u32 {
            let m = icosphere(Vec3::ZERO, 1.0, levels);
            assert_eq!(m.triangle_count(), 20 * 4usize.pow(levels));
            // Euler's formula for a closed triangulated sphere: V - E + F = 2,
            // and E = 3F/2, so V = F/2 + 2. Getting this right is precisely the
            // check that the midpoint cache is sharing vertices; without it the
            // vertex count would be 3F.
            assert_eq!(m.vertex_count(), m.triangle_count() / 2 + 2);
        }
    }

    #[test]
    fn icosphere_vertices_lie_on_the_sphere() {
        let m = icosphere(Vec3::new(5.0, -2.0, 1.0), 3.0, 3);
        for (i, p) in m.positions.iter().enumerate() {
            let r = (*p - Vec3::new(5.0, -2.0, 1.0)).length();
            assert!(
                (r - 3.0).abs() < 1e-4,
                "vertex {i} at radius {r}, expected 3"
            );
            // On a sphere the normal is the outward radial direction.
            let expected = (*p - Vec3::new(5.0, -2.0, 1.0)).normalize();
            assert!((m.normals[i] - expected).length() < 1e-5);
        }
    }

    /// Subdividing must converge to the true sphere area, 4*pi*r^2. This checks
    /// that subdivision is actually projecting onto the sphere rather than just
    /// splitting flat triangles.
    #[test]
    fn icosphere_area_converges() {
        let area = |levels: u32| -> f32 {
            let m = icosphere(Vec3::ZERO, 1.0, levels);
            m.indices
                .iter()
                .map(|t| {
                    let (a, b, c) = (
                        m.positions[t[0] as usize],
                        m.positions[t[1] as usize],
                        m.positions[t[2] as usize],
                    );
                    0.5 * (b - a).cross(c - a).length()
                })
                .sum()
        };
        let exact = 4.0 * std::f32::consts::PI;
        let mut prev_err = f32::INFINITY;
        for levels in 0..5 {
            let err = (area(levels) - exact).abs();
            assert!(err < prev_err, "area error grew at level {levels}");
            prev_err = err;
        }
        // At level 4 the polyhedron should be within a fraction of a percent.
        assert!(prev_err / exact < 2e-3, "level 4 area error is {prev_err}");
    }

    /// Normals must go through the inverse transpose, not the transform itself.
    #[test]
    fn non_uniform_scale_transforms_normals_correctly() {
        // A 45-degree plane in the x-y plane: normal is (1, 1, 0)/sqrt(2).
        let mut m = MeshBuilder::default();
        m.push_vertex(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0).normalize(),
            Vec2::ZERO,
        );
        m.push_vertex(
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0).normalize(),
            Vec2::ZERO,
        );
        m.push_vertex(
            Vec3::new(0.0, 1.0, 1.0),
            Vec3::new(1.0, 1.0, 0.0).normalize(),
            Vec2::ZERO,
        );
        m.indices.push([0, 1, 2]);

        // Squash x by 1/2. The surface tilts one way; a naively transformed
        // normal would tilt the other.
        let m = m.transformed(glam::Mat4::from_scale(Vec3::new(0.5, 1.0, 1.0)));

        let (a, b, c) = (m.positions[0], m.positions[1], m.positions[2]);
        let geometric = (b - a).cross(c - a).normalize();
        let shading = m.normals[0];
        let agreement = geometric.dot(shading).abs();
        assert!(
            agreement > 0.999,
            "transformed normal {shading} disagrees with the transformed surface {geometric} \
             (dot = {agreement}) — the inverse transpose is not being applied"
        );
    }
}
