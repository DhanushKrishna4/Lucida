// Interpolating vertex attributes at a triangle hit.
//
// Split out from bvh.wgsl because it is the only part of triangle handling
// that needs the `vertex_attrs` binding. A kernel that only traverses — the
// wavefront's occlusion stage, which never shades anything — would otherwise
// have to declare a binding it never reads, and WebGPU's eight-storage-buffer
// guarantee makes that a real cost rather than a tidiness question.
//
// Requires bvh.wgsl for TriHit and tri_position.

// Interpolate vertex attributes at a triangle hit.
struct TriShading {
  normal: vec3<f32>,            // shading normal, oriented against the ray
  geometric_normal: vec3<f32>,  // facet normal, oriented against the ray
  front_face: bool,
  material: u32,
  // Surface area, for the multiple importance sampling weight when a BSDF
  // sample lands on an emissive triangle.
  area: f32,
};

fn shade_triangle(hit: TriHit, dir: vec3<f32>) -> TriShading {
  let tri = triangles[hit.triangle];
  let p0 = tri_position(tri.i0);
  let p1 = tri_position(tri.i1);
  let p2 = tri_position(tri.i2);

  // Möller–Trumbore returns (u, v) for vertices 1 and 2; vertex 0 takes the
  // remainder.
  let w = 1.0 - hit.u - hit.v;
  let shading_raw = w * vertex_attrs[tri.i0].normal
                  + hit.u * vertex_attrs[tri.i1].normal
                  + hit.v * vertex_attrs[tri.i2].normal;
  let geometric_raw = cross(p1 - p0, p2 - p0);

  var out: TriShading;
  // Half the parallelogram spanned by the two edges.
  out.area = 0.5 * length(geometric_raw);
  let geometric = normalize(geometric_raw);
  // Front/back is decided by the *geometric* normal. The interpolated one can
  // disagree near a silhouette, and letting it decide would switch one-sided
  // emission on and off across a smooth surface.
  out.front_face = dot(geometric, dir) < 0.0;
  let flip = select(-1.0, 1.0, out.front_face);
  out.normal = normalize(shading_raw) * flip;
  out.geometric_normal = geometric * flip;
  out.material = tri.material;
  return out;
}
