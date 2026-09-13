// Closest-hit over the whole scene. Requires primitives.wgsl, bvh.wgsl and
// the geometry bindings.

fn scene_intersect(ray: Ray) -> Hit {
  var hit: Hit;
  hit.valid = false;
  hit.t = T_MAX;
  hit.steps = 0u;

  // Analytic primitives: spheres and quads share one tagged array, because
  // WebGPU guarantees only eight storage buffers per stage and this renderer
  // needs all eight.
  var best_index: u32 = 0u;
  var best_t: f32 = T_MAX;

  for (var i: u32 = 0u; i < U.num_primitives; i = i + 1u) {
    let prim = primitives[i];
    var t: f32;
    if (prim.kind == PRIM_KIND_QUAD) {
      t = intersect_quad(prim, ray, T_MIN, best_t);
    } else {
      t = intersect_sphere(prim, ray, T_MIN, best_t);
    }
    if (t > 0.0) {
      best_t = t;
      best_index = i;
      hit.valid = true;
    }
  }

  // Instances, through the two-level hierarchy — and **instead of** the
  // single-level triangle path, not before it.
  //
  // Every triangle in an instanced scene belongs to some BLAS and therefore
  // lives in its mesh's object space, so traversing them directly would render
  // an extra untransformed copy of every mesh at the origin. That happened, and
  // it was invisible for an identity transform because the ghost coincided with
  // the instance.
  if (U.num_instances > 0u) {
    let ih = intersect_instances(ray.origin, ray.dir, T_MIN, best_t);
    hit.steps = ih.tri.steps;
    if (ih.tri.valid) {
      let inst = read_instance(ih.instance);
      // Shade in object space, then bring the normals back. The position needs
      // no bringing back: t is the same in both spaces, so the world ray
      // evaluated there is already it.
      let o = instance_transform_point(inst, ray.origin);
      let d = instance_transform_vector(inst, ray.dir);
      let sh = shade_triangle(ih.tri, d);
      var n = instance_transform_normal(inst, sh.normal);
      var gn = instance_transform_normal(inst, sh.geometric_normal);
      // `shade_triangle` already oriented these against the object-space ray,
      // which is the right call for any transform that does not mirror. A
      // mirroring transform flips the winding, and that is decided in world
      // space below.
      let front = dot(gn, ray.dir) < 0.0;
      let flip = select(-1.0, 1.0, front);
      hit.valid = true;
      hit.t = ih.tri.t;
      hit.position = ray.origin + ih.tri.t * ray.dir;
      hit.normal = n * flip;
      hit.geometric_normal = gn * flip;
      hit.front_face = front;
      hit.material = select(sh.material, inst.material, inst.material != 0xffffffffu);
      // Instanced triangles are not in the light list: `build_lights` flattens
      // emissive geometry in world space and these live in object space, so
      // there is no light-sampling density for MIS to balance against.
      hit.light_area = 0.0;
      return hit;
    }
    if (!hit.valid) {
      return hit;
    }
    return shade_analytic(best_index, best_t, ray, hit);
  }

  // Triangles, through the BVH. Analytic primitives stay brute force: there are
  // a handful of them, and mixing primitive types into one hierarchy would need
  // a type tag in every leaf.
  let tri = intersect_triangles(ray.origin, ray.dir, T_MIN, best_t);
  hit.steps = tri.steps;
  if (tri.valid) {
    let sh = shade_triangle(tri, ray.dir);
    hit.valid = true;
    hit.t = tri.t;
    hit.position = ray.origin + tri.t * ray.dir;
    hit.normal = sh.normal;
    hit.geometric_normal = sh.geometric_normal;
    hit.front_face = sh.front_face;
    hit.material = sh.material;
    hit.light_area = sh.area;
    return hit;
  }

  if (!hit.valid) {
    return hit;
  }

  return shade_analytic(best_index, best_t, ray, hit);
}

// Fill in a hit for the closest analytic primitive.
//
// Shared by the single-level and instanced paths, which both end here: the
// analytic primitives sit outside the hierarchy either way.
fn shade_analytic(best_index: u32, best_t: f32, ray: Ray, hit_in: Hit) -> Hit {
  var hit = hit_in;
  let position = ray.origin + best_t * ray.dir;
  let prim = primitives[best_index];
  var geom_normal: vec3<f32>;
  if (prim.kind == PRIM_KIND_QUAD) {
    hit.light_area = length(cross(prim.edge_u, prim.edge_v));
    geom_normal = prim.normal;
  } else {
    // Spheres are not in the light list, so there is no light-sampling density
    // for the MIS weight to balance against.
    hit.light_area = 0.0;
    // Divide by the stored radius rather than normalising: exact for a point
    // that is genuinely on the sphere, and it matches the CPU tracer bit for bit.
    geom_normal = (position - prim.position) / prim.radius;
  }
  let material = prim.material;

  let front_face = dot(geom_normal, ray.dir) < 0.0;
  let oriented = select(-geom_normal, geom_normal, front_face);
  hit.t = best_t;
  hit.position = position;
  // Analytic primitives have an exact normal, so shading and geometric coincide.
  hit.normal = oriented;
  hit.geometric_normal = oriented;
  hit.front_face = front_face;
  hit.material = material;
  return hit;
}
