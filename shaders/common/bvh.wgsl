// BVH traversal and triangle intersection. Mirrors `crates/core/src/bvh.rs`.
//
// Requires: generated.wgsl (BvhNode, Triangle, VertexAttr), math.wgsl (T_MIN /
// T_MAX), and the geometry bindings declared by the including shader.

// Reciprocal of a ray direction, guaranteed finite.
//
// WGSL leaves f32 division by zero **implementation-defined**, and an
// axis-aligned ray is not exotic — it is what a camera looking down an axis
// produces. Even where division yields infinity, the slab test then computes
// `(plane - origin) * inf`, which is `0 * inf = NaN` when the origin lies
// exactly on that plane; the NaN silently turns a hit into a miss. Clamping to a
// large finite magnitude removes both problems with no branches: `0 * 1e16` is
// zero, and no infinity ever enters the arithmetic.
//
// See the long-form explanation in crates/core/src/bvh.rs.
const INV_DIR_BIG: f32 = 1.0e16;
const INV_DIR_SMALL: f32 = 1.0e-16;

fn safe_inv_component(d: f32) -> f32 {
  // `sign` returns 0 for +/-0.0, so read the sign bit directly to keep -0.0
  // pointing the right way; otherwise the slab ordering flips.
  let s = select(-INV_DIR_BIG, INV_DIR_BIG, (bitcast<u32>(d) & 0x80000000u) == 0u);
  return select(s, 1.0 / d, abs(d) > INV_DIR_SMALL);
}

fn safe_inv_dir(d: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    safe_inv_component(d.x),
    safe_inv_component(d.y),
    safe_inv_component(d.z),
  );
}

// Slab test. Returns the entry distance, or -1.0 for a miss.
fn aabb_hit(
  bounds_min: vec3<f32>,
  bounds_max: vec3<f32>,
  origin: vec3<f32>,
  inv_dir: vec3<f32>,
  t_min: f32,
  t_max: f32,
) -> f32 {
  let t0 = (bounds_min - origin) * inv_dir;
  let t1 = (bounds_max - origin) * inv_dir;
  let near = min(t0, t1);
  let far = max(t0, t1);
  let entry = max(max(near.x, near.y), max(near.z, t_min));
  let exit = min(min(far.x, far.y), min(far.z, t_max));
  return select(-1.0, entry, entry <= exit);
}

struct TriHit {
  t: f32,
  u: f32,
  v: f32,
  triangle: u32,
  valid: bool,
  // BVH nodes visited finding this hit, for the traversal heatmap. Counted
  // always rather than behind a flag: it is one increment in a loop that is
  // already memory-bound, and a diagnostic that has to be switched on at build
  // time is a diagnostic nobody uses.
  steps: u32,
};

fn tri_position(i: u32) -> vec3<f32> {
  return positions[i].xyz;
}

// Möller–Trumbore. Double-sided (the test is on |det|, not det > 0) because a
// path tracer has to see the inside of surfaces — that is how a ray inside glass
// finds its way out.
fn intersect_triangle(
  tri: Triangle,
  origin: vec3<f32>,
  dir: vec3<f32>,
  t_min: f32,
  t_max: f32,
) -> vec3<f32> {  // (t, u, v); t < 0 means miss
  let p0 = tri_position(tri.i0);
  let e1 = tri_position(tri.i1) - p0;
  let e2 = tri_position(tri.i2) - p0;

  let pv = cross(dir, e2);
  let det = dot(e1, pv);
  if (abs(det) < 1.0e-12) {
    return vec3<f32>(-1.0, 0.0, 0.0);
  }
  let inv_det = 1.0 / det;

  let tv = origin - p0;
  let u = dot(tv, pv) * inv_det;
  if (u < 0.0 || u > 1.0) {
    return vec3<f32>(-1.0, 0.0, 0.0);
  }

  let qv = cross(tv, e1);
  let v = dot(dir, qv) * inv_det;
  if (v < 0.0 || u + v > 1.0) {
    return vec3<f32>(-1.0, 0.0, 0.0);
  }

  let t = dot(e2, qv) * inv_det;
  if (t < t_min || t > t_max) {
    return vec3<f32>(-1.0, 0.0, 0.0);
  }
  return vec3<f32>(t, u, v);
}

// Traversal stack depth.
//
// This is a real trade. The stack lives in private (per-thread) memory, and on
// most hardware that means registers; every slot costs occupancy for every
// thread, whether or not it is ever used. Workgroup memory is not an option
// either — at 16 KB shared between 64 threads there is not room for a per-thread
// stack of any useful depth.
//
// The stack only ever holds *deferred siblings*, one per level actually
// descended, so its depth is the tree's depth, not its node count. The SAH
// builder is asserted to stay under 3x log2(N) deep (see `depth_stays_logarithmic`),
// which is 19 for the 138k-triangle stress scene. 32 leaves comfortable headroom
// at half the CPU's 64.
//
// Overflow drops a subtree rather than corrupting memory: geometry would go
// missing, which is visible, instead of the traversal reading garbage.
const BVH_STACK_SIZE: u32 = 32u;

// Closest triangle hit, by BVH traversal when a BVH is present and by brute
// force otherwise. The brute-force path is not dead code — it is what the
// correctness tests compare against, selected by setting num_bvh_nodes to 0.
// Walk one BVH starting at `root`.
//
// Parameterised on the root because a two-level hierarchy needs it: a BLAS's
// nodes sit at an offset inside the shared array, and the TLAS sits after all of
// them. Leaf `left_first` values are rebased on the host, so a leaf indexes the
// shared triangle array absolutely whichever BLAS it came from.
fn intersect_bvh_from(root: u32, origin: vec3<f32>, dir: vec3<f32>, t_min: f32, t_max: f32) -> TriHit {
  var hit: TriHit;
  hit.valid = false;
  hit.t = t_max;
  hit.steps = 0u;
  var closest = t_max;

  if (U.num_bvh_nodes == 0u) {
    for (var i: u32 = 0u; i < U.num_triangles; i = i + 1u) {
      let r = intersect_triangle(triangles[i], origin, dir, t_min, closest);
      if (r.x > 0.0) {
        closest = r.x;
        hit.t = r.x;
        hit.u = r.y;
        hit.v = r.z;
        hit.triangle = i;
        hit.valid = true;
      }
    }
    return hit;
  }

  let inv_dir = safe_inv_dir(dir);
  var stack: array<u32, 32>;
  var sp: u32 = 0u;
  var node: u32 = root;

  loop {
    let n = bvh_nodes[node];
    hit.steps = hit.steps + 1u;

    if (n.count > 0u) {
      // Leaf. `left_first` indexes the triangle array directly: the build
      // compacts triangles into traversal order, so there is no indirection
      // through a primitive-index table and a leaf's triangles are contiguous.
      for (var i: u32 = 0u; i < n.count; i = i + 1u) {
        let ti = n.left_first + i;
        let r = intersect_triangle(triangles[ti], origin, dir, t_min, closest);
        if (r.x > 0.0) {
          closest = r.x;
          hit.t = r.x;
          hit.u = r.y;
          hit.v = r.z;
          hit.triangle = ti;
          hit.valid = true;
        }
      }
    } else {
      let c0 = n.left_first;
      let c1 = c0 + 1u;
      let a = bvh_nodes[c0];
      let b = bvh_nodes[c1];
      let d0 = aabb_hit(a.bounds_min, a.bounds_max, origin, inv_dir, t_min, closest);
      let d1 = aabb_hit(b.bounds_min, b.bounds_max, origin, inv_dir, t_min, closest);

      if (d0 >= 0.0 && d1 >= 0.0) {
        // Descend into the nearer child first. By the time the farther one is
        // popped, `closest` has often already shrunk below its entry distance
        // and the whole subtree is culled without being entered. Without the
        // ordering, half of all traversals do the work in the useless order.
        var near = c0;
        var far = c1;
        if (d1 < d0) {
          near = c1;
          far = c0;
        }
        if (sp < BVH_STACK_SIZE) {
          stack[sp] = far;
          sp = sp + 1u;
        }
        node = near;
        continue;
      } else if (d0 >= 0.0) {
        node = c0;
        continue;
      } else if (d1 >= 0.0) {
        node = c1;
        continue;
      }
    }

    if (sp == 0u) {
      break;
    }
    sp = sp - 1u;
    node = stack[sp];
  }

  return hit;
}

// The single-level entry point: one BVH over every triangle, rooted at 0.
fn intersect_triangles(origin: vec3<f32>, dir: vec3<f32>, t_min: f32, t_max: f32) -> TriHit {
  return intersect_bvh_from(0u, origin, dir, t_min, t_max);
}

// A hit found inside an instance.
struct InstanceHit {
  tri: TriHit,
  // Index into `primitives`, where the instances are appended.
  instance: u32,
};

// Read an instance out of the shared primitive array.
//
// `GpuInstance` overlays `Primitive` exactly — same size, same `kind` offset —
// so the four vec3 slots carry the rows of the world-to-object 3x3 and its
// translation, and the scalars between them carry the BLAS root, the tag and the
// material override. Forced by the binding budget: EXTEND already binds the
// eight storage buffers WebGPU guarantees, so instancing had to cost none.
struct InstanceRef {
  row0: vec3<f32>,
  row1: vec3<f32>,
  row2: vec3<f32>,
  translation: vec3<f32>,
  blas_root: u32,
  material: u32,
};

fn read_instance(i: u32) -> InstanceRef {
  let p = primitives[i];
  var out: InstanceRef;
  out.row0 = p.position;
  out.row1 = p.edge_u;
  out.row2 = p.edge_v;
  out.translation = p.normal;
  // `radius` and `kind` share the words the transform does not use.
  out.blas_root = bitcast<u32>(p.radius);
  out.material = p.material;
  return out;
}

// World -> object. The direction is deliberately **not** normalised: keeping its
// world-space scale means a hit at parameter t is at the same t in both spaces,
// so t values from different instances are comparable and the traversal's t_max
// culling keeps working across levels. Normalise it and scaled instances sort
// incorrectly against one another.
fn instance_transform_point(inst: InstanceRef, p: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(dot(inst.row0, p), dot(inst.row1, p), dot(inst.row2, p))
       + inst.translation;
}

fn instance_transform_vector(inst: InstanceRef, v: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(dot(inst.row0, v), dot(inst.row1, v), dot(inst.row2, v));
}

// Object -> world for a normal, by the inverse-transpose. The stored matrix is
// already the inverse, so this is just its transpose — no inversion needed.
//
// Using the matrix itself is the classic error and is invisible until something
// is non-uniformly scaled, because under rotation and uniform scale the two
// agree up to a length the renormalisation removes.
fn instance_transform_normal(inst: InstanceRef, n: vec3<f32>) -> vec3<f32> {
  let t = vec3<f32>(
    inst.row0.x * n.x + inst.row1.x * n.y + inst.row2.x * n.z,
    inst.row0.y * n.x + inst.row1.y * n.y + inst.row2.y * n.z,
    inst.row0.z * n.x + inst.row1.z * n.y + inst.row2.z * n.z,
  );
  return normalize(t);
}

// Two-level traversal: walk the TLAS in world space, and at each instance it
// reaches, transform the ray and walk that instance's BLAS.
//
// `closest` is threaded through both levels unchanged, which is only sound
// because the transform preserves t. It is also what makes the second level
// cheap: an instance whose bounds start beyond the closest hit so far is
// rejected at the TLAS without transforming anything.
fn intersect_instances(origin: vec3<f32>, dir: vec3<f32>, t_min: f32, t_max: f32) -> InstanceHit {
  var out: InstanceHit;
  out.tri.valid = false;
  out.tri.t = t_max;
  out.tri.steps = 0u;
  out.instance = 0u;
  // Both levels count: a heatmap that showed only the BLAS would hide a badly
  // built TLAS entirely.
  var steps = 0u;
  if (U.num_instances == 0u) {
    return out;
  }
  var closest = t_max;

  let inv_dir = safe_inv_dir(dir);
  var stack: array<u32, 32>;
  var sp: u32 = 0u;
  var node: u32 = U.tlas_root;

  loop {
    let n = bvh_nodes[node];
    steps = steps + 1u;
    if (n.count > 0u) {
      // Leaf. `left_first` indexes `primitives`, where the instances are
      // appended after the analytic ones — rebased on the host so no
      // indirection table is needed.
      for (var i: u32 = 0u; i < n.count; i = i + 1u) {
        let ii = n.left_first + i;
        let inst = read_instance(ii);
        let o = instance_transform_point(inst, origin);
        let d = instance_transform_vector(inst, dir);
        let h = intersect_bvh_from(inst.blas_root, o, d, t_min, closest);
        steps = steps + h.steps;
        if (h.valid) {
          closest = h.t;
          out.tri = h;
          out.instance = ii;
        }
      }
    } else {
      let c0 = n.left_first;
      let c1 = c0 + 1u;
      let a = bvh_nodes[c0];
      let b = bvh_nodes[c1];
      let d0 = aabb_hit(a.bounds_min, a.bounds_max, origin, inv_dir, t_min, closest);
      let d1 = aabb_hit(b.bounds_min, b.bounds_max, origin, inv_dir, t_min, closest);
      if (d0 >= 0.0 && d1 >= 0.0) {
        var near = c0;
        var far = c1;
        if (d1 < d0) {
          near = c1;
          far = c0;
        }
        if (sp < BVH_STACK_SIZE) {
          stack[sp] = far;
          sp = sp + 1u;
        }
        node = near;
        continue;
      } else if (d0 >= 0.0) {
        node = c0;
        continue;
      } else if (d1 >= 0.0) {
        node = c1;
        continue;
      }
    }
    if (sp == 0u) {
      break;
    }
    sp = sp - 1u;
    node = stack[sp];
  }
  out.tri.steps = steps;
  return out;
}
