// EXTEND: closest-hit for every live path.
//
// The stage the whole architecture exists to isolate. It is the most expensive
// and by far the most register-hungry kernel — BVH traversal carries a 32-entry
// stack in private memory — and in a megakernel that register pressure is paid
// by *every* thread, including ones that are only shading. Here nothing else
// shares its occupancy.
//
// It writes a **complete** hit record, including interpolated shading normals,
// so SHADE needs no geometry bindings at all. That split is what lets both
// kernels fit inside the eight-storage-buffer limit.

//!include "common/generated.wgsl"

@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read> primitives: array<Primitive>;
@group(0) @binding(2) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> vertex_attrs: array<VertexAttr>;
@group(0) @binding(4) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(5) var<storage, read> bvh_nodes: array<BvhNode>;

@group(1) @binding(0) var<storage, read> paths: array<PathState>;
@group(1) @binding(1) var<storage, read_write> hits: array<HitRecord>;
@group(1) @binding(2) var<storage, read> queue_in: array<u32>;

//!include "common/math.wgsl"
//!include "common/ray.wgsl"
//!include "common/bvh.wgsl"
//!include "common/triangle_shading.wgsl"
//!include "common/primitives.wgsl"
//!include "common/scene.wgsl"

const INVALID_PATH: u32 = 0xFFFFFFFFu;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  // No queue-length binding: the queue is sentinel-terminated instead, which
  // costs 63 writes in RESET and saves this kernel one of its eight slots.
  if (gid.x >= arrayLength(&queue_in)) {
    return;
  }
  let path_index = queue_in[gid.x];
  var rec: HitRecord;
  rec.valid = 0u;
  if (path_index == INVALID_PATH) {
    hits[gid.x] = rec;
    return;
  }

  let p = paths[path_index];
  var ray: Ray;
  ray.origin = p.origin;
  ray.dir = p.direction;
  let hit = scene_intersect(ray);

  rec.valid = select(0u, 1u, hit.valid);
  rec.t = hit.t;
  rec.position = hit.position;
  rec.normal = hit.normal;
  rec.geometric_normal = hit.geometric_normal;
  rec.material = hit.material;
  rec.light_area = hit.light_area;
  rec.front_face = select(0u, 1u, hit.front_face);
  // Indexed by queue slot, not path index: SHADE walks the same queue, so the
  // two stages read memory in the same order rather than scattering through a
  // sparse array.
  hits[gid.x] = rec;
}
