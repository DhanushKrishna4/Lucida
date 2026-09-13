// CONNECT: resolve deferred shadow rays.
//
// The smallest kernel in the pipeline, deliberately. SHADE already computed each
// ray's contribution, so this only answers a visibility question — no BSDF, no
// material, no light sampling. Its register footprint is the traversal stack and
// little else, which is the best occupancy any stage here gets.
//
// It needs no vertex attributes: occlusion never shades a surface. That is worth
// a binding, and bindings are the scarce resource.

//!include "common/generated.wgsl"

@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read> primitives: array<Primitive>;
@group(0) @binding(2) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(4) var<storage, read> bvh_nodes: array<BvhNode>;

@group(1) @binding(0) var<storage, read> shadow_rays: array<ShadowRay>;
@group(1) @binding(1) var<storage, read_write> paths: array<PathState>;
// `read_write`, even though this kernel only reads `shadow_count`.
//
// WGSL requires any storage variable whose type *contains* an atomic to be
// read_write, regardless of what the shader does with it, and WavefrontCounters
// holds three. Chrome enforces this; naga (and so the native harness) does not,
// which is how a shader that passes `cargo test` failed to compile in the
// browser. Worth remembering: the native harness catches light-transport bugs,
// not every WGSL conformance difference.
@group(1) @binding(2) var<storage, read_write> counters: WavefrontCounters;

//!include "common/math.wgsl"
//!include "common/ray.wgsl"
//!include "common/bvh.wgsl"
//!include "common/primitives.wgsl"

fn shadow_ray_blocked(origin: vec3<f32>, wi: vec3<f32>, distance: f32) -> bool {
  // Shortened relative to the distance, not by a fixed epsilon, so the emitter
  // does not count as its own occluder at any scene scale.
  let t_max = distance * (1.0 - 1.0e-3);

  for (var i: u32 = 0u; i < U.num_primitives; i = i + 1u) {
    let prim = primitives[i];
    var t: f32;
    if (prim.kind == PRIM_KIND_QUAD) {
      t = intersect_quad(prim, Ray(origin, wi), T_MIN, t_max);
    } else {
      t = intersect_sphere(prim, Ray(origin, wi), T_MIN, t_max);
    }
    if (t > 0.0) {
      return true;
    }
  }
  // Any-hit: return on the first blocker, and never narrow t_max — a shadow ray
  // only needs to know *whether* something is in the way.
  return intersect_triangles(origin, wi, T_MIN, t_max).valid;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  // The snapshot, not the atomic: RESET clears the atomic for the next bounce
  // before this kernel runs.
  if (gid.x >= counters.shadow_count) {
    return;
  }
  let sr = shadow_rays[gid.x];
  if (shadow_ray_blocked(sr.origin, sr.direction, sr.distance)) {
    return;
  }
  // Added back into the path, not straight into the accumulator.
  //
  // The accumulator was indexed by pixel, which was safe only while exactly one
  // path per pixel was in flight. A batch carries several samples of the same
  // pixel at once, so two shadow rays in the same bounce could land on the same
  // element and race. Path indices stay unique — SHADE emits at most one shadow
  // ray per path per bounce — so this write needs no atomics.
  //
  // It is also the accumulation order the megakernel uses: the next bounce's
  // emission is added to `radiance` after this bounce's direct lighting, because
  // compute passes are ordered and CONNECT runs before the next EXTEND.
  //
  // Casting more than one shadow ray per bounce would break the uniqueness this
  // relies on, which is worth remembering before adding multi-light sampling.
  paths[sr.path].radiance = paths[sr.path].radiance + sr.contribution;
}
