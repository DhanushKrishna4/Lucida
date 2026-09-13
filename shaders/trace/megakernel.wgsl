// Megakernel path tracer: one thread per pixel, the whole path in one shader.
//
// This is deliberately the *simple* architecture. It is known to perform badly —
// divergent material evaluation wrecks occupancy, and register pressure from the
// heaviest branch is paid by every thread — and it gets replaced by the
// wavefront architecture at build step 10. It stays behind a flag after that, as
// the baseline the wavefront version is measured against.
//
// Right now its job is correctness, not speed: it must reproduce the CPU
// reference tracer's output, and it is written line-for-line against
// `crates/core/src/integrator.rs` to make any divergence a bug in one specific
// place rather than an architectural difference.

//!include "common/generated.wgsl"
//!include "common/math.wgsl"
//!include "common/rng.wgsl"

// Exactly eight storage buffers. WebGPU guarantees eight per shader stage, so
// there is no room for a ninth — which is why BVH leaves index the triangle
// array directly instead of going through a primitive-index table.
@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read> materials: array<Material>;
// Spheres and quads share one tagged array so the freed binding can carry the
// light list: WebGPU guarantees only eight storage buffers per shader stage.
@group(0) @binding(2) var<storage, read> primitives: array<Primitive>;
@group(0) @binding(3) var<storage, read> lights: array<Light>;
// xyz = sum of radiance estimates, w = number of samples accumulated.
@group(0) @binding(4) var<storage, read_write> accum: array<Accum>;
// Positions are separate from the other vertex attributes because they are hot:
// every leaf test reads three of them, while normals and UVs are read once, at
// the closest hit.
@group(0) @binding(5) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> vertex_attrs: array<VertexAttr>;
@group(0) @binding(7) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(8) var<storage, read> bvh_nodes: array<BvhNode>;

//!include "common/bsdf.wgsl"
//!include "common/bvh.wgsl"
//!include "common/ray.wgsl"
//!include "common/triangle_shading.wgsl"
//!include "common/primitives.wgsl"
//!include "common/scene.wgsl"
//!include "common/envmap.wgsl"
//!include "common/light.wgsl"
//!include "common/camera.wgsl"

// Unidirectional path tracing, BSDF sampling only.
//
// No next event estimation (build step 8), no MIS (step 9), no Russian roulette
// (step 9). Paths are truncated at `max_depth`, which is a small deterministic
// bias that both devices share exactly.
// What the denoiser needs to know about the first hit: a noise-free description
// of the surface the pixel is looking at.
struct Guide {
  albedo: vec3<f32>,
  normal: vec3<f32>,
  depth: f32,
  steps: f32,
};

fn trace_path(ray_in: Ray, rng: ptr<function, Rng>, guide: ptr<function, Guide>) -> vec3<f32> {
  var radiance = vec3<f32>(0.0, 0.0, 0.0);
  var throughput = vec3<f32>(1.0, 1.0, 1.0);
  var ray = ray_in;

  // Carried across iterations for the MIS weight: to decide how much credit
  // BSDF sampling deserves for landing on an emitter, we need the density it
  // used and the vertex it started from.
  var prev_bsdf_pdf = 0.0;
  var prev_position = ray_in.origin;

  for (var depth: u32 = 0u; depth < U.max_depth; depth = depth + 1u) {
    let hit = scene_intersect(ray);
    if (!hit.valid) {
      // The environment. A *constant* background is only ever found by BSDF
      // sampling. An environment **map** is importance-sampled like any other
      // light, so arriving here by BSDF sampling has to share credit with the
      // connection that could have found the same direction — skipping that
      // weight double-counts the sky.
      if (env_present()) {
        var w = 1.0;
        if (depth > 0u) {
          if (U.sampling_mode == SAMPLING_NEE_ONLY) {
            w = 0.0;
          } else if (U.sampling_mode == SAMPLING_MIS) {
            let pdf_env = env_pdf(ray.dir) / f32(max(light_strategy_count(), 1u));
            w = power_heuristic(prev_bsdf_pdf, pdf_env);
          }
        }
        if (w > 0.0) {
          radiance = radiance + throughput * env_radiance_at(ray.dir) * w;
        }
      } else {
        radiance = radiance + throughput * U.background;
      }
      break;
    }

    let m = materials[hit.material];

    // Record what the first hit is looking at, once, for the denoiser.
    //
    // The *first* hit specifically: the guides describe the surface this pixel
    // shows, and a later bounce describes somewhere else entirely. Emission is
    // added to the albedo so that a light fixture is not demodulated into a
    // blinding division by a near-black base colour.
    if (depth == 0u) {
      let surf0 = surface_from_material(m, hit.front_face);
      (*guide).albedo = max(
        surf0.diffuse_albedo + surf0.f0 + m.emissive,
        vec3<f32>(1.0e-3),
      );
      (*guide).normal = hit.normal;
      (*guide).depth = hit.t;
      (*guide).steps = f32(hit.steps);
    }

    // One-sided emission: only the face the stored normal points from emits.
    if (hit.front_face && max(m.emissive.x, max(m.emissive.y, m.emissive.z)) > 0.0) {
      // depth == 0 is special in every mode: no shadow ray preceded the camera
      // ray, so a directly visible emitter is found only this way and takes full
      // credit. Forgetting it renders the light fixture black while the room it
      // lights looks perfect.
      var weight = 1.0;
      if (depth > 0u) {
        if (U.sampling_mode == SAMPLING_NEE_ONLY) {
          // The shadow ray from the previous vertex already accounted for this
          // emitter; counting it again doubles every light path.
          weight = 0.0;
        } else if (U.sampling_mode == SAMPLING_MIS) {
          // Share the credit: how likely was light sampling to have produced
          // this same direction from the previous vertex?
          // Scaled by the light count over the strategy count, because the
          // environment map competes for the same uniform selection draw.
          let pdf_light = light_pdf_from_geometry(
            hit.light_area, hit.geometric_normal, prev_position, hit.position)
            * (f32(U.num_lights) / f32(max(light_strategy_count(), 1u)));
          weight = power_heuristic(prev_bsdf_pdf, pdf_light);
        }
      }
      if (weight > 0.0) {
        radiance = radiance + throughput * m.emissive * weight;
      }
    }

    // --- BSDF sampling ---------------------------------------------------
    //
    // In the local shading frame, where the normal is +Z and wo points back
    // toward the previous path vertex.
    let surf = surface_from_material(m, hit.front_face);
    let wo = to_local(-ray.dir, hit.normal);
    if (wo.z <= 0.0) {
      // The shading normal disagrees with the geometric one strongly enough
      // that the viewer sits below the shading hemisphere. Terminating beats
      // producing a negative cosine and a black or exploding pixel.
      break;
    }

    // --- Next event estimation -------------------------------------------
    //
    // Only while a further bounce would still be allowed: a light connection
    // from this vertex forms a path one segment longer than stopping here, so
    // running it on the final iteration would let NEE reach paths BSDF sampling
    // cannot at the same max_depth.
    //
    // The emptiness check is part of the contract, not an optimisation — it
    // decides whether random numbers are drawn, so the CPU must make the same
    // decision or the two streams diverge.
    if (U.sampling_mode != SAMPLING_BSDF_ONLY && depth + 1u < U.max_depth && light_strategy_count() > 0u) {
    // Over lights *and* the environment map: either can be connected to, and
    // the decision must be the same on every device because it controls whether
    // random numbers are drawn.
      radiance = radiance + throughput * direct_light(
        surf, hit.position, hit.normal, hit.geometric_normal, wo, rng, U.sampling_mode);
    }

    // Draw order is part of the CPU/GPU contract: lobe choice first, then the
    // two-dimensional sample within the chosen lobe.
    let u_lobe = rng_next_f32(rng);
    let u = rng_next_vec2(rng);
    let bs = surface_sample(surf, wo, u_lobe, u);
    if (!bs.valid) {
      break;
    }

    // The weight already carries f * cos / pdf, including the combined two-lobe
    // density — see common/bsdf.wgsl.
    throughput = throughput * bs.weight;

    // A black surface absorbs. Both devices take this branch on exactly the same
    // condition, so their random streams stay in lockstep.
    if (max(throughput.x, max(throughput.y, throughput.z)) <= 0.0) {
      break;
    }

    prev_bsdf_pdf = bs.pdf;
    prev_position = hit.position;

    let dir = to_world(bs.wi, hit.normal);
    // Offset along the *geometric* normal: on a smooth mesh the interpolated
    // normal can lean far enough from the facet that offsetting along it leaves
    // the new origin below the surface.
    // Offset toward whichever side the ray leaves on: a transmitted ray goes
    // *into* the surface, and pushing it out along the outward normal leaves
    // it on the wrong side, where it re-hits the surface it just crossed.
    // That renders glass as solid black.
    let exit_normal = select(hit.geometric_normal, -hit.geometric_normal, bs.wi.z < 0.0);
    ray.origin = offset_ray_origin(hit.position, exit_normal);
    ray.dir = dir;
  }

  return radiance;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= U.width || gid.y >= U.height) {
    return;
  }
  let pixel_index = gid.y * U.width + gid.x;

  var sum = vec3<f32>(0.0, 0.0, 0.0);
  var guide_sum = vec3<f32>(0.0);
  var normal_sum = vec3<f32>(0.0);
  var depth_sum = 0.0;
  var steps_sum = 0.0;
  var sum_sq = vec3<f32>(0.0);
  for (var s: u32 = 0u; s < U.samples_per_launch; s = s + 1u) {
    var rng = rng_init(U.frame_seed, pixel_index, U.sample_offset + s);

    // Draw order is part of the CPU/GPU contract. The lens sample is consumed
    // unconditionally even for a pinhole camera, so that enabling depth of field
    // does not shift every subsequent random number.
    let pixel_uv = rng_next_vec2(&rng);
    let lens_uv = rng_next_vec2(&rng);

    let ray = generate_ray(gid.x, gid.y, pixel_uv, lens_uv);
    var g: Guide;
    g.albedo = vec3<f32>(1.0);
    g.normal = vec3<f32>(0.0);
    g.depth = 0.0;
    g.steps = 0.0;
    let r = trace_path(ray, &rng, &g);
    sum = sum + r;
    // Squares accumulated per *sample*, not from the running mean: the variance
    // wanted is of the estimator, and squaring an average would measure nothing.
    sum_sq = sum_sq + r * r;
    steps_sum = steps_sum + g.steps;
    guide_sum = guide_sum + g.albedo;
    normal_sum = normal_sum + g.normal;
    depth_sum = depth_sum + g.depth;
  }

  // Accumulate rather than overwrite: the browser renders progressively across
  // many bounded dispatches. Summing then dividing at display time (rather than
  // keeping a running mean) keeps the arithmetic identical to the CPU tracer's.
  var a = accum[pixel_index];
  a.radiance = a.radiance + sum;
  a.samples = a.samples + f32(U.samples_per_launch);
  a.albedo = a.albedo + guide_sum;
  a.normal = a.normal + normal_sum;
  a.depth = a.depth + depth_sum;
  a.traversal = a.traversal + steps_sum;
  a.radiance_sq = a.radiance_sq + sum_sq;
  accum[pixel_index] = a;
}
