// SHADE: emission, next event estimation, and the next BSDF sample.
//
// Reads the hit records EXTEND produced, updates path state, appends a shadow
// ray for the light connection, and appends the path to the next bounce's queue
// if it survives. Paths that die are simply not appended — that is the
// compaction, and it is what makes later bounces cheap.
//
// Notice what is *absent*: no positions, no triangles, no BVH. EXTEND wrote a
// complete hit record precisely so this kernel needs no geometry, which is what
// lets both fit inside WebGPU's eight-storage-buffer guarantee.

//!include "common/generated.wgsl"

@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read> materials: array<Material>;
@group(0) @binding(2) var<storage, read> lights: array<Light>;

@group(1) @binding(0) var<storage, read_write> paths: array<PathState>;
@group(1) @binding(1) var<storage, read> hits: array<HitRecord>;
@group(1) @binding(2) var<storage, read_write> shadow_rays: array<ShadowRay>;
@group(1) @binding(3) var<storage, read_write> counters: WavefrontCounters;
@group(1) @binding(4) var<storage, read> queue_in: array<u32>;
@group(1) @binding(5) var<storage, read_write> queue_out: array<u32>;

//!include "common/math.wgsl"
//!include "common/rng.wgsl"
//!include "common/bsdf.wgsl"
//!include "common/envmap.wgsl"

const INVALID_PATH: u32 = 0xFFFFFFFFu;

const SAMPLING_BSDF_ONLY: u32 = 0u;
const SAMPLING_NEE_ONLY: u32 = 1u;
const SAMPLING_MIS: u32 = 2u;
const MIN_COS_LIGHT: f32 = 1.0e-6;

// Light sampling, duplicated here rather than included from common/light.wgsl:
// that file's `direct_light` casts its own shadow ray, which is exactly what the
// wavefront defers. The sampling and the measure conversion are the same.
struct LightSample {
  wi: vec3<f32>,
  distance: f32,
  radiance: vec3<f32>,
  pdf: f32,
  valid: bool,
};

fn sample_light_point(light: Light, u: vec2<f32>) -> vec3<f32> {
  if (light.kind == LIGHT_KIND_TRIANGLE) {
    let su = sqrt(u.x);
    return light.origin + (1.0 - su) * light.edge_u + (u.y * su) * light.edge_v;
  }
  return light.origin + u.x * light.edge_u + u.y * light.edge_v;
}

fn sample_lights(shading_point: vec3<f32>, u_select: f32, u_area: vec2<f32>) -> LightSample {
  var out: LightSample;
  out.valid = false;
  out.pdf = 0.0;
  if (U.num_lights == 0u) {
    return out;
  }
  let n = U.num_lights;
  let light = lights[min(u32(u_select * f32(n)), n - 1u)];

  let to_light = sample_light_point(light, u_area) - shading_point;
  let dist_sq = dot(to_light, to_light);
  if (dist_sq <= 0.0) {
    return out;
  }
  let distance = sqrt(dist_sq);
  let wi = to_light / distance;
  let cos_light = dot(light.normal, -wi);
  if (cos_light <= MIN_COS_LIGHT) {
    return out;
  }
  // Area measure -> solid angle; see crates/core/src/light.rs for the derivation.
  let pdf = (1.0 / (f32(n) * light.area)) * dist_sq / cos_light;
  if (pdf <= 0.0) {
    return out;
  }
  out.wi = wi;
  out.distance = distance;
  out.radiance = materials[light.material].emissive;
  out.pdf = pdf;
  out.valid = true;
  return out;
}

fn light_pdf_from_geometry(
  area: f32, light_normal: vec3<f32>, shading_point: vec3<f32>, hit_point: vec3<f32>,
) -> f32 {
  if (U.num_lights == 0u || area <= 0.0) {
    return 0.0;
  }
  let to_light = hit_point - shading_point;
  let dist_sq = dot(to_light, to_light);
  if (dist_sq <= 0.0) {
    return 0.0;
  }
  let cos_light = dot(light_normal, -(to_light / sqrt(dist_sq)));
  if (cos_light <= MIN_COS_LIGHT) {
    return 0.0;
  }
  return (1.0 / (f32(U.num_lights) * area)) * dist_sq / cos_light;
}

fn power_heuristic(pdf_a: f32, pdf_b: f32) -> f32 {
  if (pdf_a <= 0.0) {
    return 0.0;
  }
  let r = pdf_b / pdf_a;
  return 1.0 / (1.0 + r * r);
}

// Workgroup size must match the divisor used when appending, below.
const WG: u32 = 64u;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= arrayLength(&queue_in)) {
    return;
  }
  let path_index = queue_in[gid.x];
  if (path_index == INVALID_PATH) {
    return;
  }
  var p = paths[path_index];
  let rec = hits[gid.x];

  var rng: Rng;
  rng.state = p.rng_state;
  // Rebuild the Sobol half of the sampler. `index` and `seed` are pure
  // functions of the sample and the pixel, so they cost nothing to carry in the
  // path state; `dim` is not, and restarting it would restart the sequence
  // every bounce — the same reason `rng_state` is carried.
  // Rebuild the Sobol half of the sampler.
  //
  // The sample index is recovered from the **path index**, exactly as GENERATE
  // derives it: the pool is laid out as `sample_slot * pixels + pixel`, and a
  // path's slot in that pool never moves. Reading `U.sample_offset` instead is
  // wrong and was the bug here — the wavefront *batches* several samples into
  // one pass, so `sample_offset` is only the base of the batch and every path
  // beyond the first slot got the wrong point of the sequence.
  //
  // It stayed invisible until now because the independent sampler seeds once in
  // GENERATE and never re-reads the sample index; only Sobol, which
  // reconstructs its state every stage, could expose it.
  let pixels = U.width * U.height;
  rng.index = U.sample_offset + path_index / pixels;
  rng.seed = sobol_hash(p.pixel + U.frame_seed * 0x9e3779b9u);
  rng.dim = p.sampler_dim;

  // --- Escaped: add the environment and let the path die --------------------
  if (rec.valid == 0u) {
    // A *constant* background is only ever found by BSDF sampling, so there is
    // no double-counting to guard against. An environment **map** is
    // importance-sampled like any other light, so arriving here by BSDF
    // sampling has to share credit with the connection that could have found
    // the same direction.
    if (env_present()) {
      var w = 1.0;
      if (p.depth > 0u) {
        if (U.sampling_mode == SAMPLING_NEE_ONLY) {
          w = 0.0;
        } else if (U.sampling_mode == SAMPLING_MIS) {
          let pdf_env = env_pdf(p.direction) / f32(max(light_strategy_count(), 1u));
          w = power_heuristic(p.prev_bsdf_pdf, pdf_env);
        }
      }
      if (w > 0.0) {
        p.radiance = p.radiance + p.throughput * env_radiance_at(p.direction) * w;
      }
    } else {
      p.radiance = p.radiance + p.throughput * U.background;
    }
    paths[path_index] = p;
    return;
  }

  let m = materials[rec.material];
  let front_face = rec.front_face != 0u;

  // --- Emission -------------------------------------------------------------
  if (front_face && max(m.emissive.x, max(m.emissive.y, m.emissive.z)) > 0.0) {
    // depth == 0 is special in every mode: no shadow ray preceded the camera
    // ray, so a directly visible emitter takes full credit.
    var weight = 1.0;
    if (p.depth > 0u) {
      if (U.sampling_mode == SAMPLING_NEE_ONLY) {
        weight = 0.0;
      } else if (U.sampling_mode == SAMPLING_MIS) {
        // Scaled by the light count over the strategy count, because the
        // environment map competes for the same uniform selection draw.
        let pdf_light = light_pdf_from_geometry(
          rec.light_area, rec.geometric_normal, p.prev_position, rec.position)
          * (f32(U.num_lights) / f32(max(light_strategy_count(), 1u)));
        weight = power_heuristic(p.prev_bsdf_pdf, pdf_light);
      }
    }
    if (weight > 0.0) {
      p.radiance = p.radiance + p.throughput * m.emissive * weight;
    }
  }

  let surf = surface_from_material(m, front_face);

  // Record what the first hit is looking at, for the denoiser. The *first*
  // specifically: the guides describe the surface this pixel shows, and a later
  // bounce describes somewhere else. Emission joins the albedo so a light
  // fixture is not demodulated into a division by a near-black base colour.
  if (p.depth == 0u) {
    p.guide_albedo = max(surf.diffuse_albedo + surf.f0 + m.emissive, vec3<f32>(1.0e-3));
    p.guide_normal = rec.normal;
    p.guide_depth = rec.t;
  }
  let wo = to_local(-p.direction, rec.normal);
  if (wo.z <= 0.0) {
    paths[path_index] = p;
    return;
  }

  // --- Next event estimation: build a shadow ray, do not cast it ------------
  //
  // The visibility test is deferred to CONNECT. Everything expensive about the
  // sample — the light pdf, the BSDF evaluation, the MIS weight — is resolved
  // here, so the occlusion kernel stays small and coherent.
  if (U.sampling_mode != SAMPLING_BSDF_ONLY && p.depth + 1u < U.max_depth && light_strategy_count() > 0u) {
    // Over lights *and* the environment map: either can be connected to, and
    // the decision must be the same on every device because it controls whether
    // random numbers are drawn.
    // Drawn unconditionally, even when the sample contributes nothing, so the
    // stream stays aligned with the megakernel and the CPU reference.
    let u_select = rng_next_f32(&rng);
    let u_area = rng_next_vec2(&rng);

    // Strategy selection, matching `direct_light` in common/light.wgsl step for
    // step. The environment map is one strategy among the area lights, so every
    // density carries a 1/total selection factor.
    let n_lights = U.num_lights;
    let total = light_strategy_count();
    let pick = min(u32(u_select * f32(total)), total - 1u);

    var wi_world = vec3<f32>(0.0, 1.0, 0.0);
    var radiance_in = vec3<f32>(0.0);
    var distance = 0.0;
    var pdf = 0.0;
    var valid = false;
    if (pick < n_lights) {
      let u_light = (f32(pick) + 0.5) / f32(n_lights);
      let ls = sample_lights(rec.position, u_light, u_area);
      if (ls.valid) {
        wi_world = ls.wi;
        radiance_in = ls.radiance;
        distance = ls.distance;
        pdf = ls.pdf * (f32(n_lights) / f32(total));
        valid = true;
      }
    } else {
      let es = env_sample(u_area);
      if (es.valid) {
        wi_world = es.direction;
        radiance_in = es.radiance;
        distance = ENV_DISTANCE;
        pdf = es.pdf / f32(total);
        valid = true;
      }
    }

    if (valid && pdf > 0.0) {
      let wi = to_local(wi_world, rec.normal);
      // Below the shading hemisphere is not automatically a rejection: a
      // transmissive surface can be lit from behind. This has to match
      // `direct_light` in common/light.wgsl exactly, since the megakernel uses
      // that one and the two are compared against each other — the wavefront
      // reimplements the connection inline only because it defers the shadow
      // ray, not because the rule differs.
      if (wi.z != 0.0 && (wi.z > 0.0 || surf.transmission > 0.0)) {
        let f = surface_eval(surf, wo, wi);
        if (max(f.x, max(f.y, f.z)) > 0.0) {
          var mis_weight = 1.0;
          if (U.sampling_mode == SAMPLING_MIS) {
            mis_weight = power_heuristic(pdf, surface_pdf(surf, wo, wi));
          }
          if (mis_weight > 0.0) {
            var sr: ShadowRay;
            // Toward whichever side the shadow ray leaves on: a connection
            // through a transmissive surface starts on the far side, and
            // starting it on the near side makes the surface its own occluder.
            let shadow_normal =
              select(rec.geometric_normal, -rec.geometric_normal, wi.z < 0.0);
            sr.origin = offset_ray_origin(rec.position, shadow_normal);
            sr.direction = wi_world;
            sr.distance = distance;
            sr.path = path_index;
            // abs(cos): the projected-solid-angle factor is positive on both sides.
            sr.contribution =
              p.throughput * f * abs(wi.z) * radiance_in * (mis_weight / pdf);
            let slot = atomicAdd(&counters.shadow_len, 1u);
            shadow_rays[slot] = sr;
          }
        }
      }
    }
  }

  // --- BSDF sampling --------------------------------------------------------
  let u_lobe = rng_next_f32(&rng);
  let u = rng_next_vec2(&rng);
  let bs = surface_sample(surf, wo, u_lobe, u);

  p.rng_state = rng.state;
  p.sampler_dim = rng.dim;

  if (!bs.valid) {
    paths[path_index] = p;
    return;
  }
  let next_throughput = p.throughput * bs.weight;
  if (max(next_throughput.x, max(next_throughput.y, next_throughput.z)) <= 0.0) {
    paths[path_index] = p;
    return;
  }
  if (p.depth + 1u >= U.max_depth) {
    // Truncated at the bounce limit. Not appended, so the path is compacted out.
    paths[path_index] = p;
    return;
  }

  p.throughput = next_throughput;
  p.prev_bsdf_pdf = bs.pdf;
  p.prev_position = rec.position;
  let exit_normal = select(rec.geometric_normal, -rec.geometric_normal, bs.wi.z < 0.0);
  p.origin = offset_ray_origin(rec.position, exit_normal);
  p.direction = to_world(bs.wi, rec.normal);
  p.depth = p.depth + 1u;
  paths[path_index] = p;

  // Survived: append to the next bounce's queue. Queue order varies between
  // runs, which does not affect the result — every path writes to its own pixel
  // and the RNG is seeded from (frame, pixel, sample) rather than from queue
  // position.
  let slot = atomicAdd(&counters.next_queue_len, 1u);
  queue_out[slot] = path_index;
  // Grow the next dispatch as items land, so no separate pass is needed to turn
  // a count into a workgroup count.
  atomicMax(&counters.next_trace_x, (slot / WG) + 1u);
}
