//! The CPU reference path tracer.
//!
//! This is the oracle. It is written to be *obviously* correct rather than fast:
//! brute-force intersection, no acceleration structure, no clever reuse. Every
//! GPU result in the project is ultimately checked against this.
//!
//! # Estimator (build steps 2–3)
//!
//! Unidirectional path tracing with a selectable strategy: BSDF sampling only,
//! next event estimation only, or both combined with multiple importance
//! sampling. All three estimate the same integral and must converge to the same
//! image, differing only in variance — which is asserted directly in
//! `tests/sampling_modes.rs`.
//!
//! Russian roulette is deliberately **off**. Paths are truncated at
//! `max_depth`, which introduces a small, deterministic, *identical-on-both-
//! devices* bias. Adding RR now would only make the CPU/GPU comparison noisier
//! without testing anything new; it arrives with the rest of the light transport
//! work at build step 9.

use crate::bsdf::surface::{self, Surface};
use crate::camera::generate_ray;
use crate::gpu_layout::GpuUniforms;
use crate::light;
use crate::math::{offset_ray_origin, to_local, to_world};
use crate::sobol::{Sampler, SamplerKind};
use crate::scene::{Ray, Scene};
use crate::scenes::SceneDef;
use glam::Vec3;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Which strategy the integrator uses to find light.
///
/// All of these estimate the *same* integral and must converge to the same
/// image, differing only in variance. That equality is the strongest test of the
/// whole light transport implementation, and it is asserted directly once
/// multiple importance sampling lands at build step 9.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplingMode {
    /// Light is found only by randomly walking into an emitter. Unbiased, and
    /// hopeless for small lights — a 130 x 105 light in a 555-unit box is hit by
    /// roughly one bounce in a thousand.
    BsdfOnly,
    /// Connect every path vertex directly to a sampled point on a light.
    /// Converges enormously faster for small lights, and worse for large ones or
    /// for near-specular surfaces, where the light-sampling density is a poor
    /// match for the BSDF.
    NeeOnly,
    /// Both strategies at every vertex, combined with the power heuristic.
    ///
    /// Each strategy is weighted by how densely *it* sampled the direction
    /// relative to the other, so whichever was better dominates and the weights
    /// still sum to 1. The result beats both: light sampling carries the small
    /// bright emitter, BSDF sampling carries the glossy highlight, and neither
    /// has to be good at the other's job.
    Mis,
}

impl SamplingMode {
    pub const ALL: [SamplingMode; 3] = [
        SamplingMode::BsdfOnly,
        SamplingMode::NeeOnly,
        SamplingMode::Mis,
    ];

    /// Does this mode connect to lights explicitly?
    pub fn uses_nee(self) -> bool {
        matches!(self, SamplingMode::NeeOnly | SamplingMode::Mis)
    }

    /// Index passed to the shader.
    pub fn index(self) -> u32 {
        match self {
            SamplingMode::BsdfOnly => 0,
            SamplingMode::NeeOnly => 1,
            SamplingMode::Mis => 2,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            SamplingMode::BsdfOnly => "bsdf",
            SamplingMode::NeeOnly => "nee",
            SamplingMode::Mis => "mis",
        }
    }
    pub fn parse(s: &str) -> Option<SamplingMode> {
        match s {
            "bsdf" => Some(SamplingMode::BsdfOnly),
            "nee" => Some(SamplingMode::NeeOnly),
            "mis" => Some(SamplingMode::Mis),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RenderParams {
    pub width: u32,
    pub height: u32,
    pub samples: u32,
    pub max_depth: u32,
    pub frame_seed: u32,
    /// Index of the first sample. Non-zero lets a render be resumed or split
    /// across devices while keeping every sample's RNG stream distinct.
    pub sample_offset: u32,
    /// 0 = one thread per available core.
    pub threads: usize,
    pub sampling: SamplingMode,
    /// Which point set to draw from. See [`crate::sobol`].
    pub sampler: SamplerKind,
}

impl Default for RenderParams {
    fn default() -> Self {
        Self {
            width: 512,
            height: 512,
            samples: 256,
            max_depth: 8,
            frame_seed: 0x5eed_1234,
            sample_offset: 0,
            threads: 0,
            sampling: SamplingMode::Mis,
            sampler: SamplerKind::default(),
        }
    }
}

/// A linear-HDR image. Row-major, **row 0 is the top row**.
#[derive(Clone, Debug)]
pub struct Film {
    pub width: u32,
    pub height: u32,
    pub data: Vec<Vec3>,
}

impl Film {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            data: vec![Vec3::ZERO; (width * height) as usize],
        }
    }
    pub fn pixel(&self, x: u32, y: u32) -> Vec3 {
        self.data[(y * self.width + x) as usize]
    }
}

/// Fill in the uniform block the GPU will receive, from a scene and render
/// parameters. The CPU tracer then reads the *same* struct, so there is exactly
/// one place where camera and scene parameters are resolved.
pub fn build_uniforms(def: &SceneDef, params: &RenderParams) -> GpuUniforms {
    let mut u = GpuUniforms {
        width: params.width,
        height: params.height,
        sample_offset: params.sample_offset,
        samples_per_launch: params.samples,
        max_depth: params.max_depth,
        frame_seed: params.frame_seed,
        num_primitives: def.scene.blob.primitives.len() as u32,
        num_triangles: def.scene.blob.triangles.len() as u32,
        num_bvh_nodes: def.scene.blob.bvh_nodes.len() as u32,
        num_lights: def.scene.blob.lights.len() as u32,
        sampling_mode: params.sampling.index(),
        background: def.background.to_array(),
        env_width: def.scene.env.width,
        env_height: def.scene.env.height,
        env_total_weight: def.scene.env.total_weight,
        sampler_kind: params.sampler.index(),
        num_instances: def.scene.blob.instances.len() as u32,
        tlas_root: def.scene.tlas_root,
        ..Default::default()
    };
    def.camera
        .write_uniforms(&mut u, params.width as f32 / params.height as f32);
    u
}

/// Estimate the radiance arriving along `ray`.
///
/// Mirrored exactly by `trace_path` in `shaders/trace/megakernel.wgsl`, down to
/// the order in which random numbers are drawn.
/// What the first hit is looking at, for the denoiser's edge-stopping weights.
///
/// Noise-free by construction: it comes from one deterministic intersection
/// rather than from an integral, which is what makes it usable as a guide for
/// filtering something that is noisy.
#[derive(Clone, Copy, Debug, Default)]
pub struct GuideSample {
    pub albedo: Vec3,
    pub normal: Vec3,
    pub depth: f32,
    /// BVH nodes visited by the first ray, for the traversal heatmap.
    pub steps: f32,
}

pub fn radiance(
    scene: &Scene,
    background: Vec3,
    ray: Ray,
    rng: &mut Sampler,
    max_depth: u32,
    mode: SamplingMode,
) -> Vec3 {
    radiance_with_guide(scene, background, ray, rng, max_depth, mode, &mut GuideSample::default())
}

pub fn radiance_with_guide(
    scene: &Scene,
    background: Vec3,
    mut ray: Ray,
    rng: &mut Sampler,
    max_depth: u32,
    mode: SamplingMode,
    guide: &mut GuideSample,
) -> Vec3 {
    let mut radiance = Vec3::ZERO;
    let mut throughput = Vec3::ONE;

    // Carried across iterations for the multiple importance sampling weight: to
    // decide how much credit BSDF sampling deserves for landing on an emitter,
    // we need the density it used and the vertex it started from.
    let mut prev_bsdf_pdf = 0.0f32;
    let mut prev_position = ray.origin;

    for depth in 0..max_depth {
        let Some(hit) = scene.intersect(&ray) else {
            // The environment. A *constant* background is only ever found by
            // BSDF sampling, and that is correct in every mode: there is no
            // light-sampling strategy for it to be balanced against.
            //
            // An environment **map** is different — it is importance-sampled
            // like any other light, so arriving here by BSDF sampling has to
            // share credit with the connection that could have found the same
            // direction. Skipping that weight double-counts the sky and makes
            // every outdoor scene about twice as bright as it should be.
            let env = &scene.env;
            if !env.is_empty() {
                let l = env.radiance(ray.dir);
                let weight = if depth == 0 {
                    // No shadow ray preceded the camera ray, so a directly
                    // visible sky takes full credit.
                    1.0
                } else {
                    match mode {
                        SamplingMode::BsdfOnly => 1.0,
                        SamplingMode::NeeOnly => 0.0,
                        SamplingMode::Mis => {
                            let total = scene.light_strategy_count().max(1);
                            let pdf_env = env.pdf(ray.dir) / total as f32;
                            light::power_heuristic(prev_bsdf_pdf, pdf_env)
                        }
                    }
                };
                if weight > 0.0 {
                    radiance += throughput * l * weight;
                }
            } else {
                radiance += throughput * background;
            }
            break;
        };
        let m = scene.material(hit.material);

        // --- Emission --------------------------------------------------------
        //
        // One-sided: only the face the stored normal points from emits. Without
        // this the ceiling light would also illuminate the 0.1-unit gap above
        // it, and every emitter would leak backwards.
        let emissive = Vec3::from_array(m.emissive);
        if hit.front_face && emissive.max_element() > 0.0 {
            // `depth == 0` is special in every mode: no shadow ray preceded the
            // camera ray, so a directly visible emitter is found only this way
            // and takes full credit. Forgetting it renders the light fixture
            // black while the room it lights looks perfect.
            let weight = if depth == 0 {
                1.0
            } else {
                match mode {
                    // Light sampling is not running, so nothing to share with.
                    SamplingMode::BsdfOnly => 1.0,
                    // The shadow ray from the previous vertex already accounted
                    // for this emitter; counting it again would double every
                    // light path.
                    SamplingMode::NeeOnly => 0.0,
                    // Share the credit. How likely was light sampling to have
                    // produced this same direction from the previous vertex?
                    SamplingMode::Mis => {
                        // Scaled by the light count over the strategy count,
                        // because the environment map competes for the same
                        // uniform selection draw.
                        let n = scene.blob.lights.len() as u32;
                        let total = scene.light_strategy_count();
                        let pdf_light = light::light_pdf_from_geometry(
                            n,
                            hit.light_area,
                            hit.geometric_normal,
                            prev_position,
                            hit.position,
                        ) * (n as f32 / total.max(1) as f32);
                        light::power_heuristic(prev_bsdf_pdf, pdf_light)
                    }
                }
            };
            if weight > 0.0 {
                radiance += throughput * emissive * weight;
            }
        }

        let surf = Surface::from_material(m, hit.front_face);

        // Record the first hit, once. The *first* specifically: the guides
        // describe the surface this pixel shows, and a later bounce describes
        // somewhere else. Emission joins the albedo so a light fixture is not
        // demodulated into a division by a near-black base colour.
        if depth == 0 {
            guide.albedo = (surf.diffuse_albedo + surf.f0 + emissive).max(Vec3::splat(1.0e-3));
            guide.normal = hit.normal;
            guide.depth = hit.t;
            guide.steps = hit.steps as f32;
        }
        let wo = to_local(-ray.dir, hit.normal);
        if wo.z <= 0.0 {
            // The shading normal disagrees with the geometric one strongly
            // enough that the viewer sits below the shading hemisphere. There is
            // nothing sensible to evaluate; terminating beats producing a
            // negative cosine and a black or exploding pixel.
            break;
        }

        // --- Next event estimation -------------------------------------------
        //
        // Only while a further bounce would still be allowed. A light connection
        // from this vertex forms a path one segment longer than stopping here,
        // so running it on the final iteration would let the light-sampling
        // modes reach paths BSDF sampling cannot at the same `max_depth` — and
        // the modes would then converge to different images for reasons that
        // have nothing to do with any of them being wrong.
        //
        // The emptiness check is part of the contract, not an optimisation: it
        // decides whether random numbers are drawn, so the shader must make the
        // same decision or the two streams diverge.
        //
        // The emptiness check is over lights *and* the environment map: either
        // can be connected to, and the decision has to be the same on every
        // device because it controls whether random numbers are drawn.
        let has_env = !scene.env.is_empty();
        if mode.uses_nee()
            && depth + 1 < max_depth
            && (!scene.blob.lights.is_empty() || has_env)
        {
            radiance += throughput * direct_light(scene, &surf, &hit, wo, rng, mode);
        }

        // --- BSDF sampling ---------------------------------------------------
        //
        // Draw order is part of the CPU/GPU contract: lobe choice first, then
        // the two-dimensional sample within the chosen lobe.
        let u_lobe = rng.next_f32();
        let u = rng.next_vec2();
        let sample = surface::sample(&surf, wo, u_lobe, u);
        if !sample.is_valid() {
            break;
        }

        // The weight already carries f * cos / pdf, including the combined
        // two-lobe density — see `bsdf::surface`.
        throughput *= sample.weight;
        if throughput.max_element() <= 0.0 {
            break;
        }

        prev_bsdf_pdf = sample.pdf;
        prev_position = hit.position;

        let dir = to_world(sample.wi, hit.normal);
        ray = Ray {
            // Offset along the *geometric* normal, not the shading normal: on a
            // smooth mesh the interpolated normal can lean far enough from the
            // facet that offsetting along it leaves the origin below the surface.
            //
            // And along the side the ray is actually leaving on. A transmitted
            // ray goes *into* the surface, so pushing it out along the outward
            // normal leaves it on the wrong side — where it either re-hits the
            // surface it just crossed or never enters the object at all. Both
            // render glass as solid black, which is exactly what this produced
            // before the sign was tied to the sampled direction rather than
            // assumed.
            origin: offset_ray_origin(
                hit.position,
                if sample.wi.z < 0.0 {
                    -hit.geometric_normal
                } else {
                    hit.geometric_normal
                },
            ),
            dir,
        };
    }

    radiance
}

/// One light-sampling connection from a shading point.
///
/// Returns the radiance to add, already divided by the sampling density:
///
/// ```text
///   L = f(wo, wi) * cos(theta_i) * L_e * V / p_w(wi)
/// ```
///
/// where `p_w` is the light sampler's density **in solid angle** — see
/// [`crate::light`] for why that conversion is where energy bugs live.
///
/// Random numbers are drawn unconditionally, even when the sample turns out to
/// contribute nothing, so the stream stays aligned with the shader's.
fn direct_light(
    scene: &Scene,
    surf: &Surface,
    hit: &crate::scene::Hit,
    wo: Vec3,
    rng: &mut Sampler,
    mode: SamplingMode,
) -> Vec3 {
    let u_select = rng.next_f32();
    let u_area = rng.next_vec2();

    // The environment map is one strategy among the area lights, chosen with the
    // same uniform draw. Every density below therefore carries a 1/total
    // selection factor — including the one the BSDF side uses for its MIS
    // weight, which is why `light_selection_pdf_scale` exists rather than the
    // scaling being written out three times and getting out of step once.
    let n_lights = scene.blob.lights.len() as u32;
    let total = scene.light_strategy_count();
    if total == 0 {
        return Vec3::ZERO;
    }
    let pick = ((u_select * total as f32) as u32).min(total - 1);

    let (wi_world, radiance, distance, pdf) = if pick < n_lights {
        // Select this light exactly. The fraction within the cell is not needed
        // for the choice — the point on the light comes from `u_area` — so
        // handing `sample_lights` the cell centre wastes no randomness.
        let u_light = (pick as f32 + 0.5) / n_lights as f32;
        let Some(ls) = light::sample_lights(
            &scene.blob.lights,
            &scene.blob.materials,
            hit.position,
            u_light,
            u_area,
        ) else {
            return Vec3::ZERO;
        };
        // `sample_lights` already divided by the light count; rescale to the
        // full strategy count.
        let scale = n_lights as f32 / total as f32;
        (ls.wi, ls.radiance, ls.distance, ls.pdf * scale)
    } else {
        let es = scene.env.sample(u_area);
        if es.pdf <= 0.0 {
            return Vec3::ZERO;
        }
        // Nothing occludes the sky from beyond itself, so the shadow ray runs to
        // infinity — any hit at all blocks it.
        (
            es.direction,
            es.radiance,
            f32::INFINITY,
            es.pdf / total as f32,
        )
    };
    if pdf <= 0.0 {
        return Vec3::ZERO;
    }

    let wi = to_local(wi_world, hit.normal);
    // Below the shading hemisphere is *not* automatically a rejection any more:
    // a transmissive surface can be lit from behind, and that is most of what
    // makes glass look like glass.
    //
    // It has to be allowed for MIS to stay unbiased, too. The BSDF strategy can
    // reach a light through the surface, and the weight it gets assumes light
    // sampling could have found the same direction. If light sampling refuses
    // every transmitted direction while the BSDF side still discounts itself for
    // them, the two strategies sum to less than one and light seen through glass
    // is systematically dim.
    if wi.z == 0.0 || (wi.z < 0.0 && surf.transmission <= 0.0) {
        return Vec3::ZERO;
    }
    let f = surface::eval(surf, wo, wi);
    if f.max_element() <= 0.0 {
        return Vec3::ZERO;
    }

    // Under MIS the BSDF strategy could also have produced this direction, so
    // the two share the credit. Both densities are in solid angle at this
    // shading point — mixing measures here is the classic way to get an image
    // that is subtly wrong everywhere and obviously wrong nowhere.
    let mis_weight = match mode {
        SamplingMode::Mis => {
            let pdf_bsdf = surface::pdf(surf, wo, wi);
            light::power_heuristic(pdf, pdf_bsdf)
        }
        // Light sampling alone: it takes full credit.
        _ => 1.0,
    };
    if mis_weight <= 0.0 {
        return Vec3::ZERO;
    }

    // The shadow ray goes last, because it is by far the most expensive part of
    // this function and everything above can reject the sample for free.
    //
    // Offset toward whichever side the shadow ray leaves on — a connection
    // through a transmissive surface starts on the far side, and starting it on
    // the near side makes the surface its own occluder.
    let origin = offset_ray_origin(
        hit.position,
        if wi.z < 0.0 {
            -hit.geometric_normal
        } else {
            hit.geometric_normal
        },
    );
    if !light::unoccluded(scene, origin, wi_world, distance) {
        return Vec3::ZERO;
    }

    // `|cos|`: the projected-solid-angle factor is positive on both sides.
    f * wi.z.abs() * radiance * (mis_weight / pdf)
}

/// Render one pixel's mean radiance.
#[inline]
fn render_pixel(
    def: &SceneDef,
    u: &GpuUniforms,
    x: u32,
    y: u32,
    guide: &mut GuideSample,
) -> Vec3 {
    let pixel_index = y * u.width + x;
    let mut sum = Vec3::ZERO;
    for s in 0..u.samples_per_launch {
        // Read from the uniforms rather than passed separately, so the CPU
        // and the shader make the identical choice from the identical byte.
        let kind = if u.sampler_kind == SamplerKind::Sobol.index() {
            SamplerKind::Sobol
        } else {
            SamplerKind::Independent
        };
        let mut rng = Sampler::new(kind, u.frame_seed, pixel_index, u.sample_offset + s);

        // Draw order is part of the CPU/GPU contract. The lens sample is drawn
        // unconditionally, even for a pinhole camera where it goes unused, so
        // that turning depth of field on or off does not shift every subsequent
        // random number and silently change the image.
        let pixel_uv = rng.next_vec2();
        let lens_uv = rng.next_vec2();

        let ray = generate_ray(u, x, y, pixel_uv, lens_uv);
        let mode = match u.sampling_mode {
            0 => SamplingMode::BsdfOnly,
            1 => SamplingMode::NeeOnly,
            _ => SamplingMode::Mis,
        };
        let mut g = GuideSample::default();
        sum += radiance_with_guide(
            &def.scene,
            def.background,
            ray,
            &mut rng,
            u.max_depth,
            mode,
            &mut g,
        );
        guide.albedo += g.albedo;
        guide.normal += g.normal;
        guide.depth += g.depth;
        guide.steps += g.steps;
    }
    let inv = 1.0 / u.samples_per_launch as f32;
    guide.albedo *= inv;
    guide.normal *= inv;
    guide.depth *= inv;
    guide.steps *= inv;
    sum * inv
}

/// Render `def` into a [`Film`], in parallel across scoped threads.
///
/// Work is handed out as 32-row bands through an atomic counter rather than
/// split into equal contiguous ranges: cost per pixel varies a lot (a ray that
/// escapes through the open front of the box terminates immediately, a ray that
/// rattles around the corner does not), so static partitioning leaves cores idle
/// at the end.
pub fn render(def: &SceneDef, params: &RenderParams) -> Film {
    render_with_progress(def, params, |_, _| {})
}

/// Render, and also return the denoiser's guide channels.
///
/// Single-threaded, deliberately: this exists for tests and for the CLI's
/// denoise path, where the extra allocation and the loss of the banded
/// scheduler matter less than keeping the fast path free of an out-parameter it
/// does not use.
pub fn render_with_guides(
    def: &SceneDef,
    params: &RenderParams,
) -> (Film, crate::denoise::Guides) {
    let u = build_uniforms(def, params);
    let mut film = Film::new(params.width, params.height);
    let n = (params.width * params.height) as usize;
    let mut guides = crate::denoise::Guides {
        albedo: vec![Vec3::ZERO; n],
        normal: vec![Vec3::ZERO; n],
        depth: vec![0.0; n],
        steps: vec![0.0; n],
    };
    for y in 0..params.height {
        for x in 0..params.width {
            let i = (y * params.width + x) as usize;
            let mut g = GuideSample::default();
            film.data[i] = render_pixel(def, &u, x, y, &mut g);
            guides.albedo[i] = g.albedo;
            guides.normal[i] = g.normal;
            guides.depth[i] = g.depth;
            guides.steps[i] = g.steps;
        }
    }
    (film, guides)
}

pub fn render_with_progress(
    def: &SceneDef,
    params: &RenderParams,
    progress: impl Fn(u32, u32) + Sync,
) -> Film {
    const BAND: u32 = 32;

    let u = build_uniforms(def, params);
    let mut film = Film::new(params.width, params.height);

    let n_threads = if params.threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        params.threads
    };

    let num_bands = params.height.div_ceil(BAND);
    let next_band = AtomicUsize::new(0);
    let done_bands = AtomicUsize::new(0);

    // Split the film into disjoint mutable row-bands so threads never alias.
    let mut bands: Vec<&mut [Vec3]> = Vec::new();
    {
        let mut rest: &mut [Vec3] = &mut film.data;
        for b in 0..num_bands {
            let rows = BAND.min(params.height - b * BAND);
            let (head, tail) = rest.split_at_mut((rows * params.width) as usize);
            bands.push(head);
            rest = tail;
        }
        debug_assert!(rest.is_empty());
    }
    // Wrap each band so a worker can claim one by index. `Mutex` would serialise
    // nothing useful here — the bands are already disjoint — so hand out raw
    // slices guarded only by the atomic index, via `Option::take`.
    let slots: Vec<std::sync::Mutex<Option<&mut [Vec3]>>> = bands
        .into_iter()
        .map(|b| std::sync::Mutex::new(Some(b)))
        .collect();

    std::thread::scope(|scope| {
        for _ in 0..n_threads {
            scope.spawn(|| loop {
                let b = next_band.fetch_add(1, Ordering::Relaxed);
                if b >= num_bands as usize {
                    break;
                }
                let band = slots[b].lock().unwrap().take().expect("band claimed twice");
                let y0 = b as u32 * BAND;
                for (row, chunk) in band.chunks_mut(params.width as usize).enumerate() {
                    let y = y0 + row as u32;
                    for (x, px) in chunk.iter_mut().enumerate() {
                        *px = render_pixel(def, &u, x as u32, y, &mut GuideSample::default());
                    }
                }
                let d = done_bands.fetch_add(1, Ordering::Relaxed) + 1;
                progress(d as u32, num_bands);
            });
        }
    });

    film
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenes::cornell_box;

    /// Threading must not change the answer. Same seed, same image, regardless
    /// of how many workers and in what order bands are claimed.
    #[test]
    fn render_is_deterministic_across_thread_counts() {
        let def = cornell_box();
        let base = RenderParams {
            width: 48,
            height: 48,
            samples: 4,
            max_depth: 4,
            threads: 1,
            ..Default::default()
        };
        let a = render(&def, &base);
        let b = render(&def, &RenderParams { threads: 8, ..base });
        for i in 0..a.data.len() {
            assert_eq!(
                a.data[i], b.data[i],
                "pixel {i} differs between 1 and 8 threads"
            );
        }
    }

    /// Nothing in a closed diffuse box may be negative or NaN.
    #[test]
    fn render_is_finite_and_non_negative() {
        let def = cornell_box();
        let film = render(
            &def,
            &RenderParams {
                width: 64,
                height: 64,
                samples: 16,
                max_depth: 6,
                ..Default::default()
            },
        );
        for (i, p) in film.data.iter().enumerate() {
            assert!(p.is_finite(), "pixel {i} is not finite: {p}");
            assert!(p.min_element() >= 0.0, "pixel {i} is negative: {p}");
        }
    }

    /// Colour bleeding: the floor next to the red wall must pick up more red
    /// than the floor next to the green wall, and vice versa. This is a direct
    /// test that indirect illumination is being transported at all.
    #[test]
    fn colour_bleeds_from_the_walls() {
        let def = cornell_box();
        let film = render(
            &def,
            &RenderParams {
                width: 128,
                height: 128,
                samples: 512,
                max_depth: 6,
                ..Default::default()
            },
        );
        // Sample a patch of floor near the left (red, world x = 555) wall and
        // one near the right (green, x = 0) wall, both below the spheres.
        let patch = |x0: u32, y0: u32| -> Vec3 {
            let mut acc = Vec3::ZERO;
            for y in y0..y0 + 6 {
                for x in x0..x0 + 6 {
                    acc += film.pixel(x, y);
                }
            }
            acc / 36.0
        };
        let left = patch(8, 108);
        let right = patch(114, 108);

        let left_ratio = left.x / left.y.max(1e-6);
        let right_ratio = right.z / right.y.max(1e-6);
        let left_green_ratio = left.y / left.x.max(1e-6);

        assert!(
            left_ratio > 1.15,
            "floor by the red wall is not reddened: rgb = {left} (r/g = {left_ratio})"
        );
        assert!(
            left_green_ratio < 1.0 / 1.15,
            "floor by the red wall looks green: rgb = {left}"
        );
        assert!(
            right.y / right.x.max(1e-6) > 1.15,
            "floor by the green wall is not greened: rgb = {right} (g/r = {})",
            right.y / right.x.max(1e-6)
        );
        let _ = right_ratio;
    }

    /// Raising the bounce limit may only *add* energy (each extra bounce
    /// contributes a non-negative amount), and the increments must shrink
    /// geometrically since every bounce multiplies throughput by an albedo < 1.
    /// A version that gains energy without bound has a broken estimator.
    #[test]
    fn energy_converges_as_depth_increases() {
        let def = cornell_box();
        let mean = |depth: u32| {
            let film = render(
                &def,
                &RenderParams {
                    width: 64,
                    height: 64,
                    samples: 128,
                    max_depth: depth,
                    ..Default::default()
                },
            );
            film.data
                .iter()
                .fold(Vec3::ZERO, |a, b| a + *b)
                .element_sum()
                / (film.data.len() as f32 * 3.0)
        };
        let m: Vec<f32> = (1..=6).map(mean).collect();
        for w in m.windows(2) {
            assert!(w[1] >= w[0] - 1e-5, "energy decreased with depth: {m:?}");
        }
        let d1 = m[2] - m[1];
        let d2 = m[5] - m[4];
        assert!(
            d2 < d1 * 0.6,
            "bounce contributions are not decaying: {m:?}"
        );
    }
}
