//! Wavefront path tracing: one kernel per stage, with compaction between them.
//!
//! # Why not a megakernel
//!
//! A megakernel runs the whole path in one shader. That is simpler, and it
//! performs badly for two reasons, both of which show up as wasted occupancy:
//!
//! * **The register budget is set by the heaviest branch.** BVH traversal
//!   carries a 32-entry stack in private memory; the BSDF evaluation and the
//!   light sampling each want registers of their own. A thread that is only
//!   shading still pays for the traversal stack, because the shader's register
//!   allocation is static.
//! * **Dead lanes stay in the loop.** Threads execute in lockstep within a warp,
//!   so a warp keeps iterating until its longest-lived path finishes. Cost
//!   therefore tracks the bounce *limit* rather than the average path length.
//!
//! Splitting into stages fixes both: each kernel's register use is its own, and
//! paths that terminate are compacted out of the queue, so later bounces
//! dispatch over fewer threads.
//!
//! # What it actually buys
//!
//! Measured at 256x256, 8 spp, raising the bounce limit from 1 to 32 (ms):
//!
//! ```text
//!                     d=1    d=2    d=4    d=8   d=16   d=32   ratio
//!   cornell-mesh
//!     megakernel      5.25   9.87  22.42  42.67  65.82  74.92   14.3x
//!     wavefront      13.68  18.12  26.17  40.09  45.65  48.36    3.5x
//!   bvh-stress
//!     megakernel     10.29  19.90  44.61  85.35 130.97 157.49   15.3x
//!     wavefront      16.34  26.20  41.58  63.06  82.37  88.66    5.4x
//! ```
//!
//! That is the claim, measured: the megakernel's cost scales with the limit
//! (14-15x for 32x the bounces) while the wavefront's scales with how long paths
//! actually live (3.5-5.4x). They cross over around depth 4 to 8, and by depth
//! 32 the wavefront is 1.55x faster on `cornell-mesh` and 1.78x on
//! `bvh-stress`.
//!
//! # What it costs
//!
//! Path state moves from registers to global memory and back at every bounce —
//! 80 bytes written and read per path per stage — and the queue is only worth
//! compacting if paths actually die at different times. On a geometrically
//! trivial scene neither pays off, and the wavefront is *slower*: at 512x512,
//! 8 spp, depth 8, it runs at 0.16x the megakernel on `furnace-test` and 0.58x
//! on `cornell-box`, against 1.22x on `cornell-mesh` and 1.31x on
//! `bvh-stress`.
//!
//! That is why the megakernel stays available behind a flag rather than being
//! deleted. Neither architecture dominates, and which one wins is a property of
//! the scene.
//!
//! # The pipeline
//!
//! ```text
//!   GENERATE  once per sample: camera rays, queue = every pixel
//!   loop, up to max_depth times:
//!     EXTEND   closest hit for each queued path        (indirect)
//!     SHADE    emission, NEE, next BSDF sample         (indirect)
//!     CONNECT  resolve deferred shadow rays            (indirect)
//!     RESET    roll queues forward
//!   RESOLVE   fold path radiance into the accumulator
//! ```
//!
//! Dispatch sizes never return to the host. Each stage sizes the next one by
//! writing workgroup counts into a buffer that `dispatch_workgroups_indirect`
//! reads — a readback between stages would stall for longer than the
//! architecture saves.

use crate::{storage_buffer, Gpu, GpuError};
use glam::Vec3;
use pt_core::gpu_layout::{
    GpuDispatchArgs, GpuHitRecord, GpuPathState, GpuShadowRay, GpuUniforms, GpuWavefrontCounters,
};
use pt_core::integrator::{build_uniforms, Film, RenderParams};
use pt_core::scenes::SceneDef;
use wgpu::util::DeviceExt;

/// Must match `@workgroup_size` in every wavefront kernel.
const WORKGROUP: u32 = 64;

use pt_core::gpu_layout::GpuAccum as Accum;

struct Stage {
    pipeline: wgpu::ComputePipeline,
}

fn entry(binding: u32, writable: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage {
                read_only: !writable,
            },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl Stage {
    fn new(
        gpu: &Gpu,
        shader: &str,
        layouts: &[Option<&wgpu::BindGroupLayout>],
    ) -> Result<Self, GpuError> {
        let module = gpu.create_shader(shader)?;
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(shader),
                bind_group_layouts: layouts,
                ..Default::default()
            });
        let pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(shader),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        Ok(Self { pipeline })
    }
}

/// Render `def` with the wavefront architecture.
pub fn render_native_on(
    gpu: &Gpu,
    def: &SceneDef,
    params: &RenderParams,
) -> Result<Film, GpuError> {
    render_native_on_inner(gpu, def, params, false)
}

/// As [`render_native_on`], also leaving the convergence figure in
/// `crate::LAST_CONVERGENCE`. See `crate::render_with_measured`.
pub(crate) fn render_native_measured(
    gpu: &Gpu,
    def: &SceneDef,
    params: &RenderParams,
) -> Result<Film, GpuError> {
    render_native_on_inner(gpu, def, params, true)
}

fn render_native_on_inner(
    gpu: &Gpu,
    def: &SceneDef,
    params: &RenderParams,
    measure: bool,
) -> Result<Film, GpuError> {
    let device = &gpu.device;
    let (w, h) = (params.width, params.height);
    let pixels = (w as usize) * (h as usize);
    let uniforms = build_uniforms(def, params);

    // How many samples to keep in flight at once.
    //
    // The bounce loop costs four compute passes per bounce whether or not the
    // queue still holds anything, and the host cannot tell that it has emptied
    // without a readback that would stall for longer than the architecture
    // saves. The fixed cost is therefore `ceil(samples / batch) * (2 + 4 * depth)`
    // passes, measured at roughly 22 us each on an M3. Running k samples as one
    // batch divides that by k, and keeps the queues denser for longer besides.
    //
    // The price is memory, and it is not free: every per-path buffer grows by k,
    // and past a few hundred thousand paths the larger working set costs more
    // than the passes saved. Measured at 256x256, holding work fixed and varying
    // only the batch:
    //
    // ```text
    //   depth 8,  spp 8    batch 1  2  4   8     ms 16.9 13.5 13.8 14.2
    //   depth 32, spp 16   batch 1  4  8  16     ms 68.4 28.5 25.6 25.4
    // ```
    //
    // So the useful pool size grows with the depth limit — which is what the
    // cost model says, since the fixed cost per batch is proportional to it —
    // and flattens once launch overhead stops dominating. Hence a path budget
    // that scales with `max_depth`, which lands on or next to the optimum in
    // every row above.
    //
    // The hard ceiling is WebGPU's *guaranteed* 128 MiB storage-binding limit
    // rather than whatever this adapter reports, so the native benchmark
    // predicts what the browser will do. PathState is the largest per-path entry
    // at 80 bytes, so sizing against it bounds the others too.
    const BASE_PATH_BUDGET: usize = 1 << 18;
    const MAX_BINDING_BYTES: usize = 128 * 1024 * 1024;
    let max_paths = MAX_BINDING_BYTES / std::mem::size_of::<GpuPathState>();
    let budget = (BASE_PATH_BUDGET * (params.max_depth.max(1) as usize).div_ceil(8)).min(max_paths);

    // Overridable so the benchmark can isolate what batching buys: holding
    // samples and depth fixed while varying only the batch changes the pass
    // count without changing the work, which is the one comparison that
    // separates launch overhead from real GPU time.
    let batch = match std::env::var("PT_WAVEFRONT_BATCH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        Some(n) => n,
        None => budget / pixels.max(1),
    }
    // Clamped against the binding limit even when overridden: a pool that does
    // not fit produces an invalid bind group, which WebGPU reports out of band
    // and which therefore reads as a very fast render rather than as an error.
    .clamp(1, (max_paths / pixels.max(1)).max(1))
    .clamp(1, params.samples.max(1) as usize) as u32;
    let in_flight = pixels * batch as usize;

    // --- Scene resources (group 0) ------------------------------------------
    let blob = &def.scene.blob;
    let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let materials = storage_buffer(device, "materials", blob.materials_bytes(), 80);
    // Analytic primitives with the instances appended; the TLAS's leaves index
    // past `num_primitives` into the second half.
    let prim_upload = blob.primitives_upload();
    let primitives = storage_buffer(
        device,
        "primitives",
        bytemuck::cast_slice(&prim_upload),
        64,
    );
    let lights = storage_buffer(device, "lights", blob.lights_bytes(), 64);
    let positions = storage_buffer(device, "positions", blob.positions_bytes(), 16);
    let vertex_attrs = storage_buffer(device, "vertex_attrs", blob.vertex_attrs_bytes(), 32);
    let triangles = storage_buffer(device, "triangles", blob.triangles_bytes(), 16);
    let bvh_nodes = storage_buffer(device, "bvh_nodes", blob.bvh_nodes_bytes(), 32);

    let accum = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("accum"),
        size: (pixels * std::mem::size_of::<Accum>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // --- Wavefront state (group 1) ------------------------------------------
    let paths = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("path state"),
        size: (in_flight * std::mem::size_of::<GpuPathState>()) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let hits = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hit records"),
        size: (in_flight * std::mem::size_of::<GpuHitRecord>()) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let shadow_rays = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("shadow rays"),
        size: (in_flight * std::mem::size_of::<GpuShadowRay>()) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let counters = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wavefront counters"),
        size: std::mem::size_of::<GpuWavefrontCounters>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // Separate from the counters, and INDIRECT so the GPU sizes its own
    // dispatches. They cannot share a buffer: WebGPU forbids one being both a
    // read-write binding and the indirect source within a single dispatch, and
    // SHADE writes counters while being dispatched indirectly.
    let dispatch_args = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dispatch args"),
        size: std::mem::size_of::<GpuDispatchArgs>() as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::INDIRECT
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let queue_size = (in_flight * 4) as u64;
    let queue_a = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("queue a"),
        size: queue_size,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let queue_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("queue b"),
        size: queue_size,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    // --- Per-stage bind groups ----------------------------------------------
    //
    // Each kernel gets a layout holding exactly what it declares. One shared
    // layout was the obvious first design and it fails outright: WebGPU's
    // eight-storage-buffer guarantee is per shader *stage*, counted across every
    // bind group, and the union of what these six kernels touch is fourteen.
    let env = crate::upload_envmap(gpu, &def.scene.env);
    // The environment map goes in as textures, not storage buffers. SHADE
    // already binds exactly eight of those — WebGPU's guaranteed per-stage
    // limit, counted across every bind group — so there is no room for another.
    let tex = |b: u32| wgpu::BindGroupLayoutEntry {
        binding: b,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };

    let ro = |b: u32| entry(b, false);
    let rw = |b: u32| entry(b, true);
    let uni = |b: u32| wgpu::BindGroupLayoutEntry {
        binding: b,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let layout = |label: &str, entries: &[wgpu::BindGroupLayoutEntry]| {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries,
        })
    };
    let group = |label: &str, l: &wgpu::BindGroupLayout, bufs: &[&wgpu::Buffer]| {
        let entries: Vec<wgpu::BindGroupEntry> = bufs
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: l,
            entries: &entries,
        })
    };

    // GENERATE: uniform | paths, queue_out
    let gen_scene_l = layout("generate scene", &[uni(0)]);
    let gen_state_l = layout("generate state", &[rw(0), rw(1)]);
    let gen_scene = group("generate scene", &gen_scene_l, &[&uniform_buf]);
    let gen_state_b = group("generate -> b", &gen_state_l, &[&paths, &queue_b]);

    // EXTEND: uniform, geometry | paths, hits, queue_in
    let ext_scene_l = layout("extend scene", &[uni(0), ro(1), ro(2), ro(3), ro(4), ro(5)]);
    let ext_state_l = layout("extend state", &[ro(0), rw(1), ro(2)]);
    let ext_scene = group(
        "extend scene",
        &ext_scene_l,
        &[
            &uniform_buf,
            &primitives,
            &positions,
            &vertex_attrs,
            &triangles,
            &bvh_nodes,
        ],
    );
    let ext_state_a = group("extend from a", &ext_state_l, &[&paths, &hits, &queue_a]);
    let ext_state_b = group("extend from b", &ext_state_l, &[&paths, &hits, &queue_b]);

    // SHADE: uniform, materials, lights | paths, hits, shadow_rays, counters,
    //        queue_in, queue_out
    let shade_scene_l = layout("shade scene", &[uni(0), ro(1), ro(2), tex(90), tex(91)]);
    let shade_state_l = layout("shade state", &[rw(0), ro(1), rw(2), rw(3), ro(4), rw(5)]);
    let shade_scene = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("shade scene"),
        layout: &shade_scene_l,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: materials.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: lights.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 90,
                resource: wgpu::BindingResource::TextureView(&env.radiance),
            },
            wgpu::BindGroupEntry {
                binding: 91,
                resource: wgpu::BindingResource::TextureView(&env.cdf),
            },
        ],
    });
    let shade_state_ab = group(
        "shade a->b",
        &shade_state_l,
        &[&paths, &hits, &shadow_rays, &counters, &queue_a, &queue_b],
    );
    let shade_state_ba = group(
        "shade b->a",
        &shade_state_l,
        &[&paths, &hits, &shadow_rays, &counters, &queue_b, &queue_a],
    );

    // CONNECT: uniform, primitives, positions, triangles, bvh | shadow, accum,
    //          counters. No vertex attributes: occlusion never shades.
    let conn_scene_l = layout("connect scene", &[uni(0), ro(1), ro(2), ro(3), ro(4)]);
    // Binding 2 is read-write, not read-only: WGSL requires it, because the
    // counters struct contains atomics. See the note in connect.wgsl.
    let conn_state_l = layout("connect state", &[ro(0), rw(1), rw(2)]);
    let conn_scene = group(
        "connect scene",
        &conn_scene_l,
        &[
            &uniform_buf,
            &primitives,
            &positions,
            &triangles,
            &bvh_nodes,
        ],
    );
    let conn_state = group(
        "connect state",
        &conn_state_l,
        &[&shadow_rays, &paths, &counters],
    );

    // RESET: uniform | counters, queue_out
    let reset_scene_l = layout("reset scene", &[uni(0)]);
    let reset_state_l = layout("reset state", &[rw(0), rw(1), rw(2)]);
    let reset_scene = group("reset scene", &reset_scene_l, &[&uniform_buf]);
    let reset_state_a = group(
        "reset -> a",
        &reset_state_l,
        &[&counters, &dispatch_args, &queue_a],
    );
    let reset_state_b = group(
        "reset -> b",
        &reset_state_l,
        &[&counters, &dispatch_args, &queue_b],
    );

    // RESOLVE: uniform | paths, accum
    let res_scene_l = layout("resolve scene", &[uni(0)]);
    let res_state_l = layout("resolve state", &[ro(0), rw(1)]);
    let res_scene = group("resolve scene", &res_scene_l, &[&uniform_buf]);
    let res_state = group("resolve state", &res_state_l, &[&paths, &accum]);

    // --- Pipelines -----------------------------------------------------------
    let generate = Stage::new(
        gpu,
        "wavefront/generate.wgsl",
        &[Some(&gen_scene_l), Some(&gen_state_l)],
    )?;
    let extend = Stage::new(
        gpu,
        "wavefront/extend.wgsl",
        &[Some(&ext_scene_l), Some(&ext_state_l)],
    )?;
    let shade = Stage::new(
        gpu,
        "wavefront/shade.wgsl",
        &[Some(&shade_scene_l), Some(&shade_state_l)],
    )?;
    let connect = Stage::new(
        gpu,
        "wavefront/connect.wgsl",
        &[Some(&conn_scene_l), Some(&conn_state_l)],
    )?;
    let reset = Stage::new(
        gpu,
        "wavefront/reset.wgsl",
        &[Some(&reset_scene_l), Some(&reset_state_l)],
    )?;
    let resolve = Stage::new(
        gpu,
        "wavefront/resolve.wgsl",
        &[Some(&res_scene_l), Some(&res_state_l)],
    )?;

    // --- Record ---------------------------------------------------------------
    //
    // One command buffer per sample, not one for the whole render. Two separate
    // reasons, and the first produces a wrong image rather than an error:
    //
    // * `write_buffer` is not ordered against encoder commands. The queue
    //   flushes every pending write *before* the command buffers submitted
    //   alongside them, so recording all samples into one encoder would apply
    //   all `samples` uniform updates first and then render every sample with
    //   the last one's seed — the same sample, `samples` times, averaged. The
    //   image still looks like the scene; it just never gets less noisy.
    // * The pass count is `samples * (2 + 4 * max_depth)`: 2176 at 64 spp and
    //   depth 8. That is more than the driver will take, and it surfaces far
    //   from its cause — as a failed buffer map, after the queue is dropped.
    //
    // Submitting per sample bounds both, and costs one submit per sample, which
    // is nothing against a full pipeline pass over every pixel.
    // GENERATE and the first EXTEND cover the whole pool; RESOLVE is threaded by
    // pixel, so it covers only one sample's worth.
    let pixel_groups = (pixels as u32).div_ceil(WORKGROUP);

    {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("clear accum"),
        });
        encoder.clear_buffer(&accum, 0, None);
        gpu.queue.submit([encoder.finish()]);
    }

    for base in (0..params.samples).step_by(batch as usize) {
        // The last batch may be short of a full one.
        let this_batch = batch.min(params.samples - base);
        let pool_groups = (pixels as u32 * this_batch).div_ceil(WORKGROUP);

        let mut u = uniforms;
        u.sample_offset = params.sample_offset + base;
        u.samples_per_launch = this_batch;
        gpu.queue
            .write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&u));

        // Every path starts alive, so the first queue is the identity and its
        // length is known on the host — no atomics needed to build it.
        gpu.queue.write_buffer(
            &counters,
            0,
            bytemuck::bytes_of(&GpuWavefrontCounters::default()),
        );
        gpu.queue.write_buffer(
            &dispatch_args,
            0,
            bytemuck::bytes_of(&GpuDispatchArgs {
                trace: [pool_groups, 1, 1],
                shadow: [0, 1, 1],
                ..Default::default()
            }),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wavefront sample"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("generate"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&generate.pipeline);
            pass.set_bind_group(0, &gen_scene, &[]);
            pass.set_bind_group(1, &gen_state_b, &[]);
            pass.dispatch_workgroups(pool_groups, 1, 1);
        }

        for bounce in 0..params.max_depth {
            // GENERATE filled queue_b, so bounce 0 reads b and writes a.
            let reads_b = bounce % 2 == 0;
            let ext_state = if reads_b { &ext_state_b } else { &ext_state_a };
            let shade_state = if reads_b {
                &shade_state_ba
            } else {
                &shade_state_ab
            };
            let reset_state = if reads_b {
                &reset_state_a
            } else {
                &reset_state_b
            };

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("extend"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&extend.pipeline);
                pass.set_bind_group(0, &ext_scene, &[]);
                pass.set_bind_group(1, ext_state, &[]);
                pass.dispatch_workgroups_indirect(&dispatch_args, 0);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("shade"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&shade.pipeline);
                pass.set_bind_group(0, &shade_scene, &[]);
                pass.set_bind_group(1, shade_state, &[]);
                pass.dispatch_workgroups_indirect(&dispatch_args, 0);
            }
            // RESET runs here, between SHADE and CONNECT, so CONNECT can be
            // sized from the shadow rays SHADE just appended. It also rolls the
            // path queue forward for the next bounce, which nothing before the
            // next EXTEND reads.
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("reset"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&reset.pipeline);
                pass.set_bind_group(0, &reset_scene, &[]);
                pass.set_bind_group(1, reset_state, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("connect"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&connect.pipeline);
                pass.set_bind_group(0, &conn_scene, &[]);
                pass.set_bind_group(1, &conn_state, &[]);
                pass.dispatch_workgroups_indirect(&dispatch_args, 12);
            }
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("resolve"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&resolve.pipeline);
            pass.set_bind_group(0, &res_scene, &[]);
            pass.set_bind_group(1, &res_state, &[]);
            pass.dispatch_workgroups(pixel_groups, 1, 1);
        }

        gpu.queue.submit([encoder.finish()]);
    }

    // --- Read back ------------------------------------------------------------
    let size = (pixels * std::mem::size_of::<Accum>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback"),
    });
    encoder.copy_buffer_to_buffer(&accum, 0, &staging, 0, size);
    // Recorded into the readback encoder, so measuring costs one reduction over
    // an accumulation that is already finished and no extra submit.
    let meter = measure
        .then(|| crate::stats::ConvergenceMeter::new(gpu))
        .transpose()?;
    if let Some(m) = meter.as_ref() {
        m.record(gpu, &mut encoder, &accum, pixels as u32);
    }
    gpu.queue.submit([encoder.finish()]);

    let film = read_accum(gpu, &staging, w, h, params.samples)?;
    if let Some(m) = meter.as_ref() {
        let e = m.read(gpu)?;
        crate::LAST_CONVERGENCE.with(|c| c.set(e));
    }
    // After the poll, not before: validation errors arrive during submit and
    // during the poll, so an earlier check reliably finds nothing. Without this
    // an oversized path pool produced an invalid bind group, drew nothing, and
    // still reported a time.
    gpu.check_errors()?;
    staging.unmap();
    Ok(film)
}

fn read_accum(
    gpu: &Gpu,
    staging: &wgpu::Buffer,
    w: u32,
    h: u32,
    samples: u32,
) -> Result<Film, GpuError> {
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| GpuError::Readback(format!("device poll: {e}")))?;
    rx.recv()
        .map_err(|e| GpuError::Readback(e.to_string()))?
        .map_err(|e| GpuError::Readback(e.to_string()))?;

    let data = slice
        .get_mapped_range()
        .map_err(|e| GpuError::Readback(e.to_string()))?;
    let accum: &[Accum] = bytemuck::cast_slice(&data);
    let mut film = Film::new(w, h);
    let n = samples.max(1) as f32;
    for (i, a) in accum.iter().enumerate() {
        // The sample count lives on the host here rather than in the buffer's
        // `w` channel: every pass contributes exactly one sample per pixel.
        film.data[i] = Vec3::from_array(a.radiance) / n;
    }
    drop(data);
    Ok(film)
}

/// Convenience wrapper that creates its own device.
pub fn render_native(def: &SceneDef, params: &RenderParams) -> Result<Film, GpuError> {
    let gpu = Gpu::new()?;
    render_native_on(&gpu, def, params)
}

/// Exposed so the uniform type is reachable from tests without a second import.
pub type Uniforms = GpuUniforms;
