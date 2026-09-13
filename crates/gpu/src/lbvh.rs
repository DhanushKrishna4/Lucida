//! GPU linear-BVH construction. Mirrors `crates/core/src/lbvh.rs`.
//!
//! ```text
//!   prepare    triangle AABBs, a scene-bounds reduction, Morton codes
//!   sort       radix sort by code, carrying the primitive index
//!   hierarchy  Karras: every internal node determined independently
//!   fit        node bounds, each computed from its own sorted range
//!   relayout   depth-first into adjacent-sibling nodes  (host)
//! ```
//!
//! # Where the time actually went
//!
//! The first working version took 130 ms on 368k triangles — twice as slow as
//! the sequential CPU builder it was meant to beat. Three guesses about why were
//! made and all three were wrong, which is why this module carries a
//! `PT_LBVH_TIMING` switch and why the numbers below are quoted rather than
//! reasoned about:
//!
//! * *"It recompiles six kernels on every call."* It did, and hoisting that into
//!   [`LbvhBuilder::new`] bought 6 ms of 130. Compilation is ~180 ms once, cold,
//!   and a few milliseconds thereafter from the driver's cache.
//! * *"It stalls on six separate readbacks."* Batching them into one submit and
//!   one poll bought another 2 ms.
//! * *"Duplicate Morton codes make Karras build linear chains."* `bvh-stress`
//!   has no duplicate codes at all and a tree depth of 23.
//!
//! Instrumenting instead of guessing put 43 ms of 58 in the AABB fit, for a
//! reason none of the guesses came near: it dispatched one workgroup per node,
//! and seven eighths of a Morton tree's nodes are near the leaves with ranges
//! under eight, so ~322k workgroups of 256 threads existed to run a handful of
//! iterations in a single thread. Splitting the fit by node size took it to
//! 3.2 ms. See `WORKGROUP_RANGE` in `shaders/lbvh/fit.wgsl`.
//!
//! The end state, at 368k triangles: 29 ms against the binned-SAH builder's
//! 60 ms, for a tree 1.26x more expensive to traverse. On a 10k-triangle scene
//! the fixed costs dominate and the CPU still wins outright — `bench
//! --bvh-build` prints both.
//!
//! # Pipelines are built once, not per build
//!
//! [`LbvhBuilder::new`] pays for shader compilation; [`LbvhBuilder::build`] is
//! what a renderer would call per frame. That is also the honest thing to
//! benchmark — a builder that exists to rebuild geometry every frame does not
//! recompile its shaders every frame.
//!
use crate::{storage_buffer, Gpu, GpuError};
use bytemuck::{Pod, Zeroable};
use pt_core::bvh::Aabb;
use pt_core::gpu_layout::GpuTriangle;
use pt_core::lbvh::Lbvh;
use wgpu::util::DeviceExt;

/// Must match `WG` in every kernel under `shaders/lbvh/`.
const WG: u32 = 256;
/// Must match `RADIX_BITS`. Eight passes of four bits covers a 32-bit key.
const RADIX_BITS: u32 = 4;
const RADIX: u32 = 1 << RADIX_BITS;
const PASSES: u32 = 32 / RADIX_BITS;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct SortParams {
    count: u32,
    shift: u32,
    num_groups: u32,
    _pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct BuildParams {
    count: u32,
    _pad: [u32; 3],
}

/// An AABB as the build kernels see it. 32 bytes, because `vec3<f32>` has
/// alignment 16 in WGSL and a tightly packed 24-byte struct would be read at the
/// wrong stride — every box after the first would be garbage.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default, Debug)]
struct AabbGpu {
    lo: [f32; 3],
    _p0: f32,
    hi: [f32; 3],
    _p1: f32,
}

impl From<AabbGpu> for Aabb {
    fn from(a: AabbGpu) -> Aabb {
        Aabb {
            min: glam::Vec3::from_array(a.lo),
            max: glam::Vec3::from_array(a.hi),
        }
    }
}

fn uni(b: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: b,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn sto(b: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: b,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Compiled kernels and bind group layouts, reusable across builds and scenes.
pub struct LbvhBuilder {
    sort_layout: wgpu::BindGroupLayout,
    histogram: wgpu::ComputePipeline,
    scan: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,

    prep_layout: wgpu::BindGroupLayout,
    tri_bounds_pipe: wgpu::ComputePipeline,
    scene_bounds_pipe: wgpu::ComputePipeline,
    morton_pipe: wgpu::ComputePipeline,

    hier_layout: wgpu::BindGroupLayout,
    hierarchy: wgpu::ComputePipeline,

    fit_layout: wgpu::BindGroupLayout,
    fit_leaves: wgpu::ComputePipeline,
    fit_small: wgpu::ComputePipeline,
    /// Its own layout, deliberately omitting the dispatch-args binding.
    ///
    /// WebGPU forbids a buffer being a read-write binding and the indirect
    /// source *within one dispatch*, and this kernel is dispatched from the very
    /// buffer the small kernel appends to. `internal_large` never reads that
    /// binding, so leaving it out of the layout is enough — the same fix the
    /// wavefront needed for its queue counters.
    fit_large_layout: wgpu::BindGroupLayout,
    fit_large: wgpu::ComputePipeline,
}

impl LbvhBuilder {
    pub fn new(gpu: &Gpu) -> Result<Self, GpuError> {
        let device = &gpu.device;
        let layout = |label: &str, entries: &[wgpu::BindGroupLayoutEntry]| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries,
            })
        };
        let build = |shader: &str,
                     l: &wgpu::BindGroupLayout,
                     entry_points: &[&str]|
         -> Result<Vec<wgpu::ComputePipeline>, GpuError> {
            let module = gpu.create_shader(shader)?;
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(shader),
                bind_group_layouts: &[Some(l)],
                ..Default::default()
            });
            Ok(entry_points
                .iter()
                .map(|ep| {
                    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(ep),
                        layout: Some(&pl),
                        module: &module,
                        entry_point: Some(ep),
                        compilation_options: Default::default(),
                        cache: None,
                    })
                })
                .collect())
        };

        let sort_layout = layout(
            "radix",
            &[
                uni(0),
                sto(1, true),
                sto(2, true),
                sto(3, false),
                sto(4, false),
                sto(5, false),
            ],
        );
        let mut sort = build(
            "lbvh/radix.wgsl",
            &sort_layout,
            &["histogram", "scan", "scatter"],
        )?;

        let prep_layout = layout(
            "prepare",
            &[
                uni(0),
                sto(1, true),
                sto(2, true),
                sto(3, false),
                sto(4, false),
                sto(5, false),
                sto(6, false),
            ],
        );
        let mut prep = build(
            "lbvh/prepare.wgsl",
            &prep_layout,
            &["triangle_bounds", "scene_bounds", "morton"],
        )?;

        let hier_layout = layout(
            "hierarchy",
            &[
                uni(0),
                sto(1, true),
                sto(2, false),
                sto(3, false),
                sto(4, false),
            ],
        );
        let mut hier = build("lbvh/hierarchy.wgsl", &hier_layout, &["main"])?;

        let fit_layout = layout(
            "fit",
            &[
                uni(0),
                sto(1, true),
                sto(2, true),
                sto(3, true),
                sto(4, false),
                sto(5, false),
                sto(6, false),
                sto(7, false),
            ],
        );
        let mut fit = build("lbvh/fit.wgsl", &fit_layout, &["leaves", "internal_small"])?;
        let fit_large_layout = layout(
            "fit large",
            &[
                uni(0),
                sto(1, true),
                sto(2, true),
                sto(3, true),
                sto(4, false),
                sto(5, false),
                sto(6, false),
            ],
        );
        let mut fit_large = build("lbvh/fit.wgsl", &fit_large_layout, &["internal_large"])?;

        Ok(Self {
            sort_layout,
            histogram: sort.remove(0),
            scan: sort.remove(0),
            scatter: sort.remove(0),
            prep_layout,
            tri_bounds_pipe: prep.remove(0),
            scene_bounds_pipe: prep.remove(0),
            morton_pipe: prep.remove(0),
            hier_layout,
            hierarchy: hier.remove(0),
            fit_layout,
            fit_leaves: fit.remove(0),
            fit_small: fit.remove(0),
            fit_large_layout,
            fit_large: fit_large.remove(0),
        })
    }

    /// Record a full radix sort of `keys_a`/`vals_a` into the same buffers.
    ///
    /// Ping-pongs through the `b` pair; `PASSES` is even, so the sorted data
    /// ends up back in `a`.
    ///
    /// Each pass needs its own `shift`, and `write_buffer` is flushed ahead of
    /// the command buffers submitted with it — so a single uniform buffer
    /// rewritten per pass would run all eight passes with the last shift. Rather
    /// than submit eight times, every pass gets its own small uniform buffer,
    /// all written up front, and the whole sort records into one encoder.
    #[allow(clippy::too_many_arguments)]
    fn record_sort(
        &self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        n: u32,
        keys_a: &wgpu::Buffer,
        vals_a: &wgpu::Buffer,
        keys_b: &wgpu::Buffer,
        vals_b: &wgpu::Buffer,
        counts: &wgpu::Buffer,
    ) {
        let device = &gpu.device;
        let num_groups = n.div_ceil(WG);

        for pass in 0..PASSES {
            let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sort params"),
                contents: bytemuck::bytes_of(&SortParams {
                    count: n,
                    shift: pass * RADIX_BITS,
                    num_groups,
                    _pad0: 0,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            // Even passes read a and write b; odd passes the reverse.
            let (src_k, src_v, dst_k, dst_v) = if pass % 2 == 0 {
                (keys_a, vals_a, keys_b, vals_b)
            } else {
                (keys_b, vals_b, keys_a, vals_a)
            };
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("radix pass"),
                layout: &self.sort_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: src_k.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: src_v.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: dst_k.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: dst_v.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: counts.as_entire_binding(),
                    },
                ],
            });
            for (pipeline, groups) in [
                (&self.histogram, num_groups),
                (&self.scan, 1),
                (&self.scatter, num_groups),
            ] {
                let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                p.set_pipeline(pipeline);
                p.set_bind_group(0, &group, &[]);
                p.dispatch_workgroups(groups, 1, 1);
            }
        }
    }

    /// Run every GPU stage and read the results back.
    pub fn build_parts(
        &self,
        gpu: &Gpu,
        triangles: &[GpuTriangle],
        positions: &[[f32; 4]],
    ) -> Result<GpuBuildOutput, GpuError> {
        let device = &gpu.device;
        let n = triangles.len() as u32;
        assert!(n > 0, "empty geometry has no tree");
        let groups = n.div_ceil(WG);

        // Coarse stage timing, behind an environment variable. Kept rather than
        // deleted because every guess made about where this build spent its time
        // was wrong — shader compilation, then readback syncs — and each wrong
        // guess cost an optimisation that bought nothing.
        let timing = std::env::var("PT_LBVH_TIMING").is_ok();
        let mark = std::time::Instant::now();
        let lap = |t: &mut std::time::Instant, label: &str| {
            if timing {
                eprintln!(
                    "  lbvh {label:<26} {:>8.2} ms",
                    t.elapsed().as_secs_f64() * 1000.0
                );
            }
            *t = std::time::Instant::now();
        };
        let mut t = mark;

        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("build params"),
            contents: bytemuck::bytes_of(&BuildParams {
                count: n,
                _pad: [0; 3],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let tri_buf = storage_buffer(device, "triangles", bytemuck::cast_slice(triangles), 16);
        let pos_buf = storage_buffer(device, "positions", bytemuck::cast_slice(positions), 16);

        let rw = |label: &str, bytes: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let internal = (n.max(1) - 1).max(1) as u64;
        let total_nodes = (2 * n as u64).max(2) - 1;

        let tri_bounds = rw("tri bounds", n as u64 * 32);
        let scene_box = rw("scene bounds", 32);
        let codes = rw("morton codes", n as u64 * 4);
        let indices = rw("indices", n as u64 * 4);
        let codes_b = rw("morton codes b", n as u64 * 4);
        let indices_b = rw("indices b", n as u64 * 4);
        let counts = rw("radix counts", (RADIX * groups) as u64 * 4);
        let karras = rw("karras nodes", internal * 8);
        let parent = rw("parent", total_nodes * 4);
        let ranges = rw("node ranges", internal * 8);
        let node_box = rw("node bounds", total_nodes * 32);
        let subtree = rw("subtree size", internal * 4);
        let large_list = rw("large node list", internal * 4);
        // Both STORAGE (appended to by `internal_small`) and INDIRECT (read by
        // the dispatch of `internal_large`). Legal because those are different
        // dispatches — WebGPU only forbids a buffer being a read-write binding
        // and the indirect source *within* one dispatch.
        let large_args = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("large node dispatch"),
            size: 12,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let group = |label: &str, l: &wgpu::BindGroupLayout, bufs: &[&wgpu::Buffer]| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: l,
                entries: &bufs
                    .iter()
                    .enumerate()
                    .map(|(i, b)| wgpu::BindGroupEntry {
                        binding: i as u32,
                        resource: b.as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            })
        };
        lap(&mut t, "buffers + upload");

        let prep_g = group(
            "prepare",
            &self.prep_layout,
            &[
                &params,
                &tri_buf,
                &pos_buf,
                &tri_bounds,
                &scene_box,
                &codes,
                &indices,
            ],
        );
        let hier_g = group(
            "hierarchy",
            &self.hier_layout,
            &[&params, &codes, &karras, &parent, &ranges],
        );
        let fit_g = group(
            "fit",
            &self.fit_layout,
            &[
                &params,
                &ranges,
                &tri_bounds,
                &indices,
                &node_box,
                &subtree,
                &large_list,
                &large_args,
            ],
        );
        let fit_large_g = group(
            "fit large",
            &self.fit_large_layout,
            &[
                &params,
                &ranges,
                &tri_bounds,
                &indices,
                &node_box,
                &subtree,
                &large_list,
            ],
        );

        // The root has no parent; the sentinel is what the CPU relayout and the
        // hierarchy test both look for. `clear_buffer` only zeroes, so this one
        // fill comes from the host.
        let sentinel = vec![u32::MAX; total_nodes as usize];
        gpu.queue
            .write_buffer(&parent, 0, bytemuck::cast_slice(&sentinel));
        // count = 0, to be appended to; y and z fixed at 1.
        gpu.queue
            .write_buffer(&large_args, 0, bytemuck::cast_slice(&[0u32, 1, 1]));

        // Recorded as four command buffers rather than one. Submits are ordered,
        // so this is equivalent — and it lets the timing path poll between
        // stages and say which kernel the time went to, which guessing did not.
        let sync = |label: &str, enc: wgpu::CommandEncoder, t: &mut std::time::Instant| {
            gpu.queue.submit([enc.finish()]);
            if timing {
                let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
                lap(t, label);
            }
        };

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("lbvh prepare"),
        });
        for (pipe, g) in [
            (&self.tri_bounds_pipe, groups),
            (&self.scene_bounds_pipe, 1),
            (&self.morton_pipe, groups),
        ] {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, &prep_g, &[]);
            pass.dispatch_workgroups(g, 1, 1);
        }
        sync("prepare", enc, &mut t);

        // Sorts in place: codes/indices go in and come back sorted, so the
        // hierarchy and fit bind groups above already point at the right data.
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("lbvh sort"),
        });
        self.record_sort(
            gpu, &mut enc, n, &codes, &indices, &codes_b, &indices_b, &counts,
        );
        sync("sort", enc, &mut t);

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("lbvh hierarchy"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hierarchy"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.hierarchy);
            pass.set_bind_group(0, &hier_g, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        sync("hierarchy", enc, &mut t);

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("lbvh fit"),
        });
        // `leaves` is one thread per primitive; `internal` is one *workgroup*
        // per node, so its dispatch is the node count rather than a division by
        // the workgroup size. Separate dispatches because `internal` reads the
        // leaf boxes `leaves` writes, and only a dispatch boundary orders that.
        for pipe in [&self.fit_leaves, &self.fit_small] {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fit"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, &fit_g, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        {
            // Indirectly dispatched, so the number of large nodes never returns
            // to the host — a readback here would stall the whole build for a
            // single integer.
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fit large"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fit_large);
            pass.set_bind_group(0, &fit_large_g, &[]);
            pass.dispatch_workgroups_indirect(&large_args, 0);
        }
        sync("fit", enc, &mut t);

        // One readback for all six buffers, not six.
        //
        // Every readback ends in a `poll(wait)`, which is a full pipeline sync;
        // doing that once per buffer serialised six GPU stalls back to back.
        // Copying them all in one encoder and mapping them together costs one.
        let mut r = read_many(
            gpu,
            &[
                (&codes, n as usize * 4),
                (&indices, n as usize * 4),
                (&karras, internal as usize * 8),
                (&parent, total_nodes as usize * 4),
                (&node_box, total_nodes as usize * 32),
                (&subtree, internal as usize * 4),
                (&scene_box, 32),
            ],
        )?;
        let scene_bytes = r.pop().unwrap();
        let subtree_bytes = r.pop().unwrap();
        let node_bytes = r.pop().unwrap();
        let parent_bytes = r.pop().unwrap();
        let karras_bytes = r.pop().unwrap();
        let order_bytes = r.pop().unwrap();
        let code_bytes = r.pop().unwrap();

        let out = GpuBuildOutput {
            codes: cast_vec::<u32>(&code_bytes),
            order: cast_vec::<u32>(&order_bytes),
            karras: cast_vec::<[u32; 2]>(&karras_bytes),
            parent: cast_vec::<u32>(&parent_bytes),
            node_bounds: cast_vec::<AabbGpu>(&node_bytes)
                .into_iter()
                .map(Aabb::from)
                .collect(),
            subtree_size: cast_vec::<u32>(&subtree_bytes),
            scene_bounds: cast_vec::<AabbGpu>(&scene_bytes)[0].into(),
        };
        lap(&mut t, "readback");
        gpu.check_errors()?;
        Ok(out)
    }

    /// Build a tree in the renderer's own node format.
    ///
    /// The relayout that turns Karras's output into adjacent-sibling nodes runs
    /// on the host. It is the one stage that is not a map or a sort — a
    /// depth-first walk assigning output slots — and a GPU version needs a
    /// separate ordering pass built on the subtree sizes the fit computes.
    /// Everything before it is on the GPU.
    pub fn build(
        &self,
        gpu: &Gpu,
        triangles: &[GpuTriangle],
        positions: &[[f32; 4]],
    ) -> Result<Lbvh, GpuError> {
        let out = self.build_parts(gpu, triangles, positions)?;
        Ok(Lbvh::from_parts(
            &out.karras,
            &out.subtree_size,
            &out.order,
            &out.node_bounds,
            triangles.len(),
        ))
    }

    /// Sort `keys` ascending, carrying `values`. Exposed for the sort's own
    /// tests, which check it against `slice::sort` rather than through a tree.
    pub fn sort(
        &self,
        gpu: &Gpu,
        keys: &[u32],
        values: &[u32],
    ) -> Result<(Vec<u32>, Vec<u32>), GpuError> {
        assert_eq!(keys.len(), values.len(), "keys and values must be parallel");
        if keys.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let device = &gpu.device;
        let n = keys.len() as u32;
        let groups = n.div_ceil(WG);

        let make = |label: &str, data: &[u32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
        };
        let zeros = vec![0u32; keys.len()];
        let keys_a = make("keys a", keys);
        let vals_a = make("values a", values);
        let keys_b = make("keys b", &zeros);
        let vals_b = make("values b", &zeros);
        let counts = storage_buffer(
            device,
            "radix counts",
            bytemuck::cast_slice(&vec![0u32; (RADIX * groups) as usize]),
            4,
        );

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("radix sort"),
        });
        self.record_sort(
            gpu, &mut enc, n, &keys_a, &vals_a, &keys_b, &vals_b, &counts,
        );
        gpu.queue.submit([enc.finish()]);

        let out_keys = read_u32(gpu, &keys_a, keys.len())?;
        let out_values = read_u32(gpu, &vals_a, values.len())?;
        gpu.check_errors()?;
        Ok((out_keys, out_values))
    }
}

/// What the GPU produced, before the relayout that turns it into a tree the
/// renderer can traverse.
///
/// Exposed as its own type because every stage is checked against its CPU twin
/// individually. A tree that renders correctly is not evidence that the sort was
/// stable or that the range search was right — those show up as a quietly worse
/// tree, so they are compared stage by stage rather than only at the end.
pub struct GpuBuildOutput {
    pub codes: Vec<u32>,
    pub order: Vec<u32>,
    /// (left, right) child ids per internal node, in Karras's id space.
    pub karras: Vec<[u32; 2]>,
    pub parent: Vec<u32>,
    pub node_bounds: Vec<Aabb>,
    pub subtree_size: Vec<u32>,
    pub scene_bounds: Aabb,
}

// --- Convenience wrappers --------------------------------------------------
//
// Each compiles the kernels, so they are for tests and one-off calls. Anything
// building more than once should hold an `LbvhBuilder`.

pub fn sort_u32(gpu: &Gpu, keys: &[u32], values: &[u32]) -> Result<(Vec<u32>, Vec<u32>), GpuError> {
    LbvhBuilder::new(gpu)?.sort(gpu, keys, values)
}

pub fn build_on_gpu(
    gpu: &Gpu,
    triangles: &[GpuTriangle],
    positions: &[[f32; 4]],
) -> Result<GpuBuildOutput, GpuError> {
    LbvhBuilder::new(gpu)?.build_parts(gpu, triangles, positions)
}

pub fn build_lbvh_on_gpu(
    gpu: &Gpu,
    triangles: &[GpuTriangle],
    positions: &[[f32; 4]],
) -> Result<Lbvh, GpuError> {
    LbvhBuilder::new(gpu)?.build(gpu, triangles, positions)
}

fn cast_vec<T: Pod>(bytes: &[u8]) -> Vec<T> {
    bytemuck::cast_slice::<u8, T>(bytes).to_vec()
}

/// Copy several buffers to the host in a single submit and a single poll.
fn read_many(gpu: &Gpu, sources: &[(&wgpu::Buffer, usize)]) -> Result<Vec<Vec<u8>>, GpuError> {
    let device = &gpu.device;
    let staging: Vec<wgpu::Buffer> = sources
        .iter()
        .map(|(_, size)| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size: (*size as u64).max(4),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        })
        .collect();

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("batched readback"),
    });
    for ((src, size), dst) in sources.iter().zip(staging.iter()) {
        encoder.copy_buffer_to_buffer(src, 0, dst, 0, *size as u64);
    }
    gpu.queue.submit([encoder.finish()]);

    // Every map is requested before the single poll, so they all resolve
    // together rather than one stall at a time.
    let mut receivers = Vec::with_capacity(staging.len());
    for buf in &staging {
        let (tx, rx) = std::sync::mpsc::channel();
        buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        receivers.push(rx);
    }
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| GpuError::Readback(format!("device poll: {e}")))?;

    let mut out = Vec::with_capacity(staging.len());
    for (rx, buf) in receivers.into_iter().zip(staging.iter()) {
        rx.recv()
            .map_err(|e| GpuError::Readback(e.to_string()))?
            .map_err(|e| GpuError::Readback(e.to_string()))?;
        let data = buf
            .slice(..)
            .get_mapped_range()
            .map_err(|e| GpuError::Readback(e.to_string()))?;
        out.push(data.to_vec());
        drop(data);
        buf.unmap();
    }
    Ok(out)
}

fn read_u32(gpu: &Gpu, src: &wgpu::Buffer, len: usize) -> Result<Vec<u32>, GpuError> {
    let size = (len * 4) as u64;
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sort readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(src, 0, &staging, 0, size);
    gpu.queue.submit([encoder.finish()]);

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
    let out = bytemuck::cast_slice::<u8, u32>(&data).to_vec();
    drop(data);
    staging.unmap();
    Ok(out)
}
