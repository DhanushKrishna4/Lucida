//! GPU à-trous denoising. Mirrors `crates/core/src/denoise.rs`.
//!
//! Runs as its own chain of compute passes over the accumulation buffer,
//! writing a separate output the display pass reads. The accumulation is never
//! touched — denoising is a display-time product, and one more sample still
//! converges to the unbiased answer.
//!
//! Its own bind group, so the eight-storage-buffer limit that shapes everything
//! else in this renderer is not a constraint here: the tracing kernels are not
//! involved.

use crate::{Gpu, GpuError};
use bytemuck::{Pod, Zeroable};
use pt_core::denoise::DenoiseParams;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct GpuDenoiseParams {
    width: u32,
    height: u32,
    stride: u32,
    sigma_normal: f32,
    sigma_depth: f32,
    sigma_luminance: f32,
    sigma_albedo: f32,
    _pad0: u32,
}

/// Compiled denoise kernels, reusable across frames.
pub struct Denoiser {
    layout: wgpu::BindGroupLayout,
    demodulate: wgpu::ComputePipeline,
    measure: wgpu::ComputePipeline,
    filter: wgpu::ComputePipeline,
    modulate: wgpu::ComputePipeline,
}

impl Denoiser {
    pub fn new(gpu: &Gpu) -> Result<Self, GpuError> {
        let device = &gpu.device;
        let module = gpu.create_shader("denoise/atrous.wgsl")?;
        let sto = |b: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("denoise"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                sto(1, true),
                sto(2, false),
                sto(3, false),
                sto(4, false),
                sto(5, false),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("denoise"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let stage = |entry_point: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Ok(Self {
            layout,
            demodulate: stage("demodulate"),
            measure: stage("measure"),
            filter: stage("atrous_pass"),
            modulate: stage("modulate"),
        })
    }

    /// Filter `accum` into a new buffer of RGBA pixels.
    ///
    /// Ping-pongs `a` and `b` across the à-trous passes by swapping which is
    /// bound where, so the shader always reads `buf_a` and writes `buf_b` and
    /// needs no notion of which pass it is on.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        accum: &wgpu::Buffer,
        width: u32,
        height: u32,
        params: &DenoiseParams,
        scratch: &DenoiseScratch,
    ) {
        let device = &gpu.device;
        let groups_x = width.div_ceil(8);
        let groups_y = height.div_ceil(8);

        let make_uniform = |stride: u32| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("denoise params"),
                contents: bytemuck::bytes_of(&GpuDenoiseParams {
                    width,
                    height,
                    stride,
                    sigma_normal: params.sigma_normal,
                    sigma_depth: params.sigma_depth,
                    sigma_luminance: params.sigma_luminance,
                    sigma_albedo: params.sigma_albedo,
                    _pad0: 0,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            })
        };
        let group = |u: &wgpu::Buffer, a: &wgpu::Buffer, b: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("denoise"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: u.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: accum.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: a.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: scratch.deviation.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: scratch.output.as_entire_binding(),
                    },
                ],
            })
        };
        let mut dispatch = |pipeline: &wgpu::ComputePipeline, g: &wgpu::BindGroup| {
            let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("denoise"),
                timestamp_writes: None,
            });
            p.set_pipeline(pipeline);
            p.set_bind_group(0, g, &[]);
            p.dispatch_workgroups(groups_x, groups_y, 1);
        };

        let u0 = make_uniform(1);
        let g0 = group(&u0, &scratch.a, &scratch.b);
        dispatch(&self.demodulate, &g0);
        dispatch(&self.measure, &g0);

        // Track which buffer currently holds the signal.
        let mut in_a = true;
        let mut uniforms = Vec::new();
        let mut groups = Vec::new();
        for level in 0..params.iterations {
            uniforms.push(make_uniform(1 << level));
        }
        for (level, u) in uniforms.iter().enumerate() {
            let _ = level;
            let (src, dst) = if in_a {
                (&scratch.a, &scratch.b)
            } else {
                (&scratch.b, &scratch.a)
            };
            groups.push(group(u, src, dst));
            in_a = !in_a;
        }
        for g in &groups {
            dispatch(&self.filter, g);
        }

        // `modulate` reads `buf_a`, so bind whichever buffer the last pass wrote
        // into that slot.
        let u_final = make_uniform(1);
        let g_final = if in_a {
            group(&u_final, &scratch.a, &scratch.b)
        } else {
            group(&u_final, &scratch.b, &scratch.a)
        };
        dispatch(&self.modulate, &g_final);
    }
}

/// Working buffers, sized to the render target.
pub struct DenoiseScratch {
    a: wgpu::Buffer,
    b: wgpu::Buffer,
    deviation: wgpu::Buffer,
    pub output: wgpu::Buffer,
}

impl DenoiseScratch {
    pub fn new(gpu: &Gpu, pixels: usize) -> Self {
        let zeros4 = vec![0.0f32; pixels.max(1) * 4];
        let zeros1 = vec![0.0f32; pixels.max(1)];
        let device = &gpu.device;
        let rw = |label: &str, bytes: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                // COPY_SRC so the result can be read back — the native harness
                // compares it against the CPU reference.
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
        };
        Self {
            a: rw("denoise a", bytemuck::cast_slice(&zeros4)),
            b: rw("denoise b", bytemuck::cast_slice(&zeros4)),
            deviation: rw("denoise deviation", bytemuck::cast_slice(&zeros1)),
            // Accum-shaped, so the display pass needs no change. The stride comes
            // from the struct rather than from a literal: it has been 16, 48 and
            // 64 bytes as the guide channels and then the sum of squares landed,
            // and a literal here fails as an out-of-bounds copy in whichever
            // unrelated test reads this buffer next.
            output: rw(
                "denoise output",
                &vec![0u8; pixels.max(1) * std::mem::size_of::<pt_core::gpu_layout::GpuAccum>()],
            ),
        }
    }
}
