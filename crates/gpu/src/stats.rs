//! Measuring how converged a render is, on the GPU.
//!
//! See `shaders/stats/convergence.wgsl` for what the number means. The host side
//! is a two-pass reduction and one four-byte readback.

use crate::{storage_buffer, Gpu, GpuError};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct StatsParams {
    pixels: u32,
    partials: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Workgroups in the first pass.
///
/// Fixed rather than derived from the pixel count: it bounds the partials buffer
/// and the second pass's loop, and 64 groups of the shader's 256 threads
/// saturate any GPU this runs on while keeping the fold trivially small.
const GROUPS: u32 = 64;

/// Compiled reduction kernels.
pub struct ConvergenceMeter {
    layout: wgpu::BindGroupLayout,
    reduce: wgpu::ComputePipeline,
    finish: wgpu::ComputePipeline,
    partials: wgpu::Buffer,
    result: wgpu::Buffer,
    staging: wgpu::Buffer,
}

impl ConvergenceMeter {
    pub fn new(gpu: &Gpu) -> Result<Self, GpuError> {
        let device = &gpu.device;
        let module = gpu.create_shader("stats/convergence.wgsl")?;
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
            label: Some("convergence"),
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
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("convergence"),
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
        let zeros = vec![0.0f32; (GROUPS * 2) as usize];
        Ok(Self {
            layout,
            reduce: stage("reduce"),
            finish: stage("finish"),
            partials: storage_buffer(device, "partials", bytemuck::cast_slice(&zeros), 8),
            result: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("convergence result"),
                contents: bytemuck::bytes_of(&-1.0f32),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            }),
            staging: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("convergence readback"),
                size: 4,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
        })
    }

    /// Record the reduction and copy the result to the staging buffer.
    pub fn record(&self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, accum: &wgpu::Buffer, pixels: u32) {
        let params = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("stats params"),
                contents: bytemuck::bytes_of(&StatsParams {
                    pixels,
                    partials: GROUPS,
                    _pad0: 0,
                    _pad1: 0,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("convergence"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: accum.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.partials.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.result.as_entire_binding(),
                },
            ],
        });
        for (pipeline, groups) in [(&self.reduce, GROUPS), (&self.finish, 1)] {
            let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("convergence"),
                timestamp_writes: None,
            });
            p.set_pipeline(pipeline);
            p.set_bind_group(0, &group, &[]);
            p.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.result, 0, &self.staging, 0, 4);
    }

    /// Read the recorded result. Blocks; the browser reads it asynchronously
    /// instead so a stat never stalls a frame.
    pub fn read(&self, gpu: &Gpu) -> Result<f32, GpuError> {
        let slice = self.staging.slice(..);
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
        let v = f32::from_ne_bytes([data[0], data[1], data[2], data[3]]);
        drop(data);
        self.staging.unmap();
        Ok(v)
    }
}
