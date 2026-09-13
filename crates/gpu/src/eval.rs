//! Run a shader function over an array of inputs and read the results back.
//!
//! This is the mechanism for testing WGSL *numerics* rather than WGSL images. A
//! transposed colour matrix, a mistyped polynomial coefficient, or a subtly
//! different epsilon all produce a plausible-looking render and are invisible to
//! the eye — but they show up immediately when the shader function is evaluated
//! at a hundred known inputs and diffed against its Rust twin.
//!
//! Build step 7 (chi-squared sampling tests) needs exactly this to check that a
//! BSDF's `sample()` and `pdf()` agree on the GPU, so the harness is written
//! generically rather than for tone mapping alone.

use crate::{Gpu, GpuError};
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EvalParams {
    mode: u32,
    count: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Evaluate `shader`'s `main` over `inputs`, selecting a function with `mode`.
///
/// Inputs and outputs are `vec4` because that is what a storage buffer wants to
/// be aligned to; the fourth component is unused.
pub fn run_vec3_kernel(
    gpu: &Gpu,
    shader: &str,
    mode: u32,
    inputs: &[Vec3],
) -> Result<Vec<Vec3>, GpuError> {
    let padded: Vec<[f32; 4]> = inputs.iter().map(|v| [v.x, v.y, v.z, 0.0]).collect();
    run_vec4_kernel(gpu, shader, mode, &padded)
}

/// As [`run_vec3_kernel`], but the fourth component is passed through — some
/// kernels need a fourth parameter (a second random dimension, for instance).
pub fn run_vec4_kernel(
    gpu: &Gpu,
    shader: &str,
    mode: u32,
    padded: &[[f32; 4]],
) -> Result<Vec<Vec3>, GpuError> {
    let inputs = padded;
    let device = &gpu.device;
    let module = gpu.create_shader(shader)?;

    let bytes = (padded.len() * 16) as u64;

    let params = EvalParams {
        mode,
        count: inputs.len() as u32,
        _pad0: 0,
        _pad1: 0,
    };
    let param_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("eval params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("eval inputs"),
        contents: bytemuck::cast_slice(padded),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("eval outputs"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(shader),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("eval bindings"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("eval"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("eval"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((inputs.len() as u32).div_ceil(64), 1, 1);
    }

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("eval readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&output_buf, 0, &staging, 0, bytes);
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
    let raw: &[[f32; 4]] = bytemuck::cast_slice(&data);
    let out = raw.iter().map(|v| Vec3::new(v[0], v[1], v[2])).collect();
    drop(data);
    staging.unmap();
    Ok(out)
}
