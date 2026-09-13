//! Native wgpu harness.
//!
//! # Why this exists in a project whose host is TypeScript
//!
//! The browser is a poor place to debug a compute kernel. There is no frame
//! capture, no way to step a dispatch, and the only way to see what a shader
//! wrote is to read a buffer back into JavaScript and squint at it.
//!
//! This crate runs the **same WGSL files** the browser runs, natively, under
//! `cargo test`. That buys three things:
//!
//!   * the CPU-vs-GPU comparison (build step 3) happens entirely in `cargo`,
//!     with no browser round trip, so iteration is seconds rather than minutes;
//!   * kernels can be unit tested against known inputs — which is how the radix
//!     sort, the prefix scan and the Karras hierarchy build (step 11) will be
//!     validated, rather than by staring at a BVH heatmap;
//!   * RenderDoc and Metal frame capture work.
//!
//! The browser is then validated against *this*, and any difference is a wgpu
//! -versus-browser-WebGPU difference rather than an unknown.

pub mod block_on;
pub mod eval;
pub mod denoise;
pub mod lbvh;
pub mod shaders;
pub mod stats;
pub mod wavefront;

pub use block_on::block_on;

use glam::Vec3;
use pt_core::integrator::{build_uniforms, Film, RenderParams};
use pt_core::scenes::SceneDef;
use wgpu::util::DeviceExt;

/// Which GPU path-tracing architecture to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Architecture {
    /// One shader does the whole path. Simple, and limited by the register
    /// pressure of its heaviest branch and by dead lanes staying in the loop.
    Megakernel,
    /// One kernel per stage with compaction between them. See
    /// [`crate::wavefront`].
    #[default]
    Wavefront,
}

impl Architecture {
    pub const ALL: [Architecture; 2] = [Architecture::Megakernel, Architecture::Wavefront];

    pub fn name(self) -> &'static str {
        match self {
            Architecture::Megakernel => "megakernel",
            Architecture::Wavefront => "wavefront",
        }
    }

    pub fn parse(s: &str) -> Option<Architecture> {
        match s {
            "megakernel" | "mega" => Some(Architecture::Megakernel),
            "wavefront" | "wave" => Some(Architecture::Wavefront),
            _ => None,
        }
    }
}

/// Render with the chosen architecture. Both must produce the same image.
pub fn render_with(
    gpu: &Gpu,
    def: &SceneDef,
    params: &RenderParams,
    architecture: Architecture,
) -> Result<Film, GpuError> {
    match architecture {
        Architecture::Megakernel => render_native_on(gpu, def, params),
        Architecture::Wavefront => crate::wavefront::render_native_on(gpu, def, params),
    }
}

/// As [`render_with`], also reporting the measured relative standard error.
///
/// Both architectures accumulate the sum of squares, so both can report it; the
/// figures are directly comparable, which is one more way the two are checked
/// against each other.
pub fn render_with_measured(
    gpu: &Gpu,
    def: &SceneDef,
    params: &RenderParams,
    architecture: Architecture,
) -> Result<(Film, f32), GpuError> {
    let film = match architecture {
        Architecture::Megakernel => render_native_on_inner(gpu, def, params, None, true)?,
        Architecture::Wavefront => crate::wavefront::render_native_measured(gpu, def, params)?,
    };
    Ok((film, LAST_CONVERGENCE.with(|c| c.get())))
}

#[derive(Debug)]
pub enum GpuError {
    NoAdapter(String),
    NoDevice(String),
    Shader(String),
    Readback(String),
    /// A wgpu validation error raised asynchronously.
    ///
    /// These never come back through a `Result`: WebGPU reports them out of
    /// band, so a render whose bind group was rejected still "succeeds", still
    /// reads back a buffer, and still produces a timing. That is how a render
    /// doing nothing at all came to be reported as a 20x speedup.
    Validation(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::NoAdapter(e) => write!(
                f,
                "no GPU adapter available: {e}\n\
                 (on macOS this normally means Metal is unavailable; \
                  try WGPU_BACKEND=vulkan or run with --device cpu)"
            ),
            GpuError::NoDevice(e) => write!(f, "could not create a GPU device: {e}"),
            GpuError::Shader(e) => write!(f, "shader error: {e}"),
            GpuError::Readback(e) => write!(f, "buffer readback failed: {e}"),
            GpuError::Validation(e) => write!(f, "GPU validation error: {e}"),
        }
    }
}

impl std::error::Error for GpuError {}

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
    /// Validation errors collected from the uncaptured-error handler, so a
    /// render can fail on them rather than returning a plausible-looking image
    /// that was never drawn.
    errors: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
}

impl Gpu {
    pub fn new() -> Result<Self, GpuError> {
        // `_from_env` honours WGPU_BACKEND, so a Metal-versus-Vulkan comparison
        // on the same machine is an environment variable rather than a rebuild.
        // Backend differences in WGSL edge cases are real and worth checking.
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            ..Default::default()
        }))
        .map_err(|e| GpuError::NoAdapter(e.to_string()))?;

        let adapter_info = adapter.get_info();

        // `defaults()`, not `downlevel_defaults()`. The latter targets WebGL2-era
        // hardware and allows only 4 storage buffers per stage; WebGPU
        // guarantees 8, and this renderer binds exactly 8. Requesting the
        // downlevel set would make the native harness fail on a shader the
        // browser runs perfectly — the opposite of what this harness is for.
        //
        // Equally, it must not request *more* than the browser guarantees, or it
        // would validate shaders the browser will reject.
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("pt-gpu"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::defaults(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
            ..Default::default()
        }))
        .map_err(|e| GpuError::NoDevice(e.to_string()))?;

        // Surface validation errors loudly *and* record them. Printing alone is
        // not enough: stderr scrolls past, and every caller here returns a
        // `Result` that would still say Ok.
        let errors = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = errors.clone();
        device.on_uncaptured_error(std::sync::Arc::new(move |e| {
            let text = e.to_string();
            eprintln!("\nwgpu validation error:\n{text}\n");
            if let Ok(mut v) = sink.lock() {
                v.push(text);
            }
        }));

        Ok(Self {
            device,
            queue,
            adapter_info,
            errors,
        })
    }

    /// Fail if the device reported a validation error since the last check.
    ///
    /// Call after the work has been polled to completion — validation errors
    /// arrive during submit and during the poll, not when the command is
    /// recorded, so checking any earlier reliably finds nothing.
    pub fn check_errors(&self) -> Result<(), GpuError> {
        let mut v = match self.errors.lock() {
            Ok(v) => v,
            Err(p) => p.into_inner(),
        };
        if v.is_empty() {
            return Ok(());
        }
        let all = v.join("\n");
        v.clear();
        Err(GpuError::Validation(all))
    }

    /// Compile a shader, reporting WGSL errors as errors rather than letting
    /// them surface later as "pipeline is invalid".
    ///
    /// Without the explicit error scope, a WGSL compile failure produces an
    /// invalid module, which produces an invalid pipeline, which fails at
    /// `set_pipeline` — three layers away from the actual line of WGSL that is
    /// wrong. The error scope catches it at the source.
    pub fn create_shader(&self, entry: &str) -> Result<wgpu::ShaderModule, GpuError> {
        let source = shaders::resolve(entry).map_err(GpuError::Shader)?;
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(entry),
                source: wgpu::ShaderSource::Wgsl(source.as_str().into()),
            });
        if let Some(err) = block_on(scope.pop()) {
            return Err(GpuError::Shader(format!(
                "{entry} failed to compile:\n{err}\n\n\
                 (the resolved source is a concatenation of the //!include chain; \
                 run `cargo run -p pt-cli --bin dump-shader -- {entry}` to see it with line numbers)"
            )));
        }
        Ok(module)
    }
}

/// Storage buffers must be non-empty, and WebGPU requires binding sizes to be a
/// multiple of 4. A scene with no spheres still needs a bound buffer, so pad to
/// one element of zeroes and let `num_spheres == 0` stop the loop.
/// The two textures the environment sampler reads.
///
/// Textures rather than storage buffers, and that is forced rather than
/// preferred: the wavefront's SHADE stage already binds exactly eight storage
/// buffers, which is WebGPU's guaranteed per-stage limit counted across every
/// bind group. Sampled textures come from a separate budget.
pub(crate) struct EnvTextures {
    pub radiance: wgpu::TextureView,
    pub cdf: wgpu::TextureView,
}

/// Upload an environment map, or a 1x1 stand-in when the scene has none.
///
/// WebGPU has no optional bindings, so something must be bound either way. A
/// black 1x1 texture plus `env_width == 0` in the uniforms is cheaper than
/// compiling a second pipeline, and the shader's `env_present()` never reads it.
pub(crate) fn upload_envmap(gpu: &Gpu, env: &pt_core::envmap::EnvMap) -> EnvTextures {
    let device = &gpu.device;
    let make = |label: &str, w: u32, h: u32, format: wgpu::TextureFormat| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: w.max(1),
                height: h.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    };

    if env.is_empty() {
        let radiance = make("env radiance", 1, 1, wgpu::TextureFormat::Rgba32Float);
        let cdf = make("env cdf", 1, 1, wgpu::TextureFormat::R32Float);
        gpu.queue.write_texture(
            radiance.as_image_copy(),
            bytemuck::cast_slice(&[0.0f32; 4]),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(16),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        return EnvTextures {
            radiance: radiance.create_view(&Default::default()),
            cdf: cdf.create_view(&Default::default()),
        };
    }

    let (w, h) = (env.width, env.height);
    let radiance = make("env radiance", w, h, wgpu::TextureFormat::Rgba32Float);
    gpu.queue.write_texture(
        radiance.as_image_copy(),
        bytemuck::cast_slice(&env.pixels),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(w * 16),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );

    // One texture holds both CDFs: rows 0..h-1 are each row's conditional
    // distribution over columns, and row h is the marginal over rows. Packed
    // together because a 1-D CDF would otherwise need a binding of its own, and
    // the wide dimension is `max(w + 1, h + 1)` so the marginal always fits —
    // for an equirectangular map `w = 2h`, but nothing here relies on that.
    let cdf_w = (w + 1).max(h + 1);
    let cdf_h = h + 1;
    let mut data = vec![0.0f32; (cdf_w * cdf_h) as usize];
    for j in 0..h as usize {
        let src = j * (w as usize + 1);
        let dst = j * cdf_w as usize;
        data[dst..dst + w as usize + 1]
            .copy_from_slice(&env.conditional[src..src + w as usize + 1]);
    }
    let dst = h as usize * cdf_w as usize;
    data[dst..dst + h as usize + 1].copy_from_slice(&env.marginal);

    let cdf = make("env cdf", cdf_w, cdf_h, wgpu::TextureFormat::R32Float);
    gpu.queue.write_texture(
        cdf.as_image_copy(),
        bytemuck::cast_slice(&data),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(cdf_w * 4),
            rows_per_image: Some(cdf_h),
        },
        wgpu::Extent3d {
            width: cdf_w,
            height: cdf_h,
            depth_or_array_layers: 1,
        },
    );

    EnvTextures {
        radiance: radiance.create_view(&Default::default()),
        cdf: cdf.create_view(&Default::default()),
    }
}

pub(crate) fn storage_buffer(
    device: &wgpu::Device,
    label: &str,
    bytes: &[u8],
    stride: usize,
) -> wgpu::Buffer {
    let padded: Vec<u8> = if bytes.is_empty() {
        vec![0u8; stride]
    } else {
        bytes.to_vec()
    };
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: &padded,
        usage: wgpu::BufferUsages::STORAGE,
    })
}

use pt_core::gpu_layout::GpuAccum as Accum;

/// Render `def` natively with the megakernel and read the result back.
///
/// Uses exactly the same uniform block, scene buffers and WGSL as the browser,
/// so a difference between this and the browser is a platform difference, and a
/// difference between this and the CPU tracer is a light-transport bug.
pub fn render_native(def: &SceneDef, params: &RenderParams) -> Result<Film, GpuError> {
    let gpu = Gpu::new()?;
    render_native_on(&gpu, def, params)
}

pub fn render_native_on(
    gpu: &Gpu,
    def: &SceneDef,
    params: &RenderParams,
) -> Result<Film, GpuError> {
    render_native_on_inner(gpu, def, params, None, false)
}

/// As [`render_native_on`], but also runs the GPU denoiser and returns its
/// output instead.
///
/// Exists so the WGSL denoiser can be diffed against its Rust twin in
/// `cargo test`, which is the only way to know the two agree — the filtered
/// image looks plausible whatever the weights are doing.
pub fn render_native_denoised(
    gpu: &Gpu,
    def: &SceneDef,
    params: &RenderParams,
    denoise: &pt_core::denoise::DenoiseParams,
) -> Result<Film, GpuError> {
    render_native_on_inner(gpu, def, params, Some(denoise), false)
}

/// Report the measured relative standard error of a megakernel render.
///
/// See `shaders/stats/convergence.wgsl`. A convenience over
/// [`render_with_measured`] for callers that want the figure and not the image
/// — which is most of the tests.
pub fn measure_convergence(
    gpu: &Gpu,
    def: &SceneDef,
    params: &RenderParams,
) -> Result<f32, GpuError> {
    let (_, e) = render_with_measured(gpu, def, params, Architecture::Megakernel)?;
    Ok(e)
}

thread_local! {
    /// Where the render paths leave the figure for the wrappers above.
    ///
    /// A thread-local rather than a return value because every other caller of
    /// the inner function wants a `Film` and nothing else, and threading an
    /// `Option<&mut f32>` through for one caller's benefit is worse than this.
    pub(crate) static LAST_CONVERGENCE: std::cell::Cell<f32> = const { std::cell::Cell::new(-1.0) };
}

fn render_native_on_inner(
    gpu: &Gpu,
    def: &SceneDef,
    params: &RenderParams,
    denoise: Option<&pt_core::denoise::DenoiseParams>,
    measure: bool,
) -> Result<Film, GpuError> {
    let device = &gpu.device;
    let (w, h) = (params.width, params.height);
    let pixels = (w as usize) * (h as usize);

    let uniforms = build_uniforms(def, params);
    let module = gpu.create_shader("trace/megakernel.wgsl")?;

    let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let blob = &def.scene.blob;
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
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let env = upload_envmap(gpu, &def.scene.env);

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("megakernel"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("megakernel bindings"),
        layout: &pipeline.get_bind_group_layout(0),
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
                resource: primitives.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: lights.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: accum.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: positions.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: vertex_attrs.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: triangles.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: bvh_nodes.as_entire_binding(),
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

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("render"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("megakernel"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
    }

    // Convergence, when asked for. Recorded into the same command buffer, so it
    // costs one reduction over an accumulation the GPU has just finished writing
    // rather than a separate submit.
    let meter = measure.then(|| crate::stats::ConvergenceMeter::new(gpu)).transpose()?;
    if let Some(m) = meter.as_ref() {
        m.record(gpu, &mut encoder, &accum, pixels as u32);
    }

    // The denoiser runs over the finished accumulation, into its own buffer.
    // The accumulation is never written, which is what keeps denoising a
    // display-time product rather than something that biases the render.
    let scratch = denoise.map(|_| crate::denoise::DenoiseScratch::new(gpu, pixels));
    if let (Some(p), Some(scratch)) = (denoise, scratch.as_ref()) {
        let d = crate::denoise::Denoiser::new(gpu)?;
        d.run(gpu, &mut encoder, &accum, w, h, p, scratch);
    }

    // The denoiser's output is Accum-shaped, so the readback is identical
    // either way.
    let source = match scratch.as_ref() {
        Some(s) => &s.output,
        None => &accum,
    };
    let stride = std::mem::size_of::<Accum>();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (pixels * stride) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, staging.size());
    gpu.queue.submit([encoder.finish()]);

    let film = read_accum(gpu, &staging, w, h)?;
    if let Some(m) = meter.as_ref() {
        let e = m.read(gpu)?;
        LAST_CONVERGENCE.with(|c| c.set(e));
    }
    // After the poll, not before: validation errors arrive during submit and
    // during the poll, so an earlier check reliably finds nothing.
    gpu.check_errors()?;
    staging.unmap();
    Ok(film)
}

fn read_accum(gpu: &Gpu, staging: &wgpu::Buffer, w: u32, h: u32) -> Result<Film, GpuError> {
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
    for (i, a) in accum.iter().enumerate() {
        // Divide by the accumulated sample count here, exactly as the display
        // pass does, so the reference image and what the user sees come from the
        // same arithmetic.
        let n = a.samples.max(1.0);
        film.data[i] = Vec3::from_array(a.radiance) / n;
    }
    drop(data);
    Ok(film)
}
