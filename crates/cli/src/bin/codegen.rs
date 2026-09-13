//! Emit the generated mirrors of `pt_core::gpu_layout` for WGSL and TypeScript.
//!
//! ```text
//! cargo run -p pt-cli --bin codegen            # write the files
//! cargo run -p pt-cli --bin codegen -- --check # fail if they are stale (CI)
//! ```
//!
//! This is the mechanism that makes the hybrid architecture safe. Rust owns the
//! byte layout; WGSL and TypeScript receive it. Without this step the same
//! struct would be hand-written in three languages and would eventually — not
//! hypothetically — disagree by four bytes, producing a render that is wrong in
//! a way no test at the image level can localise.

use pt_core::camera::Camera;
use pt_core::gpu_layout::{
    GpuBvhNode, GpuDispatchArgs, GpuHitRecord, GpuLight, GpuMaterial, GpuPathState, GpuPrimitive,
    GpuShadowRay, GpuTriangle, GpuUniforms, GpuVertexAttr, GpuWavefrontCounters, WGSL_STRUCTS,
};

use glam::Vec3;
use pt_core::diagnostic::RenderMode;
use pt_core::integrator::SamplingMode;
use pt_core::sobol::SamplerKind;
use pt_core::tonemap::Tonemap;
use pt_core::scenes;
use std::mem::{offset_of, size_of};
use std::path::Path;

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let check = std::env::args().any(|a| a == "--check");
    // The binary runs from the workspace root under `cargo run`, but resolve
    // relative to the manifest so it also works from a subdirectory.
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .ok_or("cannot locate the workspace root")?
        .to_path_buf();

    let files = [
        (
            root.join("shaders/common/generated.wgsl"),
            format!("{WGSL_STRUCTS}\n{}", sobol_wgsl()),
        ),
        (
            root.join("shaders/common/ggx_energy.wgsl"),
            ggx_energy_wgsl(),
        ),
        (root.join("web/src/generated/layout.ts"), layout_ts()),
        (root.join("web/src/generated/scenes.ts"), scenes_ts(&root)?),
    ];

    let mut stale = Vec::new();
    for (path, contents) in &files {
        let current = std::fs::read_to_string(path).ok();
        if current.as_deref() == Some(contents.as_str()) {
            continue;
        }
        if check {
            stale.push(path.clone());
        } else {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| format!("{parent:?}: {e}"))?;
            }
            std::fs::write(path, contents).map_err(|e| format!("{path:?}: {e}"))?;
            println!("wrote {}", path.display());
        }
    }

    if check {
        if stale.is_empty() {
            println!("generated files are up to date");
        } else {
            for p in &stale {
                eprintln!("stale: {}", p.display());
            }
            return Err(format!(
                "{} generated file(s) are out of date — run `cargo run -p pt-cli --bin codegen`",
                stale.len()
            ));
        }
    }
    Ok(())
}

/// Bake the GGX directional-albedo table into WGSL as a `const` array.
///
/// A `const` array can be indexed with a runtime value and costs **no binding**
/// — which is the deciding factor here, because the renderer already uses all
/// eight storage buffers WebGPU guarantees. The alternatives were worse: a
/// `var<private>` array is per-invocation storage (4 KB per thread), and a
/// texture would need a filterable float format that core WebGPU does not
/// provide.
///
/// The values come from `pt_core`, so the CPU and the shader share one table by
/// construction rather than by two implementations agreeing.
fn ggx_energy_wgsl() -> String {
    use pt_core::bsdf::energy;
    let table = energy::table();
    let n = energy::TABLE_SIZE;

    let mut s = String::from(HEADER);
    s.push_str(&format!(
        "\n// Directional albedo of the white-Fresnel single-scattering GGX lobe.\n\
         //\n\
         // Row index is roughness, linear in [0, 1]. Column index is\n\
         // **sqrt(cos_theta_o)**, not cos_theta_o: E is flat over most of the\n\
         // hemisphere and drops steeply in the last few degrees before grazing, so\n\
         // uniform spacing in cos wastes resolution on the flat part and then\n\
         // interpolates across the steep part.\n\
         //\n\
         // Indexed at runtime, which WGSL permits for a `const` array and which\n\
         // costs no binding — the renderer already uses all eight storage buffers\n\
         // WebGPU guarantees.\n\n\
         const GGX_E_SIZE: u32 = {n}u;\n\
         const GGX_E: array<f32, {}> = array<f32, {}>(\n",
        n * n,
        n * n
    ));
    for row in 0..n {
        s.push_str("  ");
        for col in 0..n {
            s.push_str(&format!("{:.6}, ", table[row * n + col]));
        }
        s.push_str(&format!(
            "// roughness {:.3}\n",
            row as f32 / (n - 1) as f32
        ));
    }
    s.push_str(");\n");
    s
}

const HEADER: &str = "// @generated by `cargo run -p pt-cli --bin codegen` — do not edit.\n\
                      // Source of truth: crates/core/src/gpu_layout.rs\n";

/// Byte offsets and sizes, plus the one piece of packing code the host actually
/// needs to write by hand each frame: the uniform block.
fn layout_ts() -> String {
    let mut s = String::from(HEADER);
    s.push_str(
        "\n// Every offset below is emitted from Rust's `offset_of!`, so it cannot drift\n\
         // from what the shader and the CPU reference tracer agree on.\n\n",
    );

    s.push_str(&format!(
        "export const MATERIAL_SIZE = {};\nexport const PRIMITIVE_SIZE = {};\nexport const LIGHT_SIZE = {};\nexport const UNIFORMS_SIZE = {};\n\n",
        size_of::<GpuMaterial>(),
        size_of::<GpuPrimitive>(),
        size_of::<GpuLight>(),
        size_of::<GpuUniforms>()
    ));

    // The wavefront's per-path buffers.
    //
    // These were literals in `web/src/wavefront.ts` under a comment naming this
    // file as the source of truth, which is the worst of both worlds: it reads
    // as generated and drifts like a copy. `GpuPathState` grew from 80 to 112
    // bytes when the denoiser's guide channels moved into it, the literal stayed
    // at 80, and the browser's wavefront then allocated 71% of the path pool it
    // indexed. It did not fail loudly — WGSL clamps an out-of-bounds index, so
    // paths collided on the last valid slot and the renderer produced a noisy
    // image that was simply wrong, differing from its own megakernel by 446%.
    //
    // Emitted here so `codegen --check` fails the next time one of them grows.
    s.push_str(&format!(
        "// The wavefront's per-path buffer strides.\n\
         export const PATH_STATE_SIZE = {};\n\
         export const HIT_RECORD_SIZE = {};\n\
         export const SHADOW_RAY_SIZE = {};\n\
         export const WAVEFRONT_COUNTERS_SIZE = {};\n\
         export const DISPATCH_ARGS_SIZE = {};\n\n",
        size_of::<GpuPathState>(),
        size_of::<GpuHitRecord>(),
        size_of::<GpuShadowRay>(),
        size_of::<GpuWavefrontCounters>(),
        size_of::<GpuDispatchArgs>()
    ));

    // Wire indices for every enum the host has to turn into a number.
    //
    // These were four hand-kept tables under "must match `X::index()`"
    // comments. All four happened to be right, which is exactly what the
    // wavefront's path-state stride also was, for two build steps, under the
    // same kind of comment. A shader reads these to pick a branch, so a drifted
    // one does not fail — it silently renders the wrong mode under the right
    // label. The union type is emitted alongside so a rename cannot leave a
    // stale key behind either.
    for (ts_name, variants) in [
        (
            "SamplingMode",
            SamplingMode::ALL.iter().map(|v| (v.name(), v.index())).collect::<Vec<_>>(),
        ),
        (
            "SamplerKind",
            SamplerKind::ALL.iter().map(|v| (v.name(), v.index())).collect(),
        ),
        (
            "DiagnosticMode",
            RenderMode::all().iter().map(|v| (v.name(), v.index())).collect(),
        ),
        (
            "Tonemap",
            Tonemap::ALL.iter().map(|v| (v.name(), v.index())).collect(),
        ),
    ] {
        let union = variants
            .iter()
            .map(|(n, _)| format!("'{n}'"))
            .collect::<Vec<_>>()
            .join(" | ");
        s.push_str(&format!("export type {ts_name} = {union};\n"));
        s.push_str(&format!(
            "export const {}_INDEX: Record<{ts_name}, number> = {{ ",
            ts_name
                .chars()
                .flat_map(|c| if c.is_uppercase() {
                    vec!['_', c]
                } else {
                    vec![c.to_ascii_uppercase()]
                })
                .collect::<String>()
                .trim_start_matches('_')
        ));
        s.push_str(
            &variants
                .iter()
                .map(|(n, i)| format!("{n}: {i}"))
                .collect::<Vec<_>>()
                .join(", "),
        );
        s.push_str(" };\n");
    }
    s.push('\n');

    s.push_str("export const UNIFORM_OFFSET = {\n");
    for (name, off) in [
        ("camOrigin", offset_of!(GpuUniforms, cam_origin)),
        ("lensRadius", offset_of!(GpuUniforms, lens_radius)),
        ("camUpperLeft", offset_of!(GpuUniforms, cam_upper_left)),
        ("focusDistance", offset_of!(GpuUniforms, focus_distance)),
        ("camHorizontal", offset_of!(GpuUniforms, cam_horizontal)),
        ("camVertical", offset_of!(GpuUniforms, cam_vertical)),
        ("width", offset_of!(GpuUniforms, width)),
        ("height", offset_of!(GpuUniforms, height)),
        ("sampleOffset", offset_of!(GpuUniforms, sample_offset)),
        (
            "samplesPerLaunch",
            offset_of!(GpuUniforms, samples_per_launch),
        ),
        ("maxDepth", offset_of!(GpuUniforms, max_depth)),
        ("frameSeed", offset_of!(GpuUniforms, frame_seed)),
        ("numPrimitives", offset_of!(GpuUniforms, num_primitives)),
        ("numLights", offset_of!(GpuUniforms, num_lights)),
        ("samplingMode", offset_of!(GpuUniforms, sampling_mode)),
        ("background", offset_of!(GpuUniforms, background)),
        ("numTriangles", offset_of!(GpuUniforms, num_triangles)),
        ("numBvhNodes", offset_of!(GpuUniforms, num_bvh_nodes)),
        ("renderMode", offset_of!(GpuUniforms, render_mode)),
        ("envWidth", offset_of!(GpuUniforms, env_width)),
        ("envHeight", offset_of!(GpuUniforms, env_height)),
        ("envTotalWeight", offset_of!(GpuUniforms, env_total_weight)),
        ("samplerKind", offset_of!(GpuUniforms, sampler_kind)),
        ("numInstances", offset_of!(GpuUniforms, num_instances)),
        ("tlasRoot", offset_of!(GpuUniforms, tlas_root)),
    ] {
        s.push_str(&format!("  {name}: {off},\n"));
    }
    s.push_str("} as const;\n\n");

    // Emitted rather than hand-written on the web side: the accumulation buffer
    // grew from 16 bytes to 48 when the denoiser's guide channels were appended
    // to it, and a stale stride there is not a compile error — it is a garbled
    // image.
    s.push_str(&format!(
        "/** Bytes per pixel of the accumulation buffer, from Rust's `GpuAccum`. */\n\
         export const ACCUM_BYTES_PER_PIXEL = {};\n\n",
        size_of::<pt_core::gpu_layout::GpuAccum>()
    ));

    s.push_str(
        "export interface UniformValues {\n\
         \x20 camOrigin: readonly [number, number, number];\n\
         \x20 camUpperLeft: readonly [number, number, number];\n\
         \x20 camHorizontal: readonly [number, number, number];\n\
         \x20 camVertical: readonly [number, number, number];\n\
         \x20 lensRadius: number;\n\
         \x20 focusDistance: number;\n\
         \x20 width: number;\n\
         \x20 height: number;\n\
         \x20 sampleOffset: number;\n\
         \x20 samplesPerLaunch: number;\n\
         \x20 maxDepth: number;\n\
         \x20 frameSeed: number;\n\
         \x20 numPrimitives: number;\n\
         \x20 numLights: number;\n\
         \x20 samplingMode: number;\n\
         \x20 background: readonly [number, number, number];\n\
         \x20 envWidth: number;\n\
         \x20 envHeight: number;\n\
         \x20 envTotalWeight: number;\n\
         \x20 samplerKind: number;\n\
         \x20 numInstances: number;\n\
         \x20 tlasRoot: number;\n\
         \x20 numTriangles: number;\n\
         \x20 numBvhNodes: number;\n\
         \x20 renderMode: number;\n\
         }\n\n\
         /** Pack a uniform block into `UNIFORMS_SIZE` bytes, little-endian. */\n\
         export function writeUniforms(values: UniformValues): ArrayBuffer {\n\
         \x20 const buf = new ArrayBuffer(UNIFORMS_SIZE);\n\
         \x20 const f32 = new Float32Array(buf);\n\
         \x20 const u32 = new Uint32Array(buf);\n\
         \x20 const O = UNIFORM_OFFSET;\n\
         \x20 const vec3 = (off: number, v: readonly [number, number, number]) => {\n\
         \x20   f32[off / 4] = v[0]; f32[off / 4 + 1] = v[1]; f32[off / 4 + 2] = v[2];\n\
         \x20 };\n\
         \x20 vec3(O.camOrigin, values.camOrigin);\n\
         \x20 vec3(O.camUpperLeft, values.camUpperLeft);\n\
         \x20 vec3(O.camHorizontal, values.camHorizontal);\n\
         \x20 vec3(O.camVertical, values.camVertical);\n\
         \x20 vec3(O.background, values.background);\n\
         \x20 f32[O.lensRadius / 4] = values.lensRadius;\n\
         \x20 f32[O.focusDistance / 4] = values.focusDistance;\n\
         \x20 u32[O.width / 4] = values.width;\n\
         \x20 u32[O.height / 4] = values.height;\n\
         \x20 u32[O.sampleOffset / 4] = values.sampleOffset;\n\
         \x20 u32[O.samplesPerLaunch / 4] = values.samplesPerLaunch;\n\
         \x20 u32[O.maxDepth / 4] = values.maxDepth;\n\
         \x20 u32[O.frameSeed / 4] = values.frameSeed;\n\
         \x20 u32[O.numPrimitives / 4] = values.numPrimitives;\n\
         \x20 u32[O.numLights / 4] = values.numLights;\n\
         \x20 u32[O.samplingMode / 4] = values.samplingMode;\n\
         \x20 u32[O.numTriangles / 4] = values.numTriangles;\n\
         \x20 u32[O.numBvhNodes / 4] = values.numBvhNodes;\n\
         \x20 u32[O.renderMode / 4] = values.renderMode;\n\
         \x20 u32[O.envWidth / 4] = values.envWidth;\n\
         \x20 u32[O.envHeight / 4] = values.envHeight;\n\
         \x20 f32[O.envTotalWeight / 4] = values.envTotalWeight;\n\
         \x20 u32[O.samplerKind / 4] = values.samplerKind;\n\
         \x20 u32[O.numInstances / 4] = values.numInstances;\n\
         \x20 u32[O.tlasRoot / 4] = values.tlasRoot;\n\
         \x20 return buf;\n\
         }\n",
    );
    s
}

/// 64-bit FNV-1a over a packed scene, as lower-case hex.
///
/// This is a **cache key and a corruption check**, not a security measure, and
/// the distinction is worth being explicit about. Authenticity of the download
/// comes from TLS; what this catches is a truncated or half-written transfer,
/// and a scene whose contents changed while its name did not — which is the
/// case that would otherwise leave a stale 26 MB blob in a user's IndexedDB
/// forever.
///
/// FNV-1a rather than SHA-256 because SHA-256 would be either a new dependency
/// or eighty lines of hand-rolled compression function, and neither buys
/// anything here: a 64-bit digest over a handful of assets has a collision
/// probability far below the odds of the rest of the pipeline being wrong.
fn content_hash(bytes: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}

/// Largest packed scene that is shipped to the browser as a committed asset.
///
/// The BVH stress scene is about 10 MB packed, which has no business in a git
/// repository or a Pages deploy. Scenes above this threshold stay available to
/// the CLI and the native tests — where they are generated in memory and cost
/// nothing — and return to the browser when the CDN plus IndexedDB asset
/// pipeline lands. The threshold is checked rather than hardcoding a list, so a
/// scene that grows past it drops out automatically instead of silently bloating
/// the deploy.
const MAX_BROWSER_SCENE_BYTES: usize = 2 * 1024 * 1024;

/// Arrays are concatenated at this alignment so every view can be constructed
/// without copying.
const ASSET_ALIGN: usize = 16;

struct PackedScene {
    bytes: Vec<u8>,
    /// (field name, byte offset, element count) in emission order.
    sections: Vec<(&'static str, usize, usize)>,
}

/// The Sobol direction numbers, as a WGSL constant.
///
/// Emitted from `pt_core::sobol` rather than transcribed, so the shader and the
/// CPU reference cannot disagree about a single word — and a wrong word would
/// not crash, it would quietly stop the sequence being a (0, 2)-sequence.
fn sobol_wgsl() -> String {
    let v = pt_core::sobol::direction_numbers();
    let mut s = String::from(
        "// Sobol direction numbers, generated by `codegen` from crates/core/src/sobol.rs.\n\
         // v[dim][k] is XORed into the running value when bit k of the index is set.\n",
    );
    s.push_str(&format!(
        "const SOBOL_DIMENSIONS: u32 = {}u;\nconst SOBOL_BITS: u32 = {}u;\n",
        pt_core::sobol::SOBOL_DIMENSIONS,
        pt_core::sobol::SOBOL_BITS,
    ));
    s.push_str(&format!(
        "const SOBOL_V: array<array<u32, {}>, {}> = array(\n",
        pt_core::sobol::SOBOL_BITS,
        pt_core::sobol::SOBOL_DIMENSIONS,
    ));
    for row in v.iter() {
        s.push_str("  array<u32, 32>(");
        for (k, x) in row.iter().enumerate() {
            if k > 0 {
                s.push_str(", ");
            }
            s.push_str(&format!("0x{x:08x}u"));
        }
        s.push_str("),\n");
    }
    s.push_str(");\n");
    s
}

/// Concatenate a scene's buffers into one 16-byte-aligned blob.
fn pack_scene(def: &scenes::SceneDef) -> PackedScene {
    let b = &def.scene.blob;
    let env = &def.scene.env;
    // Instances ride in the primitive array; see `SceneBlob::primitives_upload`.
    let prim_upload = b.primitives_upload();
    let prim_bytes: &[u8] = bytemuck::cast_slice(&prim_upload);

    // The environment map travels with the scene rather than being regenerated
    // in TypeScript. `procedural_sky` and the CDF build are exact arithmetic,
    // and reimplementing them on the web side would create a second source of
    // truth for a distribution that has to agree with the native harness texel
    // for texel — which is exactly what shipping the bytes avoids.
    //
    // The CDF is packed into one rectangle the same way the GPU texture is:
    // rows 0..h-1 are the per-row conditionals and row h is the marginal. Doing
    // the packing here rather than in TypeScript keeps the layout in one place.
    let (cdf_w, cdf_h) = env_cdf_dims(env);
    let mut cdf = vec![0.0f32; cdf_w * cdf_h];
    if !env.is_empty() {
        let (w, h) = (env.width as usize, env.height as usize);
        for j in 0..h {
            let src = j * (w + 1);
            let dst = j * cdf_w;
            cdf[dst..dst + w + 1].copy_from_slice(&env.conditional[src..src + w + 1]);
        }
        let dst = h * cdf_w;
        cdf[dst..dst + h + 1].copy_from_slice(&env.marginal);
    }
    let cdf_bytes: &[u8] = bytemuck::cast_slice(&cdf);
    let env_bytes: &[u8] = bytemuck::cast_slice(&env.pixels);

    let parts: [(&'static str, &[u8], usize); 9] = [
        ("materials", b.materials_bytes(), b.materials.len()),
        ("primitives", prim_bytes, prim_upload.len()),
        ("lights", b.lights_bytes(), b.lights.len()),
        ("positions", b.positions_bytes(), b.positions.len()),
        ("vertexAttrs", b.vertex_attrs_bytes(), b.vertex_attrs.len()),
        ("triangles", b.triangles_bytes(), b.triangles.len()),
        ("bvhNodes", b.bvh_nodes_bytes(), b.bvh_nodes.len()),
        ("envRadiance", env_bytes, env.pixels.len()),
        ("envCdf", cdf_bytes, cdf.len()),
    ];

    let mut bytes = Vec::new();
    let mut sections = Vec::new();
    for (name, data, count) in parts {
        while bytes.len() % ASSET_ALIGN != 0 {
            bytes.push(0);
        }
        sections.push((name, bytes.len(), count));
        bytes.extend_from_slice(data);
    }
    PackedScene { bytes, sections }
}

/// How far the camera can see in this scene, for the depth diagnostic.
///
/// The diagonal of everything's bounds, from the camera. Taken from the scene
/// rather than fixed so one depth image reads the same for a Cornell box
/// measured in hundreds and a sphere measured in ones.
fn scene_depth_scale(def: &scenes::SceneDef) -> f32 {
    let b = &def.scene.blob;
    let mut bounds = pt_core::bvh::Aabb::default();
    for t in &b.triangles {
        bounds.grow(&pt_core::bvh::triangle_bounds(t, &b.positions));
    }
    for p in &b.primitives {
        let c = Vec3::from_array(p.position);
        if p.is_quad() {
            bounds.grow_point(c);
            bounds.grow_point(c + Vec3::from_array(p.edge_u) + Vec3::from_array(p.edge_v));
        } else {
            bounds.grow_point(c - Vec3::splat(p.radius));
            bounds.grow_point(c + Vec3::splat(p.radius));
        }
    }
    if bounds.is_empty() {
        return 1.0;
    }
    let far = (bounds.max - def.camera.eye).length();
    let near = (bounds.min - def.camera.eye).length();
    far.max(near).max(1.0e-3)
}

/// Dimensions of the packed CDF rectangle. Wide enough that the marginal, which
/// occupies the last row, always fits: for an equirectangular map `w = 2h`, but
/// nothing relies on that.
fn env_cdf_dims(env: &pt_core::envmap::EnvMap) -> (usize, usize) {
    if env.is_empty() {
        return (0, 0);
    }
    let (w, h) = (env.width as usize, env.height as usize);
    ((w + 1).max(h + 1), h + 1)
}

/// Scene geometry travels to the browser as **pre-packed binary**, not as JSON
/// that TypeScript then has to lay out.
///
/// There is consequently no TypeScript code that could pack a `Sphere` or a
/// `BvhNode` wrongly — the only hand-written packing on the host side is the
/// uniform block, whose offsets are generated and whose one duplicated
/// computation (the camera basis) is covered by the fixtures below.
fn scenes_ts(root: &Path) -> Result<String, String> {
    let asset_dir = root.join("web/public/scenes");
    std::fs::create_dir_all(&asset_dir).map_err(|e| format!("{asset_dir:?}: {e}"))?;

    let mut s = String::from(HEADER);
    s.push_str(
        "\n// Scene buffers are emitted as pre-packed binary assets, fetched at runtime.\n\
         // TypeScript never lays out a scene struct, so it cannot disagree with the\n\
         // WGSL about where a field lives.\n\n\
         export interface CameraDef {\n\
         \x20 eye: [number, number, number];\n\
         \x20 lookAt: [number, number, number];\n\
         \x20 up: [number, number, number];\n\
         \x20 vfovDeg: number;\n\
         \x20 aperture: number;\n\
         \x20 focusDistance: number;\n\
         }\n\n\
         /** Byte offset and element count of one array inside a scene asset. */\n\
         export interface Section {\n\
         \x20 byteOffset: number;\n\
         \x20 count: number;\n\
         }\n\n\
         export interface SceneManifest {\n\
         \x20 name: string;\n\
         \x20 description: string;\n\
         \x20 /** For a local scene, a path under the site base. For a remote one,\n\
         \x20  *  a content-addressed filename under the asset base. */\n\
         \x20 asset: string;\n\
         \x20 byteLength: number;\n\
         \x20 /** Non-null when the scene is too large to commit and is fetched from\n\
         \x20  *  the CDN and cached in IndexedDB rather than served with the site. */\n\
         \x20 remote: { hash: string; file: string; megabytes: number } | null;\n\
         \x20 sections: {\n\
         \x20   materials: Section; primitives: Section; lights: Section;\n\
         \x20   positions: Section; vertexAttrs: Section; triangles: Section;\n\
         \x20   bvhNodes: Section;\n\
         \x20   envRadiance: Section; envCdf: Section;\n\
         \x20 };\n\
         \x20 /** Scene extent, for normalising the depth diagnostic. */\n\
         \x20 depthScale: number;\n\
         \x20 /** Instances appended to the primitives section, after the analytic ones. */\n\
         \x20 instancing: { count: number; analyticPrimitives: number; tlasRoot: number };\n\
         \x20 /** Equirectangular sky. `width === 0` means the scene has none. */\n\
         \x20 env: { width: number; height: number; totalWeight: number };\n\
         \x20 background: [number, number, number];\n\
         \x20 camera: CameraDef;\n\
         }\n\n",
    );

    let mut excluded: Vec<(String, usize)> = Vec::new();
    let mut entries = String::new();

    for def in scenes::all() {
        let packed = pack_scene(&def);
        let remote = packed.bytes.len() > MAX_BROWSER_SCENE_BYTES;
        let hash = content_hash(&packed.bytes);

        // Content-addressed when remote, so publishing a changed scene adds a
        // file rather than replacing one — every cached copy stays valid and
        // every manifest keeps pointing at the bytes it was generated against.
        // Remote blobs land beside the committed ones but under a
        // content-addressed name, which `.gitignore` excludes by pattern. So
        // they exist for `npm run dev` and for a local `vite build`, and never
        // enter the repository — the Pages workflow pulls them from the release
        // into the built site instead.
        let basename = if remote {
            format!("{}-{}.bin", def.name, hash)
        } else {
            format!("{}.bin", def.name)
        };
        let file = asset_dir.join(&basename);
        let asset = format!("scenes/{basename}");
        if remote {
            excluded.push((def.name.to_string(), packed.bytes.len()));
        }
        // Only rewrite when the content changes, so `--check` and incremental
        // builds do not churn.
        if std::fs::read(&file).ok().as_deref() != Some(packed.bytes.as_slice()) {
            std::fs::write(&file, &packed.bytes).map_err(|e| format!("{file:?}: {e}"))?;
            println!("wrote {}", file.display());
        }

        let c = &def.camera;
        entries.push_str(&format!(
            "  {{\n    name: {:?},\n    description: {:?},\n    asset: {:?},\n    byteLength: {},\n    remote: {},\n    sections: {{\n",
            def.name,
            def.description,
            asset,
            packed.bytes.len(),
            if remote {
                format!(
                    "{{ hash: {:?}, file: {:?}, megabytes: {:.1} }}",
                    hash,
                    basename,
                    packed.bytes.len() as f64 / (1024.0 * 1024.0)
                )
            } else {
                "null".to_string()
            }
        ));
        for (name, offset, count) in &packed.sections {
            entries.push_str(&format!(
                "      {name}: {{ byteOffset: {offset}, count: {count} }},\n"
            ));
        }
        entries.push_str(&format!(
            "    }},\n    depthScale: {},\n    instancing: {{ count: {}, analyticPrimitives: {}, tlasRoot: {} }},\n    env: {{ width: {}, height: {}, totalWeight: {} }},\n    background: [{}, {}, {}],\n    camera: {{ eye: [{}, {}, {}], lookAt: [{}, {}, {}], up: [{}, {}, {}], vfovDeg: {}, aperture: {}, focusDistance: {} }},\n  }},\n",
            scene_depth_scale(&def),
            def.scene.blob.instances.len(),
            def.scene.blob.primitives.len(),
            def.scene.tlas_root,
            def.scene.env.width,
            def.scene.env.height,
            def.scene.env.total_weight,
            def.background.x, def.background.y, def.background.z,
            c.eye.x, c.eye.y, c.eye.z,
            c.look_at.x, c.look_at.y, c.look_at.z,
            c.up.x, c.up.y, c.up.z,
            c.vfov_deg, c.aperture, c.focus_distance,
        ));
    }

    s.push_str("export const SCENES: SceneManifest[] = [\n");
    s.push_str(&entries);
    s.push_str("];\n\n");

    s.push_str(
        "/** Scenes fetched from the asset host rather than served with the site.\n\
         \x20*  Too large to commit, so they are downloaded once and cached in\n\
         \x20*  IndexedDB. Same entries as `SCENES.filter(s => s.remote)`, kept\n\
         \x20*  separately so the UI can talk about them without a scan. */\n\
         export const REMOTE_SCENES: { name: string; megabytes: number }[] = [\n",
    );
    for (name, bytes) in &excluded {
        s.push_str(&format!(
            "  {{ name: {name:?}, megabytes: {:.1} }},\n",
            *bytes as f64 / (1024.0 * 1024.0)
        ));
    }
    s.push_str("];\n\n");

    // Strides for slicing a packed asset, so `sceneLoader.ts` stops carrying its
    // own copy of them. The wavefront's path-state stride was a literal under a
    // comment naming this file as the source of truth, and it silently drifted
    // by 32 bytes for two build steps.
    s.push_str("/** Element sizes inside a packed scene asset. */\nexport const SCENE_STRIDE = {\n");
    for (name, stride) in [
        ("materials", size_of::<GpuMaterial>()),
        ("primitives", size_of::<GpuPrimitive>()),
        ("lights", size_of::<GpuLight>()),
        ("positions", 16),
        ("vertexAttrs", size_of::<GpuVertexAttr>()),
        ("triangles", size_of::<GpuTriangle>()),
        ("bvhNodes", size_of::<GpuBvhNode>()),
        // Format-defined rather than struct-defined: rgba32float radiance, and
        // r32float for the packed CDF rectangle.
        ("envRadiance", 16),
        ("envCdf", 4),
    ] {
        s.push_str(&format!("  {name}: {stride},\n"));
    }
    s.push_str("} as const;\n\n");

    // The camera basis is the one piece of renderer math the TypeScript host
    // must reimplement, because orbit controls need it live. These fixtures are
    // Rust's answer for a spread of configurations; `camera.ts` checks itself
    // against them at startup, so a divergence surfaces immediately rather than
    // as a subtly mis-framed render.
    //
    // Each fixture carries its own camera rather than naming a scene: the two
    // are unrelated, and coupling them meant a scene dropping out of the browser
    // build reported itself as a camera bug.
    s.push_str(
        "/** Rust's camera resolution for a spread of configurations. `camera.ts`\n\
         \x20*  checks its own implementation against these at startup — this is the\n\
         \x20*  guard on the only renderer math that genuinely exists twice. */\n\
         export const CAMERA_FIXTURES: {\n\
         \x20 label: string;\n\
         \x20 camera: CameraDef;\n\
         \x20 width: number; height: number;\n\
         \x20 camOrigin: [number, number, number];\n\
         \x20 camUpperLeft: [number, number, number];\n\
         \x20 camHorizontal: [number, number, number];\n\
         \x20 camVertical: [number, number, number];\n\
         }[] = [\n",
    );

    // Deliberately varied: square and wide and tall aspects, narrow and wide
    // fields of view, off-axis and near-vertical view directions, and a
    // non-zero aperture. A fixture set that shares one camera would pass with a
    // transposed basis.
    let cameras: [(&str, Camera, u32, u32); 7] = [
        ("cornell", scenes::cornell_box().camera, 512, 512),
        ("cornell-wide", scenes::cornell_box().camera, 800, 450),
        ("cornell-tall", scenes::cornell_box().camera, 256, 384),
        (
            "off-axis",
            Camera::look_at(
                Vec3::new(-120.0, 340.0, -410.0),
                Vec3::new(230.0, 90.0, 180.0),
                55.0,
            ),
            640,
            480,
        ),
        (
            "near-vertical",
            Camera::look_at(
                Vec3::new(278.0, 900.0, 277.5),
                Vec3::new(278.0, 0.0, 278.0),
                45.0,
            ),
            512,
            512,
        ),
        (
            "narrow-fov",
            Camera::look_at(
                Vec3::new(0.0, 0.0, -1500.0),
                Vec3::new(278.0, 278.0, 278.0),
                12.0,
            ),
            1024,
            256,
        ),
        (
            "thin-lens",
            {
                let mut c = scenes::cornell_box().camera;
                c.aperture = 40.0;
                c.focus_distance = 950.0;
                c
            },
            400,
            400,
        ),
    ];

    for (label, camera, w, h) in cameras {
        let mut u = pt_core::gpu_layout::GpuUniforms {
            width: w,
            height: h,
            ..Default::default()
        };
        camera.write_uniforms(&mut u, w as f32 / h as f32);
        s.push_str(&format!(
            "  {{ label: {label:?}, width: {w}, height: {h}, camera: {{ eye: [{}, {}, {}], lookAt: [{}, {}, {}], up: [{}, {}, {}], vfovDeg: {}, aperture: {}, focusDistance: {} }}, camOrigin: [{:?}, {:?}, {:?}], camUpperLeft: [{:?}, {:?}, {:?}], camHorizontal: [{:?}, {:?}, {:?}], camVertical: [{:?}, {:?}, {:?}] }},\n",
            camera.eye.x, camera.eye.y, camera.eye.z,
            camera.look_at.x, camera.look_at.y, camera.look_at.z,
            camera.up.x, camera.up.y, camera.up.z,
            camera.vfov_deg, camera.aperture, camera.focus_distance,
            u.cam_origin[0], u.cam_origin[1], u.cam_origin[2],
            u.cam_upper_left[0], u.cam_upper_left[1], u.cam_upper_left[2],
            u.cam_horizontal[0], u.cam_horizontal[1], u.cam_horizontal[2],
            u.cam_vertical[0], u.cam_vertical[1], u.cam_vertical[2],
        ));
    }
    s.push_str("];\n");
    Ok(s)
}
