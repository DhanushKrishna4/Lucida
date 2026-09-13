//! **Single source of truth for GPU buffer layout.**
//!
//! The hybrid architecture (TypeScript host, Rust core, WGSL kernels) means
//! three languages have an opinion about how bytes are arranged in a storage
//! buffer. Silent disagreement between them is the nastiest class of bug in the
//! project: the image renders, it just renders *wrong*, and the cause is a
//! 4-byte offset you cannot see.
//!
//! So the Rust `#[repr(C)]` types below are authoritative, and
//! `cargo run -p pt-cli --bin codegen` emits
//!
//!   * `shaders/common/generated.wgsl` — matching WGSL `struct` declarations
//!   * `web/src/generated/*.ts`        — matching TypeScript offsets and blobs
//!
//! Both generated files are checked in and CI re-runs codegen to assert they are
//! up to date, so drift becomes a build failure rather than a rendering artifact.
//!
//! # WGSL layout rules you must respect when editing these
//!
//! For the `storage` address space WGSL uses (essentially) std430:
//!
//! | type        | align | size |
//! |-------------|-------|------|
//! | `f32`,`u32` | 4     | 4    |
//! | `vec2<f32>` | 8     | 8    |
//! | `vec3<f32>` | **16**| 12   |
//! | `vec4<f32>` | 16    | 16   |
//!
//! The trap is `vec3`: align 16 but size 12, so a `vec3` followed by another
//! `vec3` leaves a 4-byte hole that Rust's `#[repr(C)]` will *not* insert for
//! `[f32; 3]`. Every `vec3` here is therefore followed by an explicit padding
//! field, and the tests at the bottom assert every offset.
//!
//! `uniform` address space is stricter still (std140: array stride and struct
//! alignment round up to 16), which is why the uniform block is padded to a
//! multiple of 16 bytes.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

/// A principled-BSDF material. For build steps 1–3 only `base_color` and
/// `emissive` are consumed (Lambertian + emitters); the remaining parameters are
/// laid out now so the buffer layout does not churn when GGX lands at step 6.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub struct GpuMaterial {
    /// Diffuse/base reflectance, linear RGB in [0, 1].
    pub base_color: [f32; 3], // 0
    pub metallic: f32, // 12
    /// Emitted radiance in W/(m^2 sr). Not clamped to [0,1] — it is radiance.
    pub emissive: [f32; 3], // 16
    pub roughness: f32, // 28
    pub ior: f32,      // 32
    pub transmission: f32, // 36
    // Padding is written as scalars, never as `vec2`/`vec3`: WGSL gives those
    // alignments of 8 and 16, which would silently relocate the field (and
    // change the struct size) relative to Rust's `#[repr(C)]`.
    pub _pad0: f32, // 40
    pub _pad1: f32, // 44

    /// Real part of the complex index of refraction, per RGB channel.
    ///
    /// Only consulted when `k` is non-zero, which is the flag for "this is a
    /// conductor, use the exact Fresnel equations". Metals do not follow the
    /// dielectric curve Schlick approximates: copper and gold *dip* in the
    /// middle of their angular response before climbing to white at grazing,
    /// and Schlick can only interpolate monotonically from F0 to 1. See
    /// [`crate::bsdf::fresnel_conductor`].
    pub eta: [f32; 3], // 48
    pub _pad2: f32, // 60
    /// Imaginary part, the extinction coefficient. Zero means dielectric.
    pub k: [f32; 3], // 64
    pub _pad3: f32, // 76, size 80
}

impl GpuMaterial {
    pub fn diffuse(base_color: Vec3) -> Self {
        Self {
            base_color: base_color.to_array(),
            roughness: 1.0,
            ior: 1.5,
            ..Default::default()
        }
    }

    /// A transmissive dielectric: glass, water, a gemstone.
    ///
    /// `tint` colours light that passes through, not light that reflects off —
    /// the specular highlight on glass is white whatever colour the glass is,
    /// because it never entered the medium. `ior` is the absolute index: 1.33
    /// water, 1.5 window glass, 2.42 diamond.
    ///
    /// Roughness is floored by `MIN_ALPHA` rather than special-cased, so 0 means
    /// "as sharp as this renderer represents" and still goes through the ordinary
    /// sample/pdf machinery. See [`crate::bsdf::dielectric`].
    pub fn glass(tint: Vec3, roughness: f32, ior: f32) -> Self {
        Self {
            base_color: tint.to_array(),
            roughness,
            ior,
            transmission: 1.0,
            ..Default::default()
        }
    }

    /// A dielectric with a GGX specular lobe over a diffuse base — plastic,
    /// paint, most non-metals.
    pub fn glossy(base_color: Vec3, roughness: f32) -> Self {
        Self {
            base_color: base_color.to_array(),
            roughness,
            ior: 1.5,
            ..Default::default()
        }
    }

    /// A metal using the artist-facing parameterisation: `base_color` is the
    /// normal-incidence reflectance, with Schlick's curve.
    pub fn metal(base_color: Vec3, roughness: f32) -> Self {
        Self {
            base_color: base_color.to_array(),
            metallic: 1.0,
            roughness,
            ior: 1.5,
            ..Default::default()
        }
    }

    /// A metal from measured optical constants, using the exact conductor
    /// Fresnel equations. See [`crate::bsdf::conductors`].
    pub fn conductor(eta: Vec3, k: Vec3, roughness: f32) -> Self {
        Self {
            // Unused on this path — Fresnel comes entirely from eta and k — but
            // set to the normal-incidence reflectance so anything that reads
            // base_color for a preview gets a sensible colour.
            base_color: crate::bsdf::fresnel_conductor(1.0, eta, k).to_array(),
            metallic: 1.0,
            roughness,
            ior: 1.5,
            eta: eta.to_array(),
            k: k.to_array(),
            ..Default::default()
        }
    }

    /// True when this material should use the exact conductor Fresnel.
    #[inline]
    pub fn is_conductor(&self) -> bool {
        self.k[0] > 0.0 || self.k[1] > 0.0 || self.k[2] > 0.0
    }

    pub fn emissive(radiance: Vec3) -> Self {
        Self {
            // Black base colour: an emitter that also reflects would need a
            // second lobe evaluated here, and at this stage we want the light to
            // be a pure source so that "hit the light" terminates the path with
            // zero further contribution and no special-casing.
            base_color: [0.0; 3],
            emissive: radiance.to_array(),
            roughness: 1.0,
            // IOR 1.0 gives F0 = 0, so the emitter has no specular lobe either.
            // Without this a "black" light source would still reflect the
            // dielectric 4%, and the path would not terminate on hitting it.
            // A real fixture does reflect; a reference emitter should not.
            ior: 1.0,
            ..Default::default()
        }
    }
}

/// [`GpuPrimitive::kind`]: an analytic sphere.
pub const PRIM_KIND_SPHERE: u32 = 0;
/// [`GpuPrimitive::kind`]: a parallelogram.
pub const PRIM_KIND_QUAD: u32 = 1;

/// An analytic primitive: a sphere or a parallelogram, in one tagged struct.
///
/// # Why one struct instead of two buffers
///
/// WebGPU guarantees only **eight** storage buffers per shader stage, and this
/// renderer needs every one of them: materials, geometry, the BVH, the
/// accumulator, and the light list. Separate sphere and quad buffers would need
/// nine.
///
/// Merging costs 32 wasted bytes per sphere, which at the scale analytic
/// primitives exist at — dozens, not millions — is nothing. It also collapses
/// two intersection loops into one, and makes an emissive sphere expressible
/// later without another binding.
///
/// Fields are shared between the two kinds; the documentation on each says which
/// kind uses it. That is less self-describing than two structs, and it is the
/// price of the binding budget.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub struct GpuPrimitive {
    /// Sphere: centre. Quad: the corner the edges emanate from.
    pub position: [f32; 3], // 0
    /// Sphere: radius. Unused by quads.
    pub radius: f32, // 12
    /// Quad: first edge vector. Unused by spheres.
    pub edge_u: [f32; 3], // 16
    /// `PRIM_KIND_SPHERE` or `PRIM_KIND_QUAD`.
    pub kind: u32, // 28
    /// Quad: second edge vector. Unused by spheres.
    pub edge_v: [f32; 3], // 32
    pub material: u32, // 44
    /// Quad: unit normal, precomputed. Unused by spheres, whose normal is
    /// exact and trivially recovered from the hit point.
    pub normal: [f32; 3], // 48
    pub _pad0: f32,    // 60, size 64
}

impl GpuPrimitive {
    pub fn sphere(center: Vec3, radius: f32, material: u32) -> Self {
        Self {
            position: center.to_array(),
            radius,
            kind: PRIM_KIND_SPHERE,
            material,
            ..Default::default()
        }
    }

    /// A parallelogram spanned by `edge_u` and `edge_v` from `origin`.
    ///
    /// The normal is `normalize(cross(edge_u, edge_v))`, so **edge order decides
    /// which way it faces**. Emission is one-sided and back-face culling of
    /// light is real behaviour, so the order matters.
    pub fn quad(origin: Vec3, edge_u: Vec3, edge_v: Vec3, material: u32) -> Self {
        Self {
            position: origin.to_array(),
            edge_u: edge_u.to_array(),
            kind: PRIM_KIND_QUAD,
            edge_v: edge_v.to_array(),
            material,
            normal: edge_u.cross(edge_v).normalize().to_array(),
            ..Default::default()
        }
    }

    #[inline]
    pub fn is_quad(&self) -> bool {
        self.kind == PRIM_KIND_QUAD
    }
}

/// Per-vertex shading attributes.
///
/// Kept *separate* from positions rather than interleaved into one vertex
/// struct. Positions are read constantly — every leaf test in every traversal
/// step touches three of them — while normals and UVs are read exactly once, at
/// the closest hit. Interleaving would drag 32 bytes of cold data through the
/// cache on every triangle test.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub struct GpuVertexAttr {
    pub normal: [f32; 3], // 0
    pub _pad0: f32,       // 12
    pub uv: [f32; 2],     // 16
    pub _pad1: [f32; 2],  // 24, size 32
}

/// A triangle as three indices into the position/attribute arrays.
///
/// Indexed rather than storing three positions inline: a closed mesh shares each
/// vertex between about six triangles, so indexing costs 16 bytes per triangle
/// instead of 48 and keeps far more of the mesh resident in cache.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub struct GpuTriangle {
    pub i0: u32,       // 0
    pub i1: u32,       // 4
    pub i2: u32,       // 8
    pub material: u32, // 12, size 16
}

/// A binary BVH node, 32 bytes.
///
/// # Why this layout
///
/// 32 bytes is half a 64-byte cache line, so a traversal step that reads a node
/// and then its two children touches at most two lines. Growing the node past 32
/// bytes is the single easiest way to lose a large fraction of traversal
/// performance, which is why the child pointer and the primitive range share a
/// field:
///
/// * `count == 0` marks an **internal** node, and `left_first` is the index of
///   its left child. The right child is always `left_first + 1` — children are
///   allocated as an adjacent pair — so one index addresses both.
/// * `count > 0` marks a **leaf**, and `left_first` is the offset into the
///   primitive-index array where its `count` primitives begin.
///
/// The AABB is stored as explicit min/max rather than centre/extent: traversal
/// does a slab test, which wants min and max directly, and reconstructing them
/// from centre/extent would cost two extra operations per node per ray.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub struct GpuBvhNode {
    pub bounds_min: [f32; 3], // 0
    /// Left child index (internal) or first primitive offset (leaf).
    pub left_first: u32, // 12
    pub bounds_max: [f32; 3], // 16
    /// 0 = internal node; otherwise the number of primitives in this leaf.
    pub count: u32, // 28, size 32
}

impl Default for GpuBvhNode {
    fn default() -> Self {
        Self {
            // An inverted AABB is the identity for union, so a freshly created
            // node absorbs the first bounds merged into it.
            bounds_min: [f32::INFINITY; 3],
            bounds_max: [f32::NEG_INFINITY; 3],
            left_first: 0,
            count: 0,
        }
    }
}

/// One pixel of the accumulation buffer, with the denoiser's guide channels.
///
/// # Why the guides live here
///
/// A denoiser needs to know what it is looking at: two neighbouring pixels
/// should only be blended if they are the same surface, and "same surface" is
/// judged from the **albedo**, the **normal** and the **depth** at the first
/// hit. Those are noise-free — they come from a single deterministic
/// intersection, not from an integral — which is exactly what makes them usable
/// as a guide for filtering something that is noisy.
///
/// They are appended to the accumulation buffer rather than given a buffer of
/// their own because the megakernel already binds the eight storage buffers
/// WebGPU guarantees per stage. Widening one costs memory; adding one does not
/// fit at all.
///
/// # Why they are accumulated rather than written once
///
/// The camera ray is jittered per sample, so the first hit differs slightly
/// between samples. Averaging the guides over samples antialiases them, which
/// matters at silhouettes: a guide sampled from one arbitrary sub-pixel position
/// would make the denoiser treat a whole edge pixel as belonging to whichever
/// surface that one ray happened to strike.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub struct GpuAccum {
    /// Sum of radiance estimates.
    pub radiance: [f32; 3], // 0
    /// Samples accumulated so far. Divides all four channels.
    pub samples: f32, // 12
    /// Sum of first-hit surface albedo, for demodulation.
    ///
    /// Filtering `radiance / albedo` and multiplying back afterwards is what
    /// keeps texture detail out of the filter's way: the denoiser sees only the
    /// lighting, which is smooth, instead of lighting times a pattern that its
    /// edge-stopping weights would try to preserve and blur anyway.
    pub albedo: [f32; 3], // 16
    /// Sum of first-hit distances. A background ray contributes zero.
    pub depth: f32, // 28
    /// Sum of first-hit world-space normals. Not renormalised until use, so the
    /// average of a silhouette pixel's normals is short — which is itself a
    /// useful signal that the pixel straddles an edge.
    pub normal: [f32; 3], // 32
    /// Sum of BVH node visits on the first ray, for the traversal heatmap.
    ///
    /// The word the guide channels left spare. Counting it costs one increment
    /// in the traversal loop and nothing anywhere else, and it is the only
    /// diagnostic here that is not already being computed for some other reason.
    pub traversal: f32, // 44
    /// Sum of **squared** radiance estimates, for the convergence readout.
    ///
    /// With the sum and the sum of squares, the per-pixel variance of the
    /// estimator is `sum_sq/N - mean^2`, and the standard error of the mean is
    /// `sqrt(var/N)`. That is a *measured* statement about how converged the
    /// image is, as opposed to the `1/sqrt(N)` a model would predict — and this
    /// renderer has spent seventeen steps establishing that the two differ
    /// whenever the integrand is heavy-tailed, which is exactly when someone
    /// wants to know.
    ///
    /// Costs 16 bytes per pixel. The struct's alignment is 16 either way, so
    /// storing full RGB here is free against storing a single luminance.
    pub radiance_sq: [f32; 3], // 48
    pub _pad0: f32, // 60, size 64
}

/// Per-dispatch uniforms. Kept under 256 bytes so it comfortably fits the
/// minimum guaranteed `maxUniformBufferBindingSize`.
///
/// The camera is stored in "corner + spanning vectors" form rather than as a
/// matrix: generating a primary ray is then two multiply-adds and no division,
/// and — more importantly — it is trivially identical in Rust, WGSL and
/// TypeScript, with no convention questions about row/column major or
/// handedness.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub struct GpuUniforms {
    pub cam_origin: [f32; 3], // 0
    pub lens_radius: f32,     // 12  (0 => pinhole)
    /// World-space corner of the image plane corresponding to pixel (0, 0).
    ///
    /// That is the **top-left** corner, because `cam_vertical` points *down*.
    /// Encoding the image flip into the basis rather than into the pixel-to-uv
    /// formula means CPU, WGSL and TypeScript all share one unconditional
    /// expression with no `1.0 -` that somebody can forget:
    ///
    /// ```text
    ///   s = (x + jitter.x) / width
    ///   t = (y + jitter.y) / height
    ///   dir = normalize(upper_left + s*horizontal + t*vertical - origin)
    /// ```
    pub cam_upper_left: [f32; 3], // 16
    pub focus_distance: f32,  // 28
    /// Spans the full image width, pointing right.
    pub cam_horizontal: [f32; 3], // 32
    pub _pad1: f32,           // 44
    /// Spans the full image height, pointing **down** (row 0 is the top row).
    pub cam_vertical: [f32; 3], // 48
    pub _pad2: f32,           // 60

    pub width: u32,  // 64
    pub height: u32, // 68
    /// Index of the first sample this dispatch will take. Feeds the RNG seed, so
    /// consecutive dispatches draw fresh, non-overlapping streams.
    pub sample_offset: u32, // 72
    /// Samples per pixel taken by this single dispatch. Bounded to keep any one
    /// dispatch short enough to stay clear of the GPU watchdog.
    pub samples_per_launch: u32, // 76

    pub max_depth: u32, // 80
    /// Global seed; changing it reruns the whole render with a different stream,
    /// which is how the Monte Carlo noise floor is measured.
    pub frame_seed: u32, // 84
    /// Analytic primitives: spheres and quads share one array.
    pub num_primitives: u32, // 88
    pub num_lights: u32, // 92

    /// Radiance returned by rays that escape the scene, i.e. a constant
    /// environment. Black for a closed scene like the Cornell box.
    ///
    /// The white furnace test *is* "put the object in a uniform environment of
    /// radiance 1", so it needs exactly this field. A real importance-sampled
    /// HDR environment arrives at build step 13.
    pub background: [f32; 3], // 96
    pub num_triangles: u32, // 108

    /// Number of BVH nodes. Zero means "no BVH", and the shader falls back to
    /// testing every triangle — which is exactly what the traversal correctness
    /// test compares against.
    pub num_bvh_nodes: u32, // 112
    /// Diagnostic render mode; 0 is the beauty pass.
    pub render_mode: u32, // 116
    /// `SamplingMode::index()`: 0 = BSDF sampling only, 1 = next event
    /// estimation only.
    pub sampling_mode: u32, // 120
    /// Environment map dimensions. Zero width means "no map"; the shader then
    /// falls back to [`GpuUniforms::background`].
    pub env_width: u32, // 124
    pub env_height: u32, // 128
    /// Sum of `luminance * sin(theta)` over the map. The normalising constant
    /// for the environment pdf; summing it in a shader would mean walking the
    /// whole map per lookup.
    pub env_total_weight: f32, // 132
    /// `SamplerKind::index()`: 0 = independent PCG, 1 = Owen-scrambled Sobol.
    pub sampler_kind: u32, // 136
    /// Instances appended to `primitives`, starting at `num_primitives`. Zero
    /// means the renderer takes the single-level path.
    pub num_instances: u32, // 140
    /// First node of the TLAS within `bvh_nodes`.
    pub tlas_root: u32, // 144
    pub _pad3: u32, // 148
    pub _pad4: u32, // 152
    pub _pad5: u32, // 156, size 160
}

/// [`GpuLight::kind`]: a parallelogram, sampled over the unit square.
/// [`GpuPrimitive::kind`]: an instance of a bottom-level acceleration
/// structure. See [`GpuInstance`].
pub const PRIM_KIND_INSTANCE: u32 = 2;

/// One placement of a mesh, laid out to **overlay** [`GpuPrimitive`].
///
/// Identical size, and `kind` at the identical offset, so instances live in the
/// same buffer as spheres and quads and the shader tells them apart by the tag
/// it already reads.
///
/// That is not cleverness for its own sake — it is forced. The wavefront's
/// EXTEND stage already binds exactly eight storage buffers, WebGPU's guaranteed
/// per-stage limit counted across every bind group, so instancing had to cost
/// **zero** new bindings. A 64-byte tagged union with four `vec3` slots and four
/// spare scalars happens to be exactly a 3x4 inverse transform plus metadata.
///
/// The four `vec3`s hold the rows of the inverse 3x3 and the inverse
/// translation; the scalars in between hold what a sphere would use for its
/// radius, tag, material and padding.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub struct GpuInstance {
    /// Row 0 of the world-to-object 3x3.
    pub inv_row0: [f32; 3], // 0
    /// First node of this instance's BLAS within `bvh_nodes`.
    pub blas_root: u32, // 12
    pub inv_row1: [f32; 3], // 16
    /// Always [`PRIM_KIND_INSTANCE`]. Same offset as [`GpuPrimitive::kind`].
    pub kind: u32, // 28
    pub inv_row2: [f32; 3], // 32
    /// Material for every triangle of this instance, or `u32::MAX` to keep each
    /// triangle's own. Same offset as [`GpuPrimitive::material`].
    pub material: u32, // 44
    /// Translation column of the world-to-object matrix.
    pub inv_translation: [f32; 3], // 48
    pub _pad0: f32, // 60, size 64
}

impl GpuInstance {
    /// Pack a [`crate::instance::Instance`] for upload.
    pub fn new(inst: &crate::instance::Instance, blas_root: u32) -> Self {
        let m = inst.world_to_object;
        // glam is column-major; the rows are what a matrix-times-vector needs.
        let row = |i: usize| [m.col(0)[i], m.col(1)[i], m.col(2)[i]];
        Self {
            inv_row0: row(0),
            blas_root,
            inv_row1: row(1),
            kind: PRIM_KIND_INSTANCE,
            inv_row2: row(2),
            material: inst.material_override,
            inv_translation: [m.col(3)[0], m.col(3)[1], m.col(3)[2]],
            _pad0: 0.0,
        }
    }

    /// Reinterpret as a primitive, for storage in the shared array.
    ///
    /// A transmute in all but name, and sound because both are `Pod` and the
    /// layout assertions below pin every offset.
    pub fn as_primitive(&self) -> GpuPrimitive {
        bytemuck::cast(*self)
    }
}

pub const LIGHT_KIND_QUAD: u32 = 0;
/// [`GpuLight::kind`]: a triangle, sampled with the square-root warp.
pub const LIGHT_KIND_TRIANGLE: u32 = 1;

/// One area light: an emissive primitive flattened into a form that can be
/// sampled directly, without going back to the geometry buffers.
///
/// A quad and a triangle differ only in how the unit square maps onto them, so
/// one struct covers both and the shader branches once on `kind` rather than
/// keeping two parallel light lists.
///
/// `area` is precomputed because the area-measure pdf needs it on every shadow
/// ray, and recovering it from the edges costs a cross product and a square root
/// each time.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub struct GpuLight {
    /// Corner the edges emanate from.
    pub origin: [f32; 3], // 0
    pub area: f32,        // 12
    pub edge_u: [f32; 3], // 16
    /// `LIGHT_KIND_QUAD` or `LIGHT_KIND_TRIANGLE`.
    pub kind: u32, // 28
    pub edge_v: [f32; 3], // 32
    pub material: u32,    // 44
    /// Outward normal. Emission is one-sided, from this face.
    pub normal: [f32; 3], // 48
    pub _pad0: f32,       // 60, size 64
}

// ---------------------------------------------------------------------------
// Wavefront path tracing
// ---------------------------------------------------------------------------

/// Everything a path in flight needs to carry between wavefront stages.
///
/// In the megakernel all of this lives in registers for the lifetime of the
/// path. Splitting the tracer into stages means it has to be written to and read
/// from global memory at every bounce instead — that is the architecture's main
/// cost, and it is why the struct is kept as small as it is. 80 bytes is five
/// 16-byte rows; every field earns its place.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub struct GpuPathState {
    pub origin: [f32; 3], // 0
    /// Which pixel this path contributes to. Paths are compacted between
    /// bounces, so a path's position in the queue says nothing about where it
    /// belongs on screen.
    pub pixel: u32, // 12
    pub direction: [f32; 3], // 16
    /// The RNG is carried explicitly rather than re-seeded per stage: the draw
    /// order across a path has to match the megakernel's exactly, and a stage
    /// that re-seeded would restart the sequence every bounce.
    pub rng_state: u32, // 28
    pub throughput: [f32; 3], // 32
    /// Density of the BSDF sample that produced this ray, for the MIS weight if
    /// it lands on an emitter.
    pub prev_bsdf_pdf: f32, // 44
    pub radiance: [f32; 3], // 48
    pub depth: u32,       // 60
    /// Vertex this ray started from, the other half of the MIS weight.
    pub prev_position: [f32; 3], // 64
    /// Next Sobol dimension this path will draw.
    ///
    /// Only `dim` needs carrying: the sample index and the per-pixel scramble
    /// seed are both derivable from `pixel` and the uniforms, but the dimension
    /// counter advances as the path goes and a stage that reset it would restart
    /// the sequence every bounce — the same reason `rng_state` is carried.
    pub sampler_dim: u32, // 76
    /// First-hit guides for the denoiser, written by SHADE at depth 0.
    ///
    /// Carried in the path state because the wavefront's SHADE stage cannot
    /// reach the accumulation buffer — it already binds the eight storage
    /// buffers WebGPU guarantees — so the values have to travel to RESOLVE,
    /// which can. Costs 32 bytes per in-flight path.
    pub guide_albedo: [f32; 3], // 80
    pub guide_depth: f32, // 92
    pub guide_normal: [f32; 3], // 96
    pub _pad1: f32, // 108, size 112
}

/// Result of the EXTEND stage, consumed by SHADE.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub struct GpuHitRecord {
    pub position: [f32; 3],         // 0
    pub t: f32,                     // 12
    pub normal: [f32; 3],           // 16
    pub material: u32,              // 28
    pub geometric_normal: [f32; 3], // 32
    pub light_area: f32,            // 44
    pub front_face: u32,            // 48
    pub valid: u32,                 // 52
    pub _pad0: u32,                 // 56
    pub _pad1: u32,                 // 60, size 64
}

/// A deferred shadow ray, produced by SHADE and resolved by CONNECT.
///
/// The contribution is computed up front and carried here, so CONNECT only has
/// to answer a visibility question — it never touches a BSDF or a material.
/// That is what keeps the occlusion kernel small and its register use low.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub struct GpuShadowRay {
    pub origin: [f32; 3],    // 0
    pub distance: f32,       // 12
    pub direction: [f32; 3], // 16
    /// Index of the path that cast this ray, **not** the pixel.
    ///
    /// CONNECT adds the contribution back into that path's radiance rather than
    /// straight into the accumulator. Writing to the accumulator was correct
    /// only while exactly one path per pixel was ever in flight; once a batch
    /// carries several samples of the same pixel at once, two shadow rays in one
    /// bounce can land on the same accumulator element and race. Path indices
    /// stay unique — one shadow ray per path per bounce — so the write needs no
    /// atomics either way.
    pub path: u32, // 28
    /// Radiance to add if the ray is unoccluded, already weighted and divided
    /// by the sampling density.
    pub contribution: [f32; 3], // 32
    pub _pad0: f32,          // 44, size 48
}

/// Indirect dispatch arguments. **A separate buffer from the counters**, and it
/// has to be.
///
/// WebGPU forbids a buffer being both a read-write storage binding and the
/// source of indirect arguments within the same dispatch. SHADE writes counters
/// while being dispatched indirectly, so the two cannot share a buffer — the
/// first attempt here put them together and every dispatch failed validation.
///
/// Written only by RESET, which dispatches directly and so may bind this
/// writable.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub struct GpuDispatchArgs {
    /// Workgroups for EXTEND and SHADE. Offset 0.
    pub trace: [u32; 3],
    /// Workgroups for CONNECT. Offset 12.
    pub shadow: [u32; 3],
    pub _pad0: [u32; 2], // 24, size 32
}

/// Queue lengths, all GPU-maintained.
///
/// Kept off the host entirely: a readback between stages would stall the
/// pipeline for longer than the whole architecture saves.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub struct GpuWavefrontCounters {
    /// Paths appended for the next bounce.
    pub next_queue_len: u32, // 0
    /// Workgroup count for the next bounce, accumulated with `atomicMax` as
    /// SHADE appends, so no pass is needed to turn a count into a dispatch size.
    pub next_trace_x: u32, // 4
    /// Shadow rays appended this bounce.
    pub shadow_len: u32, // 8
    /// Snapshot of `shadow_len` taken by RESET, for CONNECT to bound-check
    /// against. Needed because RESET clears the atomic for the next bounce
    /// before CONNECT runs.
    pub shadow_count: u32, // 12, size 16
}

/// Everything the GPU needs to describe a scene, already packed.
#[derive(Clone, Debug, Default)]
pub struct SceneBlob {
    pub materials: Vec<GpuMaterial>,
    /// Analytic primitives: spheres and parallelograms, in one tagged array.
    pub primitives: Vec<GpuPrimitive>,

    /// Vertex positions, `w` unused. `vec4` rather than a padded `vec3` struct
    /// so the shader can index it as a plain `array<vec4<f32>>`.
    pub positions: Vec<[f32; 4]>,
    pub vertex_attrs: Vec<GpuVertexAttr>,
    pub triangles: Vec<GpuTriangle>,

    /// BVH over `triangles`, built by [`crate::bvh`]. Empty when there are no
    /// triangles.
    pub bvh_nodes: Vec<GpuBvhNode>,
    /// Permutation of triangle indices; BVH leaves address this, not
    /// `triangles` directly, so the build never has to move triangle data.
    pub bvh_prim_indices: Vec<u32>,

    /// Instances, appended to `primitives` at upload. Empty for a scene with no
    /// instancing, in which case the renderer takes the single-level path.
    pub instances: Vec<GpuInstance>,

    /// Every emissive primitive, flattened for direct sampling. Built by
    /// [`crate::light::build_lights`].
    pub lights: Vec<GpuLight>,
}

impl SceneBlob {
    pub fn materials_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.materials)
    }
    /// Analytic primitives followed by the instances.
    ///
    /// One buffer, because [`GpuInstance`] overlays [`GpuPrimitive`] exactly and
    /// the shader discriminates on the `kind` tag both share. Forced by the
    /// binding budget — see [`GpuInstance`] — and it means the analytic loop
    /// stops at `num_primitives` while the TLAS's leaves index past it.
    pub fn primitives_upload(&self) -> Vec<GpuPrimitive> {
        let mut out = self.primitives.clone();
        out.extend(self.instances.iter().map(|i| i.as_primitive()));
        out
    }

    pub fn primitives_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.primitives)
    }
    pub fn positions_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.positions)
    }
    pub fn vertex_attrs_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.vertex_attrs)
    }
    pub fn triangles_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.triangles)
    }
    pub fn bvh_nodes_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.bvh_nodes)
    }
    pub fn bvh_prim_indices_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.bvh_prim_indices)
    }
    pub fn lights_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.lights)
    }
}

// ---------------------------------------------------------------------------
// WGSL declarations, emitted by the codegen binary.
// ---------------------------------------------------------------------------

/// The WGSL mirror of the types above. Kept adjacent to them on purpose: a
/// reviewer editing one has the other on screen. The offset tests below are what
/// actually enforce the correspondence.
pub const WGSL_STRUCTS: &str = r#"// @generated by `cargo run -p pt-cli --bin codegen` — do not edit.
// Mirrors crates/core/src/gpu_layout.rs. See that file for the layout rules.

struct Material {
  base_color   : vec3<f32>,   // offset 0
  metallic     : f32,         // offset 12
  emissive     : vec3<f32>,   // offset 16
  roughness    : f32,         // offset 28
  ior          : f32,         // offset 32
  transmission : f32,         // offset 36
  _pad0        : f32,         // offset 40
  _pad1        : f32,         // offset 44
  // Complex IOR, consulted only when k is non-zero (the conductor flag).
  eta          : vec3<f32>,   // offset 48
  _pad2        : f32,         // offset 60
  k            : vec3<f32>,   // offset 64
  _pad3        : f32,         // offset 76, size 80
};

// A sphere or a parallelogram in one tagged struct. WebGPU guarantees only
// eight storage buffers per stage and this renderer needs all eight, so the two
// analytic primitive types share one buffer. See crates/core/src/gpu_layout.rs.
struct Primitive {
  position : vec3<f32>,       // offset 0   sphere centre / quad origin
  radius   : f32,             // offset 12  sphere only
  edge_u   : vec3<f32>,       // offset 16  quad only
  kind     : u32,             // offset 28
  edge_v   : vec3<f32>,       // offset 32  quad only
  material : u32,             // offset 44
  normal   : vec3<f32>,       // offset 48  quad only
  _pad0    : f32,             // offset 60, size 64
};

const PRIM_KIND_SPHERE: u32 = 0u;
const PRIM_KIND_QUAD: u32 = 1u;

// 48 bytes: radiance and sample count, then the denoiser's guide channels.
// Appended here rather than given their own buffer because the megakernel
// already binds the eight storage buffers WebGPU guarantees per stage.
struct Accum {
  radiance : vec3<f32>,        // offset 0
  samples  : f32,              // offset 12
  albedo   : vec3<f32>,        // offset 16
  depth    : f32,              // offset 28
  normal   : vec3<f32>,        // offset 32
  traversal: f32,              // offset 44
  radiance_sq: vec3<f32>,      // offset 48
  _pad0    : f32,              // offset 60, size 64
};

struct Uniforms {
  cam_origin         : vec3<f32>,  // offset 0
  lens_radius        : f32,        // offset 12
  cam_upper_left     : vec3<f32>,  // offset 16
  focus_distance     : f32,        // offset 28
  cam_horizontal     : vec3<f32>,  // offset 32
  _pad1              : f32,        // offset 44
  cam_vertical       : vec3<f32>,  // offset 48
  _pad2              : f32,        // offset 60
  width              : u32,        // offset 64
  height             : u32,        // offset 68
  sample_offset      : u32,        // offset 72
  samples_per_launch : u32,        // offset 76
  max_depth          : u32,        // offset 80
  frame_seed         : u32,        // offset 84
  num_primitives     : u32,        // offset 88
  num_lights         : u32,        // offset 92
  background         : vec3<f32>,  // offset 96
  num_triangles      : u32,        // offset 108
  num_bvh_nodes      : u32,        // offset 112
  render_mode        : u32,        // offset 116
  sampling_mode      : u32,        // offset 120
  env_width          : u32,        // offset 124
  env_height         : u32,        // offset 128
  env_total_weight   : f32,        // offset 132
  sampler_kind       : u32,        // offset 136
  num_instances      : u32,        // offset 140
  tlas_root          : u32,        // offset 144
  _pad3              : u32,        // offset 148
  _pad4              : u32,        // offset 152
  _pad5              : u32,        // offset 156, size 160
};

// One area light, flattened for direct sampling. A quad and a triangle differ
// only in how the unit square maps onto them, so one struct covers both.
struct Light {
  origin   : vec3<f32>,       // offset 0
  area     : f32,             // offset 12
  edge_u   : vec3<f32>,       // offset 16
  kind     : u32,             // offset 28
  edge_v   : vec3<f32>,       // offset 32
  material : u32,             // offset 44
  normal   : vec3<f32>,       // offset 48
  _pad0    : f32,             // offset 60, size 64
};

const LIGHT_KIND_QUAD: u32 = 0u;
const LIGHT_KIND_TRIANGLE: u32 = 1u;

// --- Wavefront path tracing -------------------------------------------------
//
// In the megakernel all of this lives in registers for a path's lifetime.
// Splitting the tracer into stages means writing it to and reading it from
// global memory at every bounce instead — the architecture's main cost, and why
// PathState is kept to 80 bytes.

struct PathState {
  origin        : vec3<f32>,  // offset 0
  pixel         : u32,        // offset 12
  direction     : vec3<f32>,  // offset 16
  rng_state     : u32,        // offset 28
  throughput    : vec3<f32>,  // offset 32
  prev_bsdf_pdf : f32,        // offset 44
  radiance      : vec3<f32>,  // offset 48
  depth         : u32,        // offset 60
  prev_position : vec3<f32>,  // offset 64
  sampler_dim   : u32,        // offset 76
  guide_albedo  : vec3<f32>,  // offset 80
  guide_depth   : f32,        // offset 92
  guide_normal  : vec3<f32>,  // offset 96
  _pad1         : f32,        // offset 108, size 112
};

struct HitRecord {
  position         : vec3<f32>, // offset 0
  t                : f32,       // offset 12
  normal           : vec3<f32>, // offset 16
  material         : u32,       // offset 28
  geometric_normal : vec3<f32>, // offset 32
  light_area       : f32,       // offset 44
  front_face       : u32,       // offset 48
  valid            : u32,       // offset 52
  _pad0            : u32,       // offset 56
  _pad1            : u32,       // offset 60, size 64
};

// The contribution is computed by SHADE and carried here, so CONNECT only
// answers a visibility question — it never touches a BSDF or a material, which
// is what keeps the occlusion kernel small.
struct ShadowRay {
  origin       : vec3<f32>,  // offset 0
  distance     : f32,        // offset 12
  direction    : vec3<f32>,  // offset 16
  path         : u32,        // offset 28
  contribution : vec3<f32>,  // offset 32
  _pad0        : f32,        // offset 44, size 48
};

// Queue lengths and indirect dispatch arguments in one buffer. The dispatch
// sizes stay on the GPU: a readback between stages would stall for longer than
// the architecture saves. The *_x fields are workgroup counts maintained by
// atomicMax as items are appended, so no pass is needed to turn a count into a
// dispatch size.
// A separate buffer from the counters, and it has to be: WebGPU forbids a
// buffer being both a read-write storage binding and the source of indirect
// arguments within one dispatch, and SHADE writes counters while itself being
// dispatched indirectly.
// Scalar fields, not two vec3: a vec3<u32> has alignment 16, so `shadow` would
// land at offset 16 and the struct would be 48 bytes, while Rust's two [u32; 3]
// put it at 12 in a 32-byte struct. Indirect arguments must be exactly where the
// host says they are.
struct DispatchArgs {
  trace_x  : u32,  // offset 0   workgroups for EXTEND / SHADE
  trace_y  : u32,  // offset 4
  trace_z  : u32,  // offset 8
  shadow_x : u32,  // offset 12  workgroups for CONNECT
  shadow_y : u32,  // offset 16
  shadow_z : u32,  // offset 20
  _pad0    : u32,  // offset 24
  _pad1    : u32,  // offset 28, size 32
};

struct WavefrontCounters {
  next_queue_len : atomic<u32>,  // offset 0
  next_trace_x   : atomic<u32>,  // offset 4
  shadow_len     : atomic<u32>,  // offset 8
  // Snapshot taken by RESET, because it clears the atomic before CONNECT runs.
  shadow_count   : u32,          // offset 12, size 16
};

struct VertexAttr {
  normal : vec3<f32>,         // offset 0
  _pad0  : f32,               // offset 12
  uv     : vec2<f32>,         // offset 16
  _pad1  : vec2<f32>,         // offset 24, size 32
};

struct Triangle {
  i0       : u32,             // offset 0
  i1       : u32,             // offset 4
  i2       : u32,             // offset 8
  material : u32,             // offset 12, size 16
};

// 32 bytes: half a cache line, so a node and its two children span at most two.
// count == 0 means internal, and left_first is the LEFT child index (the right
// child is always left_first + 1). count > 0 means leaf, and left_first is the
// offset into the primitive-index array.
struct BvhNode {
  bounds_min : vec3<f32>,     // offset 0
  left_first : u32,           // offset 12
  bounds_max : vec3<f32>,     // offset 16
  count      : u32,           // offset 28, size 32
};
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    /// These assertions are the contract. If one fails, the WGSL and TypeScript
    /// mirrors are wrong too — regenerate them and re-read the layout rules at
    /// the top of this file.
    #[test]
    fn material_layout() {
        assert_eq!(size_of::<GpuMaterial>(), 80);
        assert_eq!(align_of::<GpuMaterial>(), 4);
        assert_eq!(offset_of!(GpuMaterial, base_color), 0);
        assert_eq!(offset_of!(GpuMaterial, metallic), 12);
        assert_eq!(offset_of!(GpuMaterial, emissive), 16);
        assert_eq!(offset_of!(GpuMaterial, roughness), 28);
        assert_eq!(offset_of!(GpuMaterial, ior), 32);
        assert_eq!(offset_of!(GpuMaterial, transmission), 36);
        assert_eq!(offset_of!(GpuMaterial, eta), 48);
        assert_eq!(offset_of!(GpuMaterial, k), 64);
    }

    #[test]
    fn primitive_layout() {
        assert_eq!(size_of::<GpuPrimitive>(), 64);
        assert_eq!(offset_of!(GpuPrimitive, position), 0);
        assert_eq!(offset_of!(GpuPrimitive, radius), 12);
        assert_eq!(offset_of!(GpuPrimitive, edge_u), 16);
        assert_eq!(offset_of!(GpuPrimitive, kind), 28);
        assert_eq!(offset_of!(GpuPrimitive, edge_v), 32);
        assert_eq!(offset_of!(GpuPrimitive, material), 44);
        assert_eq!(offset_of!(GpuPrimitive, normal), 48);
        // A light has the same shape as a primitive, deliberately: they are the
        // same thing viewed two ways, and keeping the layouts identical means a
        // future emissive sphere needs no new struct.
        assert_eq!(size_of::<GpuPrimitive>(), size_of::<GpuLight>());
    }

    /// [`GpuInstance`] must overlay [`GpuPrimitive`] exactly.
    ///
    /// The two share a buffer and the shader tells them apart by `kind`, so a
    /// drift in either size or the tag's offset would make the shader read a
    /// transform's bits as a sphere radius. It would not crash; it would produce
    /// geometry in the wrong place.
    #[test]
    fn instance_overlays_primitive() {
        assert_eq!(size_of::<GpuInstance>(), size_of::<GpuPrimitive>());
        assert_eq!(
            offset_of!(GpuInstance, kind),
            offset_of!(GpuPrimitive, kind),
            "the tag must sit at the same offset in both, or the union cannot be \
             discriminated"
        );
        assert_eq!(
            offset_of!(GpuInstance, material),
            offset_of!(GpuPrimitive, material)
        );
        assert_eq!(offset_of!(GpuInstance, inv_row0), 0);
        assert_eq!(offset_of!(GpuInstance, blas_root), 12);
        assert_eq!(offset_of!(GpuInstance, inv_row1), 16);
        assert_eq!(offset_of!(GpuInstance, kind), 28);
        assert_eq!(offset_of!(GpuInstance, inv_row2), 32);
        assert_eq!(offset_of!(GpuInstance, inv_translation), 48);

        // And the round trip through the shared array must be lossless.
        let inst = crate::instance::Instance::new(
            glam::Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0)),
            0,
            7,
        );
        let g = GpuInstance::new(&inst, 42);
        let back: GpuInstance = bytemuck::cast(g.as_primitive());
        assert_eq!(g, back);
        assert_eq!(g.as_primitive().kind, PRIM_KIND_INSTANCE);
        assert_eq!(g.as_primitive().material, 7);
    }

    #[test]
    fn accum_layout() {
        assert_eq!(size_of::<GpuAccum>(), 64);
        assert_eq!(offset_of!(GpuAccum, radiance_sq), 48);
        assert_eq!(offset_of!(GpuAccum, radiance), 0);
        assert_eq!(offset_of!(GpuAccum, samples), 12);
        assert_eq!(offset_of!(GpuAccum, albedo), 16);
        assert_eq!(offset_of!(GpuAccum, depth), 28);
        assert_eq!(offset_of!(GpuAccum, normal), 32);
        assert_eq!(offset_of!(GpuAccum, traversal), 44);
    }

    #[test]
    fn uniforms_layout() {
        assert_eq!(size_of::<GpuUniforms>(), 160);
        // std140 requires the uniform block size to be a multiple of 16.
        assert_eq!(size_of::<GpuUniforms>() % 16, 0);
        assert_eq!(offset_of!(GpuUniforms, cam_origin), 0);
        assert_eq!(offset_of!(GpuUniforms, lens_radius), 12);
        assert_eq!(offset_of!(GpuUniforms, cam_upper_left), 16);
        assert_eq!(offset_of!(GpuUniforms, focus_distance), 28);
        assert_eq!(offset_of!(GpuUniforms, cam_horizontal), 32);
        assert_eq!(offset_of!(GpuUniforms, cam_vertical), 48);
        assert_eq!(offset_of!(GpuUniforms, width), 64);
        assert_eq!(offset_of!(GpuUniforms, height), 68);
        assert_eq!(offset_of!(GpuUniforms, sample_offset), 72);
        assert_eq!(offset_of!(GpuUniforms, samples_per_launch), 76);
        assert_eq!(offset_of!(GpuUniforms, max_depth), 80);
        assert_eq!(offset_of!(GpuUniforms, frame_seed), 84);
        assert_eq!(offset_of!(GpuUniforms, num_primitives), 88);
        assert_eq!(offset_of!(GpuUniforms, num_lights), 92);
        assert_eq!(offset_of!(GpuUniforms, background), 96);
        assert_eq!(offset_of!(GpuUniforms, env_width), 124);
        assert_eq!(offset_of!(GpuUniforms, env_height), 128);
        assert_eq!(offset_of!(GpuUniforms, env_total_weight), 132);
        assert_eq!(offset_of!(GpuUniforms, sampler_kind), 136);
        assert_eq!(offset_of!(GpuUniforms, num_instances), 140);
        assert_eq!(offset_of!(GpuUniforms, tlas_root), 144);
        assert_eq!(offset_of!(GpuUniforms, num_triangles), 108);
        assert_eq!(offset_of!(GpuUniforms, num_bvh_nodes), 112);
        assert_eq!(offset_of!(GpuUniforms, render_mode), 116);
        assert_eq!(offset_of!(GpuUniforms, sampling_mode), 120);
    }

    #[test]
    fn wavefront_layout() {
        assert_eq!(size_of::<GpuPathState>(), 112);
        assert_eq!(offset_of!(GpuPathState, guide_albedo), 80);
        assert_eq!(offset_of!(GpuPathState, guide_depth), 92);
        assert_eq!(offset_of!(GpuPathState, guide_normal), 96);
        assert_eq!(offset_of!(GpuPathState, pixel), 12);
        assert_eq!(offset_of!(GpuPathState, direction), 16);
        assert_eq!(offset_of!(GpuPathState, rng_state), 28);
        assert_eq!(offset_of!(GpuPathState, throughput), 32);
        assert_eq!(offset_of!(GpuPathState, prev_bsdf_pdf), 44);
        assert_eq!(offset_of!(GpuPathState, radiance), 48);
        assert_eq!(offset_of!(GpuPathState, depth), 60);
        assert_eq!(offset_of!(GpuPathState, prev_position), 64);

        assert_eq!(size_of::<GpuHitRecord>(), 64);
        assert_eq!(offset_of!(GpuHitRecord, geometric_normal), 32);
        assert_eq!(offset_of!(GpuHitRecord, light_area), 44);
        assert_eq!(offset_of!(GpuHitRecord, valid), 52);

        assert_eq!(size_of::<GpuShadowRay>(), 48);
        assert_eq!(offset_of!(GpuShadowRay, path), 28);
        assert_eq!(offset_of!(GpuShadowRay, contribution), 32);

        // Indirect dispatch arguments must sit at a 4-byte-aligned offset, and
        // WebGPU reads three consecutive u32 from there.
        assert_eq!(offset_of!(GpuDispatchArgs, trace), 0);
        assert_eq!(offset_of!(GpuDispatchArgs, shadow), 12);
        assert_eq!(size_of::<GpuWavefrontCounters>(), 16);
        assert_eq!(offset_of!(GpuWavefrontCounters, shadow_count), 12);
    }

    #[test]
    fn light_layout() {
        assert_eq!(size_of::<GpuLight>(), 64);
        assert_eq!(offset_of!(GpuLight, origin), 0);
        assert_eq!(offset_of!(GpuLight, area), 12);
        assert_eq!(offset_of!(GpuLight, edge_u), 16);
        assert_eq!(offset_of!(GpuLight, kind), 28);
        assert_eq!(offset_of!(GpuLight, edge_v), 32);
        assert_eq!(offset_of!(GpuLight, material), 44);
        assert_eq!(offset_of!(GpuLight, normal), 48);
    }

    #[test]
    fn triangle_and_bvh_layout() {
        assert_eq!(size_of::<GpuVertexAttr>(), 32);
        assert_eq!(offset_of!(GpuVertexAttr, normal), 0);
        assert_eq!(offset_of!(GpuVertexAttr, uv), 16);

        assert_eq!(size_of::<GpuTriangle>(), 16);
        assert_eq!(offset_of!(GpuTriangle, material), 12);

        // The 32-byte node size is a performance contract, not an accident.
        // If this assertion starts failing, traversal just got slower.
        assert_eq!(size_of::<GpuBvhNode>(), 32);
        assert_eq!(offset_of!(GpuBvhNode, bounds_min), 0);
        assert_eq!(offset_of!(GpuBvhNode, left_first), 12);
        assert_eq!(offset_of!(GpuBvhNode, bounds_max), 16);
        assert_eq!(offset_of!(GpuBvhNode, count), 28);
    }

    /// Cross-check: the offsets written in the WGSL comments must match the real
    /// Rust offsets. Cheap, and it catches the case where somebody updates the
    /// struct but leaves the hand-written WGSL comment stale.
    #[test]
    fn wgsl_comment_offsets_match_rust() {
        let expected: &[(&str, usize)] = &[
            ("base_color", offset_of!(GpuMaterial, base_color)),
            ("metallic", offset_of!(GpuMaterial, metallic)),
            ("transmission", offset_of!(GpuMaterial, transmission)),
            ("radius", offset_of!(GpuPrimitive, radius)),
            ("edge_v", offset_of!(GpuPrimitive, edge_v)),
            (
                "samples_per_launch",
                offset_of!(GpuUniforms, samples_per_launch),
            ),
            ("num_primitives", offset_of!(GpuUniforms, num_primitives)),
            ("background", offset_of!(GpuUniforms, background)),
        ];
        for (field, off) in expected {
            let line = WGSL_STRUCTS
                .lines()
                .find(|l| l.trim_start().starts_with(&format!("{field} ")))
                .unwrap_or_else(|| panic!("field `{field}` missing from WGSL_STRUCTS"));
            assert!(
                line.contains(&format!("offset {off}")),
                "WGSL comment for `{field}` disagrees with Rust offset {off}: {line}"
            );
        }
    }
}
