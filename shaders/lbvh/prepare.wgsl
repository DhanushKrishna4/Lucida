// LBVH stage 1: primitive bounds, scene bounds, Morton codes.
//
// Three entry points sharing one binding set, because they form a chain over
// the same data and splitting them into files would mean declaring the same six
// buffers three times.
//
// The scene bounds are reduced on the GPU rather than passed in from the host.
// That is the point of the exercise — a builder that needs the CPU to look at
// every primitive first is not a GPU builder — and it costs one small kernel.

//!include "common/generated.wgsl"

struct BuildParams {
  count : u32,
  _pad0 : u32,
  _pad1 : u32,
  _pad2 : u32,
};

// 32 bytes, matching `Aabb` on the Rust side: vec3 has alignment 16 in WGSL, so
// the padding is not optional — a tightly packed 24-byte struct would be read
// at the wrong stride and every box after the first would be garbage.
struct AabbGpu {
  lo : vec3<f32>,
  _p0 : f32,
  hi : vec3<f32>,
  _p1 : f32,
};

@group(0) @binding(0) var<uniform> P: BuildParams;
@group(0) @binding(1) var<storage, read>       triangles   : array<Triangle>;
@group(0) @binding(2) var<storage, read>       positions   : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> tri_bounds  : array<AabbGpu>;
@group(0) @binding(4) var<storage, read_write> scene_box   : array<AabbGpu>;
@group(0) @binding(5) var<storage, read_write> codes       : array<u32>;
@group(0) @binding(6) var<storage, read_write> indices     : array<u32>;

const WG: u32 = 256u;

fn centroid_of(b: AabbGpu) -> vec3<f32> {
  return (b.lo + b.hi) * 0.5;
}

// ---------------------------------------------------------------------------
// One AABB per triangle.
// ---------------------------------------------------------------------------

@compute @workgroup_size(256, 1, 1)
fn triangle_bounds(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.count) {
    return;
  }
  let t = triangles[i];
  let p0 = positions[t.i0].xyz;
  let p1 = positions[t.i1].xyz;
  let p2 = positions[t.i2].xyz;
  var lo = min(p0, min(p1, p2));
  var hi = max(p0, max(p1, p2));

  // Pad outward by a magnitude-relative epsilon, exactly as `Aabb::pad` does.
  //
  // Not cosmetic, and not optional here. Every axis-aligned triangle produces a
  // zero-extent box, and a slab test against a flat box gives false negatives.
  // Matching the CPU's padding *bit for bit* additionally matters because the
  // Morton code is taken from the centroid of this box: padding is symmetric in
  // exact arithmetic but not in floats, so an unpadded GPU box and a padded CPU
  // box disagree on the centroid by an ULP and produce different codes for a
  // fraction of a percent of primitives. Measured: 192 of 368640 on
  // `bvh-stress` before this was added.
  let scale = vec3<f32>(1.0) + max(abs(lo), abs(hi));
  let p = 1.0e-6 * scale;
  lo = lo - p;
  hi = hi + p;

  var b: AabbGpu;
  b.lo = lo;
  b.hi = hi;
  b._p0 = 0.0;
  b._p1 = 0.0;
  tri_bounds[i] = b;
}

// ---------------------------------------------------------------------------
// Reduce every centroid into one box.
// ---------------------------------------------------------------------------
//
// A single workgroup walking the whole array in strided chunks, then a tree
// reduction in workgroup memory. One workgroup rather than a multi-level
// reduction with atomics for the same reason the radix scan is one workgroup:
// this runs once per build over a few hundred thousand elements, and the
// alternative needs an order-preserving float-to-u32 bijection to use
// atomicMin/atomicMax, which is a subtle thing to get right for a stage that
// does not show up in a profile.
//
// Strided rather than chunked so consecutive threads read consecutive elements,
// which is what makes the loads coalesce.

var<workgroup> red_lo: array<vec3<f32>, WG>;
var<workgroup> red_hi: array<vec3<f32>, WG>;

@compute @workgroup_size(256, 1, 1)
fn scene_bounds(@builtin(local_invocation_id) lid: vec3<u32>) {
  // An inverted box is the identity for union, so a thread that reads nothing
  // contributes nothing.
  var lo = vec3<f32>(1e30, 1e30, 1e30);
  var hi = vec3<f32>(-1e30, -1e30, -1e30);
  for (var i = lid.x; i < P.count; i = i + WG) {
    let c = centroid_of(tri_bounds[i]);
    lo = min(lo, c);
    hi = max(hi, c);
  }
  red_lo[lid.x] = lo;
  red_hi[lid.x] = hi;
  workgroupBarrier();

  for (var s = WG / 2u; s > 0u; s = s / 2u) {
    if (lid.x < s) {
      red_lo[lid.x] = min(red_lo[lid.x], red_lo[lid.x + s]);
      red_hi[lid.x] = max(red_hi[lid.x], red_hi[lid.x + s]);
    }
    workgroupBarrier();
  }

  if (lid.x == 0u) {
    var b: AabbGpu;
    b.lo = red_lo[0];
    b.hi = red_hi[0];
    b._p0 = 0.0;
    b._p1 = 0.0;
    scene_box[0] = b;
  }
}

// ---------------------------------------------------------------------------
// Morton codes. Mirrors `crates/core/src/lbvh.rs`, bit for bit.
// ---------------------------------------------------------------------------

const MORTON_SCALE: f32 = 1023.0;

// Spread the low 10 bits so each occupies every third bit. The shift-and-mask
// ladder rather than a loop: ten dependent iterations per primitive would be
// scalar work in the middle of an otherwise trivially parallel kernel.
fn expand_bits(v: u32) -> u32 {
  var x = v & 0x000003FFu;
  x = (x | (x << 16u)) & 0x030000FFu;
  x = (x | (x << 8u))  & 0x0300F00Fu;
  x = (x | (x << 4u))  & 0x030C30C3u;
  x = (x | (x << 2u))  & 0x09249249u;
  return x;
}

fn morton3d(p: vec3<f32>) -> u32 {
  // Clamped, not wrapped: a coordinate exactly on the upper bound would
  // otherwise quantise to 1024 and its top bit would collide with the next
  // axis, producing a code that sorts nowhere near its neighbours.
  let q = clamp(p * MORTON_SCALE, vec3<f32>(0.0), vec3<f32>(MORTON_SCALE));
  let x = expand_bits(u32(q.x));
  let y = expand_bits(u32(q.y));
  let z = expand_bits(u32(q.z));
  return (x << 2u) | (y << 1u) | z;
}

@compute @workgroup_size(256, 1, 1)
fn morton(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.count) {
    return;
  }
  let sb = scene_box[0];
  let extent = sb.hi - sb.lo;
  let c = centroid_of(tri_bounds[i]);
  // A zero-extent axis maps to 0.5 rather than dividing by zero. Flat scenes
  // are not exotic, and an axis with no extent carries no ordering information.
  let n = select(
    vec3<f32>(0.5, 0.5, 0.5),
    (c - sb.lo) / max(extent, vec3<f32>(1e-30)),
    extent > vec3<f32>(0.0),
  );
  codes[i] = morton3d(n);
  indices[i] = i;
}
