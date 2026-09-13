// LBVH stage 4: node bounds, computed per node from its own range.
//
// # Why not the textbook bottom-up walk
//
// Every published LBVH fits bounds by launching one thread per leaf, walking
// toward the root, and gating each node with an atomic so that the *second*
// child to arrive is the one that merges. It is O(n) and it is what CUDA
// implementations do.
//
// It is also not expressible safely in WGSL. The second thread to arrive reads
// `node_box[sibling]`, written by a thread in a different workgroup during the
// same dispatch. WGSL's atomics are relaxed and `storageBarrier()` synchronises
// a workgroup, not a device — so nothing in the memory model makes that write
// visible. It happens to work on most hardware, which is the worst property a
// race can have. Measured here before the rewrite: the root ended up covering
// 1319 of 10252 primitives, varying run to run.
//
// So the fit is restated as something with no inter-thread communication at all.
// A Karras node covers a contiguous range of the sorted array — the hierarchy
// kernel already computes it — so a node's box is just the union over that
// range, readable straight from `tri_bounds`. Nothing is read that this dispatch
// wrote.
//
// # What it costs
//
// O(n log n) work instead of O(n): a node scans its whole range, and the ranges
// at each level sum to n. That trade is only worth it because the constant is
// tiny and the work is perfectly parallel — but getting the parallelism right
// took three attempts, and the measured numbers are below at `WORKGROUP_RANGE`.
//
// Measured at 368k triangles, this stage costs 3.2 ms against the sort's 10.9 ms
// — so the asymptotically worse algorithm is both the faster thing to write and
// the easier thing to trust.

struct BuildParams {
  count : u32,
  _pad0 : u32,
  _pad1 : u32,
  _pad2 : u32,
};

struct AabbGpu {
  lo : vec3<f32>,
  _p0 : f32,
  hi : vec3<f32>,
  _p1 : f32,
};

@group(0) @binding(0) var<uniform> P: BuildParams;
@group(0) @binding(1) var<storage, read>       ranges     : array<vec2<u32>>;
@group(0) @binding(2) var<storage, read>       tri_bounds : array<AabbGpu>;
@group(0) @binding(3) var<storage, read>       order      : array<u32>;
@group(0) @binding(4) var<storage, read_write> node_box   : array<AabbGpu>;
@group(0) @binding(5) var<storage, read_write> subtree    : array<u32>;
// Indices of nodes too big for one thread, appended by `internal_small`.
@group(0) @binding(6) var<storage, read_write> large_list : array<u32>;
// Indirect dispatch arguments for `internal_large`. x is the number of entries
// appended above; y and z are pre-set to 1 by the host.
@group(0) @binding(7) var<storage, read_write> large_args : LargeArgs;

struct LargeArgs {
  x : atomic<u32>,
  y : u32,
  z : u32,
};

const WG: u32 = 256u;

// Above this a node gets a whole workgroup; at or below it, one thread.
//
// This split is the difference between a GPU build that beats the CPU and one
// that loses to it. Node ranges in a Morton tree are heavily skewed: in a
// balanced tree of n leaves, about seven eighths of all nodes sit in the bottom
// three levels with ranges under eight, while a handful at the top cover most
// of the scene.
//
// Thread-per-node therefore has a terrible critical path — the root's thread
// scans every primitive — and measured at 368k triangles it put the fit at
// ~80 ms. Workgroup-per-node fixes the critical path and replaces it with
// launch overhead: 368k workgroups of 256 threads, of which ~322k exist to run
// eight iterations in a single thread. That measured 43 ms, still most of the
// build.
//
// Splitting by size gives each node the right amount of parallelism. The small
// ones — almost all of them — cost one thread each in a normal grid dispatch.
// The large ones are appended to a list and dispatched indirectly, so the second
// kernel launches a few thousand workgroups rather than a few hundred thousand.
const WORKGROUP_RANGE: u32 = 256u;

// One leaf per sorted position. `order` is the sort's payload, so leaf k holds
// the primitive that sorted into position k.
@compute @workgroup_size(256, 1, 1)
fn leaves(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x;
  if (k >= P.count) {
    return;
  }
  node_box[(P.count - 1u) + k] = tri_bounds[order[k]];
}

// Merge a range of leaf boxes serially, from one thread.
//
// Reads the *leaf* array, which `leaves` wrote in sorted order, rather than
// `tri_bounds[order[k]]`. Same values, and the difference is everything: `order`
// is a permutation, so indexing through it is a random 32-byte gather per
// element. Reading the leaf array is sequential, so it coalesces and prefetches.
fn merge_serial(first: u32, last: u32, leaf_base: u32) -> AabbGpu {
  var lo = node_box[leaf_base + first].lo;
  var hi = node_box[leaf_base + first].hi;
  for (var k = first + 1u; k <= last; k = k + 1u) {
    let b = node_box[leaf_base + k];
    lo = min(lo, b.lo);
    hi = max(hi, b.hi);
  }
  var m: AabbGpu;
  m.lo = lo;
  m.hi = hi;
  m._p0 = 0.0;
  m._p1 = 0.0;
  return m;
}

// Small nodes: one thread each. Large ones are deferred to the list.
@compute @workgroup_size(256, 1, 1)
fn internal_small(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (P.count < 2u || i >= P.count - 1u) {
    return;
  }
  let r = ranges[i];
  let len = r.y - r.x + 1u;
  subtree[i] = len;

  if (len > WORKGROUP_RANGE) {
    let slot = atomicAdd(&large_args.x, 1u);
    large_list[slot] = i;
    return;
  }
  node_box[i] = merge_serial(r.x, r.y, P.count - 1u);
}

var<workgroup> red_lo: array<vec3<f32>, 256u>;
var<workgroup> red_hi: array<vec3<f32>, 256u>;

// Large nodes: one workgroup each, dispatched indirectly over the list above.
@compute @workgroup_size(256, 1, 1)
fn internal_large(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  // Every workgroup in this dispatch has a real entry — the dispatch size came
  // from the append counter — so there is no bounds check and no risk of a
  // barrier in non-uniform control flow.
  let i = large_list[wid.x];
  let r = ranges[i];
  let leaf_base = P.count - 1u;

  // An inverted box is the identity for union, so a thread whose stride lands
  // past the end contributes nothing.
  var lo = vec3<f32>(1e30, 1e30, 1e30);
  var hi = vec3<f32>(-1e30, -1e30, -1e30);
  for (var k = r.x + lid.x; k <= r.y; k = k + WG) {
    let b = node_box[leaf_base + k];
    lo = min(lo, b.lo);
    hi = max(hi, b.hi);
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
    var m: AabbGpu;
    m.lo = red_lo[0];
    m.hi = red_hi[0];
    m._p0 = 0.0;
    m._p1 = 0.0;
    // min and max are exact and order-independent on floats, so this union
    // agrees bit for bit with the CPU builder's pairwise bottom-up merge.
    node_box[i] = m;
  }
}
