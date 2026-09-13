// Least-significant-digit radix sort over (key, value) pairs.
//
// The LBVH's second stage, and the only one that is not a trivial map: Karras's
// construction needs the Morton codes *sorted*, and a comparison sort has a
// dependency chain that a GPU cannot exploit. A radix sort has none — every
// element's destination is a function of digit counts, and counts are a
// reduction.
//
// Three kernels per pass, over one 4-bit digit:
//
//   HISTOGRAM  each workgroup counts its own elements' digits
//   SCAN       exclusive prefix sum over all (digit, workgroup) counts
//   SCATTER    each element reads its offset and writes itself there
//
// Eight passes of four bits covers the full 32-bit key. Morton codes only
// occupy 30, but the sort is written for u32 keys generally — it is also what
// sorts the reorder keys later — and skipping the top pass would save a pass
// that costs almost nothing on already-sorted data.
//
// # Why digit-major counts
//
// `counts[digit * num_groups + group]`, not `[group][digit]`. Scanning in that
// order places *every* element of digit 0 before any element of digit 1, which
// is what makes the pass a sort rather than a shuffle. Getting this transposed
// produces output that is locally ordered and globally wrong, and looks almost
// right in a spot check.
//
// # Stability
//
// Required, not incidental. Each pass must preserve the order the previous pass
// established or the lower digits it sorted are destroyed. Within a workgroup
// that means an element's rank among its same-digit neighbours is its position
// among those that precede it — see `local_rank`.

struct SortParams {
  count       : u32,   // elements to sort
  shift       : u32,   // bit position of the digit this pass handles
  num_groups  : u32,   // workgroups the histogram/scatter were dispatched with
  _pad0       : u32,
};

@group(0) @binding(0) var<uniform> P: SortParams;
@group(0) @binding(1) var<storage, read>       keys_in    : array<u32>;
@group(0) @binding(2) var<storage, read>       values_in  : array<u32>;
@group(0) @binding(3) var<storage, read_write> keys_out   : array<u32>;
@group(0) @binding(4) var<storage, read_write> values_out : array<u32>;
@group(0) @binding(5) var<storage, read_write> counts     : array<u32>;

const WG: u32 = 256u;
const RADIX_BITS: u32 = 4u;
const RADIX: u32 = 16u;

fn digit_of(key: u32) -> u32 {
  return (key >> P.shift) & (RADIX - 1u);
}

// ---------------------------------------------------------------------------
// HISTOGRAM
// ---------------------------------------------------------------------------

var<workgroup> hist: array<atomic<u32>, RADIX>;

@compute @workgroup_size(256, 1, 1)
fn histogram(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  if (lid.x < RADIX) {
    atomicStore(&hist[lid.x], 0u);
  }
  workgroupBarrier();

  let i = gid.x;
  if (i < P.count) {
    atomicAdd(&hist[digit_of(keys_in[i])], 1u);
  }
  workgroupBarrier();

  // Transposed on write, so the scan below runs in digit-major order.
  if (lid.x < RADIX) {
    counts[lid.x * P.num_groups + wid.x] = atomicLoad(&hist[lid.x]);
  }
}

// ---------------------------------------------------------------------------
// SCAN
// ---------------------------------------------------------------------------
//
// One workgroup, exclusive prefix sum over the whole `RADIX * num_groups`
// count array. Deliberately not a multi-level scan: that array has 16 entries
// per workgroup, so even at 368k primitives it is ~23k values — small enough
// that a single workgroup handles it in three phases, and a hierarchical scan
// would be more code for a stage that does not show up in a profile.
//
//   1. each thread reduces a contiguous chunk
//   2. one thread scans the 256 chunk totals
//   3. each thread re-walks its chunk, writing running totals from its base
//
// Dispatched as exactly one workgroup; `num_groups` is the *scatter's* group
// count, not this kernel's.

var<workgroup> chunk_totals: array<u32, WG>;

@compute @workgroup_size(256, 1, 1)
fn scan(@builtin(local_invocation_id) lid: vec3<u32>) {
  let total = RADIX * P.num_groups;
  // Ceiling division so the last thread's chunk absorbs the remainder.
  let chunk = (total + WG - 1u) / WG;
  let start = lid.x * chunk;
  let end = min(start + chunk, total);

  var sum = 0u;
  for (var i = start; i < end; i = i + 1u) {
    sum = sum + counts[i];
  }
  chunk_totals[lid.x] = sum;
  workgroupBarrier();

  // 256 serial adds in one thread. A Hillis-Steele scan here would be eight
  // barriers instead, and this runs once per pass over 256 values.
  if (lid.x == 0u) {
    var running = 0u;
    for (var t = 0u; t < WG; t = t + 1u) {
      let v = chunk_totals[t];
      chunk_totals[t] = running;
      running = running + v;
    }
  }
  workgroupBarrier();

  var running = chunk_totals[lid.x];
  for (var i = start; i < end; i = i + 1u) {
    let v = counts[i];
    counts[i] = running;
    running = running + v;
  }
}

// ---------------------------------------------------------------------------
// SCATTER
// ---------------------------------------------------------------------------

var<workgroup> local_digits: array<u32, WG>;

@compute @workgroup_size(256, 1, 1)
fn scatter(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let i = gid.x;
  let valid = i < P.count;
  let key = select(0u, keys_in[i], valid);
  let digit = digit_of(key);

  // RADIX is a sentinel no real digit can take, so out-of-range threads are
  // counted by nobody without needing a branch in the rank loop.
  local_digits[lid.x] = select(RADIX, digit, valid);
  workgroupBarrier();

  if (!valid) {
    return;
  }

  // Rank among this workgroup's earlier elements of the same digit.
  //
  // A linear scan over up to 256 predecessors, which is O(WG) per thread. The
  // textbook alternative is one prefix scan per digit — sixteen scans of eight
  // barrier steps each — and it is not obviously cheaper at this width. This
  // version is the one whose stability is checkable by reading it, which is
  // worth more here than the constant factor: an unstable radix sort destroys
  // the digits the previous pass ordered, and the symptom is a subtly worse
  // tree rather than a wrong one.
  var rank = 0u;
  for (var j = 0u; j < lid.x; j = j + 1u) {
    if (local_digits[j] == digit) {
      rank = rank + 1u;
    }
  }

  // `counts` now holds, for each (digit, workgroup), the global index where
  // that pair's run begins.
  let base = counts[digit * P.num_groups + wid.x];
  let dst = base + rank;
  keys_out[dst] = key;
  values_out[dst] = values_in[i];
}
