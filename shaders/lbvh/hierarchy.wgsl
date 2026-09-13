// LBVH stage 3: Karras's hierarchy over sorted Morton codes.
//
// One thread per internal node, and that is the whole point. Each node
// determines its own range and split by reading only the sorted code array —
// nothing another thread wrote, no ordering between threads, no atomics. A
// top-down build cannot do this because a node's range depends on the partition
// its parent chose; Karras's observation is that for a *sorted* array the tree
// is already implicit in the codes' shared prefixes, so the dependency is on the
// data rather than on the other nodes.
//
// Node ids live in one space of 2n-1 slots: [0, n-1) internal, [n-1, 2n-1)
// leaves. "Is this child a leaf" is then a comparison instead of a flag bit.
//
// Mirrors `build_hierarchy` in crates/core/src/lbvh.rs; the two are diffed
// against each other in `cargo test`, not eyeballed.

struct BuildParams {
  count : u32,
  _pad0 : u32,
  _pad1 : u32,
  _pad2 : u32,
};

@group(0) @binding(0) var<uniform> P: BuildParams;
@group(0) @binding(1) var<storage, read>       codes  : array<u32>;
// x = left child id, y = right child id.
@group(0) @binding(2) var<storage, read_write> karras : array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> parent : array<u32>;
// (first, last) sorted-array range covered by each internal node, inclusive.
//
// The construction computes these anyway, and writing them out is what lets the
// AABB fit be a pure function of the node — see fit.wgsl. Two extra words a node
// against an algorithm that needs no cross-workgroup memory ordering at all.
@group(0) @binding(4) var<storage, read_write> ranges : array<vec2<u32>>;

// Length of the shared prefix of codes i and j, with the index as a tiebreak.
//
// Duplicate codes are the subtle case. Two identical codes share all 32 bits and
// the range search would never terminate. Appending the index makes every key
// distinct — indices are distinct by construction — so the search always
// converges. The node it produces is poor (it splits spatially coincident
// primitives by array position) but valid, which is the right failure mode.
//
// Out-of-range returns -1, which is how the search terminates at the array's
// edges without a bounds test in the inner loop.
fn delta(i: i32, j: i32) -> i32 {
  if (j < 0 || j >= i32(P.count)) {
    return -1;
  }
  let a = codes[u32(i)];
  let b = codes[u32(j)];
  if (a == b) {
    return 32 + i32(countLeadingZeros(u32(i) ^ u32(j)));
  }
  return i32(countLeadingZeros(a ^ b));
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  // n - 1 internal nodes. A single-primitive tree has none.
  if (P.count < 2u || gid.x >= P.count - 1u) {
    return;
  }
  let i = i32(gid.x);

  // Which way does this node's range extend? Toward whichever neighbour agrees
  // with it more.
  let d = select(-1, 1, delta(i, i + 1) >= delta(i, i - 1));

  // Everything in the range must share more prefix with i than the neighbour on
  // the far side does, so that neighbour's agreement is the search's bound.
  let delta_min = delta(i, i - d);

  // Double to bracket, then binary search. Doubling first because the range is
  // unbounded — one node can cover the whole array — and starting a binary
  // search from n would cost log2(n) steps for every node however small.
  var l_max = 2;
  while (delta(i, i + l_max * d) > delta_min) {
    l_max = l_max * 2;
  }
  var l = 0;
  var t = l_max / 2;
  while (t >= 1) {
    if (delta(i, i + (l + t) * d) > delta_min) {
      l = l + t;
    }
    t = t / 2;
  }
  let j = i + l * d;
  let first = min(i, j);
  let last = max(i, j);

  // Where does it split? At the last position still sharing the node's own
  // prefix.
  let delta_node = delta(i, j);
  var s = 0;
  var t2 = l;
  while (t2 > 1) {
    // Ceiling division: the search must be able to reach `l` itself, and
    // halving a 1 to 0 would stop one short on odd lengths.
    t2 = (t2 + 1) / 2;
    if (delta(i, i + (s + t2) * d) > delta_node) {
      s = s + t2;
    }
  }
  let split = i + s * d + min(d, 0);

  let leaf_base = P.count - 1u;
  // A child is a leaf exactly when it is a single element of the range.
  let left  = select(u32(split),      leaf_base + u32(split),      split == first);
  let right = select(u32(split + 1),  leaf_base + u32(split + 1),  split + 1 == last);

  karras[gid.x] = vec2<u32>(left, right);
  ranges[gid.x] = vec2<u32>(u32(first), u32(last));
  parent[left] = gid.x;
  parent[right] = gid.x;
}
