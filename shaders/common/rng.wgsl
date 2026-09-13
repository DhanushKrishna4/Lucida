// PCG-RXS-M-XS with a 32-bit state. Mirrors `crates/core/src/rng.rs`.
//
// A 32-bit state is not a stylistic choice: WGSL has no 64-bit integer type, so
// the usual PCG32 cannot be expressed here at all. This variant is pure u32
// arithmetic and is therefore bit-identical to the Rust implementation, which is
// what lets the CPU and GPU tracers draw the *same* random stream and be
// compared far more sharply than Monte Carlo noise would allow.

fn pcg_hash(input: u32) -> u32 {
  let state = input * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

// Carries both samplers. Which one is live is `U.sampler_kind`, a uniform
// branch — every thread takes the same side, so it costs a predictable jump and
// no divergence.
struct Rng {
  state: u32,
  // Sobol: which point of the sequence, the per-pixel scramble seed, and the
  // next dimension to draw.
  index: u32,
  seed: u32,
  dim: u32,
};

const SAMPLER_SOBOL: u32 = 1u;

// A strong 32-bit hash for deriving scramble seeds.
//
// Deliberately not `pcg_hash`: that is the stream RNG's, and reusing it here
// would correlate the scramble seeds with the sample values whenever both come
// from the same pixel index.
fn sobol_hash(x0: u32) -> u32 {
  var x = x0;
  x = x ^ (x >> 16u);
  x = x * 0x7feb352du;
  x = x ^ (x >> 15u);
  x = x * 0x846ca68bu;
  x = x ^ (x >> 16u);
  return x;
}

// The index-th value of Sobol dimension `dim`, as a 32-bit fixed-point
// fraction. Fixed 32 iterations rather than an early exit: the trip count then
// does not depend on the index, which is what keeps a warp in lockstep.
fn sobol_value(index: u32, dim: u32) -> u32 {
  var x = 0u;
  for (var k = 0u; k < SOBOL_BITS; k = k + 1u) {
    if (((index >> k) & 1u) == 1u) {
      x = x ^ SOBOL_V[dim][k];
    }
  }
  return x;
}

// The Laine-Karras permutation: mixes each bit into all the bits below it and
// none above. That triangular dependency is exactly the structure of an Owen
// scramble, which is why four multiplies stand in for a tree of random numbers.
// The constants are Burley's and are not arbitrary.
fn laine_karras_permutation(x0: u32, seed: u32) -> u32 {
  var x = x0 + seed;
  x = x ^ (x * 0x6c50b47cu);
  x = x ^ (x * 0xb82f1e52u);
  x = x ^ (x * 0xc7afe638u);
  x = x ^ (x * 0x8d22f6e6u);
  return x;
}

// Reverse, permute, reverse back: the permutation mixes downward and an Owen
// scramble needs to mix from the most significant bit toward the least.
fn owen_scramble(x: u32, seed: u32) -> u32 {
  return reverseBits(laine_karras_permutation(reverseBits(x), seed));
}

// One scrambled Sobol value. Mirrors `sample` in crates/core/src/sobol.rs.
//
// Two scrambles, both needed: the **index** scramble shuffles which point of the
// sequence this sample gets, without which every pixel walks the same points in
// the same order; the **value** scramble is the Owen scramble proper, and is
// what keeps a deterministic sequence unbiased while preserving stratification.
fn sobol_sample(index: u32, dim: u32, seed: u32) -> f32 {
  let group = dim / SOBOL_DIMENSIONS;
  let base = dim % SOBOL_DIMENSIONS;
  let group_seed = sobol_hash(seed ^ (group * 0x9e3779b9u));
  let shuffled = owen_scramble(index, group_seed);
  let v = sobol_value(shuffled, base);
  let scrambled = owen_scramble(v, sobol_hash(group_seed + base + 1u));
  return f32(scrambled >> 8u) * (1.0 / 16777216.0);
}

// Counter-based seeding: hash (frame, pixel, sample) rather than advancing a
// shared sequence. No atomics, and a pixel's stream does not depend on the order
// threads happen to run in.
fn rng_init(frame_seed: u32, pixel_index: u32, sample_index: u32) -> Rng {
  var h = pcg_hash(frame_seed);
  h = pcg_hash(h + pixel_index);
  h = pcg_hash(h + sample_index);
  var r: Rng;
  r.state = h;
  // The *unmodified* sample index: a (0, 2)-sequence guarantees one point per
  // cell only for power-of-two-aligned prefixes, so offsetting the block would
  // give up the guarantee.
  r.index = sample_index;
  // Per pixel, not per sample — every sample of a pixel has to share a scramble
  // or they are not points of one stratified set.
  r.seed = sobol_hash(pixel_index + frame_seed * 0x9e3779b9u);
  r.dim = 0u;
  return r;
}

fn rng_next_u32(rng: ptr<function, Rng>) -> u32 {
  let s = (*rng).state * 747796405u + 2891336453u;
  (*rng).state = s;
  let word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (word >> 22u) ^ word;
}

// Uniform in [0, 1). Top 24 bits scaled by 2^-24: using all 32 bits can round up
// to exactly 1.0 in f32 and break half-open-interval assumptions downstream.
fn rng_next_f32(rng: ptr<function, Rng>) -> f32 {
  if (U.sampler_kind == SAMPLER_SOBOL) {
    let v = sobol_sample((*rng).index, (*rng).dim, (*rng).seed);
    (*rng).dim = (*rng).dim + 1u;
    return v;
  }
  return f32(rng_next_u32(rng) >> 8u) * (1.0 / 16777216.0);
}

fn rng_next_vec2(rng: ptr<function, Rng>) -> vec2<f32> {
  // Written as two statements, not `vec2(f(r), f(r))`: WGSL does not guarantee
  // the evaluation order of constructor arguments, and the draw order is part of
  // the CPU/GPU contract.
  if (U.sampler_kind == SAMPLER_SOBOL) {
    // Aligned to an even dimension. Base dimensions 0 and 1 are a strict
    // (0, 2)-sequence and no other pair is, so a 2-D draw starting at an odd
    // dimension would straddle two groups and get two *independent* values
    // instead of a stratified pair — correct, and pointless. The rule has to
    // match crates/core/src/sobol.rs exactly, since it shifts which dimension
    // every later draw receives.
    (*rng).dim = ((*rng).dim + 1u) & ~1u;
    let sx = sobol_sample((*rng).index, (*rng).dim, (*rng).seed);
    let sy = sobol_sample((*rng).index, (*rng).dim + 1u, (*rng).seed);
    (*rng).dim = (*rng).dim + 2u;
    return vec2<f32>(sx, sy);
  }
  let x = rng_next_f32(rng);
  let y = rng_next_f32(rng);
  return vec2<f32>(x, y);
}
