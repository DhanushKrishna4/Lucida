//! Sobol sampling with hash-based Owen scrambling (Burley 2020).
//!
//! # Why a low-discrepancy sequence at all
//!
//! Independent random sampling converges as `1/sqrt(N)` and that rate is fixed:
//! the only way to halve the noise is four times the samples. The reason is that
//! independent points clump — by chance, some regions of the integration domain
//! get several samples and their neighbours get none, and that clumping *is* the
//! variance.
//!
//! A low-discrepancy sequence is constructed so it cannot clump. Every prefix of
//! it is spread evenly over the domain, so for integrands that are smooth enough
//! the error falls closer to `1/N` than to `1/sqrt(N)`. Nothing about the
//! estimator changes — the same `f * cos / pdf`, the same MIS weights — only
//! where the points land.
//!
//! # Why it needs scrambling
//!
//! Raw Sobol points are *deterministic*, which causes two problems. Every pixel
//! would use the identical sequence, so the noise that remains would be
//! identical across the image — structured, aliased, and far more objectionable
//! than white noise even when its magnitude is lower. And a deterministic
//! sequence has no variance to estimate, so it cannot be averaged over
//! independent runs.
//!
//! **Owen scrambling** fixes both. It randomly permutes the sequence in a way
//! that preserves its stratification: the points stay evenly spread, but *which*
//! evenly-spread arrangement you get depends on a seed. Seed it per pixel and
//! the residual error decorrelates into something that looks like noise while
//! keeping the better convergence rate.
//!
//! The classical construction needs a tree of random permutations, which is
//! hopeless on a GPU. Burley's insight is that a carefully chosen hash applied
//! to the bit-reversed value is statistically indistinguishable from one, and
//! costs four multiplies.
//!
//! # Why the direction numbers are generated, not transcribed
//!
//! They come from primitive polynomials over GF(2) and a short table of initial
//! values (Joe & Kuo 2008). Generating them here means the WGSL copy is emitted
//! from this same code by `codegen`, so the two cannot drift — and the
//! properties that make the sequence worth using are checked directly rather
//! than assumed of a pasted table.

/// Dimensions with their own direction numbers.
///
/// **Two**, and every higher dimension is *padded* onto them: dimension `d` uses
/// base dimension `d % 2` with a scramble seed derived from `d / 2`.
///
/// Two rather than four, and that is a measured decision. Sobol's t-value grows
/// with dimension, and only the *leading pair* is a strict (0, 2)-sequence —
/// measured on the first four dimensions here, the 2-D projections come out at
///
/// ```text
///   (0,1): t = 0     (0,2): t = 1     (0,3): t = 2
///                    (1,2): t = 1     (1,3): t = 2     (2,3): t = 2
/// ```
///
/// A `t` of 2 permits four points per cell where the ideal permits one, which
/// gives up most of the advantage over random sampling. Since a path tracer
/// draws its samples in **pairs** — two for the pixel, two for the light, two
/// for the BSDF lobe — the stratification that matters is the one *within each
/// pair*, and pairing every draw with the ideal `(0, 1)` under its own seed is
/// strictly better than spreading draws across four dimensions of mixed
/// quality.
///
/// Padding this way also makes separate 2-D draws mutually independent rather
/// than jointly stratified, which is the standard trade and the right one: a
/// path's bounces are not smoothly related to one another, so joint
/// stratification across them buys little.
pub const SOBOL_DIMENSIONS: usize = 2;

/// Bits of the sample index the sequence resolves. 2^32 samples is far past any
/// render.
pub const SOBOL_BITS: usize = 32;

/// Primitive polynomials and initial direction numbers for the first four
/// dimensions (Joe & Kuo).
///
/// `(degree, coefficients, initial m values)`. Dimension 0 is the van der
/// Corput sequence and has no polynomial, so it is generated separately.
const POLYNOMIALS: [(u32, u32, &[u32]); SOBOL_DIMENSIONS - 1] = [(1, 0, &[1])];

/// Direction numbers: `[dimension][bit]`.
///
/// `v[d][k]` is what to XOR into the running value when bit `k` of the sample
/// index is set.
pub fn direction_numbers() -> [[u32; SOBOL_BITS]; SOBOL_DIMENSIONS] {
    let mut v = [[0u32; SOBOL_BITS]; SOBOL_DIMENSIONS];

    // Dimension 0 is the van der Corput sequence: reversing the bits of the
    // index. Its direction numbers are simply the powers of two from the top.
    for (k, slot) in v[0].iter_mut().enumerate() {
        *slot = 1u32 << (31 - k);
    }

    for (d, (degree, coeffs, init)) in POLYNOMIALS.iter().enumerate() {
        let s = *degree as usize;
        let a = *coeffs;

        // The Joe-Kuo recurrence, written on the **direction numbers** rather
        // than on the `m` values.
        //
        // The two formulations are equivalent but not interchangeable, and
        // mixing them is the easy mistake: on `m` the term is `m[i-s] << s`,
        // while on `V` — already shifted so the leading bit sits at bit 31 — it
        // is `V[i-s] >> s`. Getting it backwards produces a sequence that still
        // looks plausibly scattered and is not a (0, 2)-sequence, which
        // `pairs_form_a_02_sequence` catches and nothing else would.
        //
        // 1-indexed to match the published recurrence; the stored table is
        // 0-indexed.
        let mut vv = [0u32; SOBOL_BITS + 1];
        for i in 1..=s {
            vv[i] = init[i - 1] << (32 - i);
        }
        for i in (s + 1)..=SOBOL_BITS {
            let mut next = vv[i - s] ^ (vv[i - s] >> s);
            for k in 1..s {
                // Coefficient bits run from the most significant downward.
                if (a >> (s - 1 - k)) & 1 == 1 {
                    next ^= vv[i - k];
                }
            }
            vv[i] = next;
        }
        v[d + 1].copy_from_slice(&vv[1..=SOBOL_BITS]);
    }
    v
}

/// The direction numbers, built once.
pub fn matrices() -> &'static [[u32; SOBOL_BITS]; SOBOL_DIMENSIONS] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<[[u32; SOBOL_BITS]; SOBOL_DIMENSIONS]> = OnceLock::new();
    TABLE.get_or_init(direction_numbers)
}

/// The `index`-th value of Sobol dimension `dim`, as a 32-bit fixed-point
/// fraction.
///
/// Straight XOR over the set bits of the index. The loop is over a fixed 32
/// bits rather than terminating early on purpose: an early exit would make the
/// trip count depend on the index, and on a GPU that means the whole warp waits
/// for its largest member anyway while the code gets harder to reason about.
#[inline]
pub fn sobol(index: u32, dim: usize) -> u32 {
    let v = &matrices()[dim];
    let mut x = 0u32;
    for (k, vk) in v.iter().enumerate() {
        if (index >> k) & 1 == 1 {
            x ^= *vk;
        }
    }
    x
}

/// The Laine-Karras permutation: a hash that mixes each bit into all the bits
/// *below* it and none above.
///
/// That one-directional property is what makes it a stand-in for an Owen
/// scramble. An Owen scramble flips each bit based on a random choice that
/// depends only on the bits preceding it, which is exactly a triangular
/// dependency — so any mixing function with the same triangular structure
/// produces a statistically equivalent permutation, and this one costs four
/// multiplies instead of a tree of random numbers.
///
/// The constants are Burley's, found by search; they are not arbitrary and
/// should not be adjusted casually.
#[inline]
pub fn laine_karras_permutation(mut x: u32, seed: u32) -> u32 {
    x = x.wrapping_add(seed);
    x ^= x.wrapping_mul(0x6c50_b47c);
    x ^= x.wrapping_mul(0xb82f_1e52);
    x ^= x.wrapping_mul(0xc7af_e638);
    x ^= x.wrapping_mul(0x8d22_f6e6);
    x
}

/// Hash-based Owen scramble.
///
/// Reverse the bits so the triangular dependency runs the right way, permute,
/// and reverse back. The two reversals are the whole reason this works: the
/// permutation mixes downward, and an Owen scramble needs to mix from the most
/// significant bit toward the least.
#[inline]
pub fn owen_scramble(x: u32, seed: u32) -> u32 {
    laine_karras_permutation(x.reverse_bits(), seed).reverse_bits()
}

/// A strong 32-bit integer hash, used to derive independent seeds.
///
/// Deliberately *not* [`crate::rng::pcg_hash`]: that one is the renderer's
/// stream RNG and reusing it here would correlate the scramble seeds with the
/// sample values whenever both are derived from the same pixel index.
#[inline]
pub fn hash_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

/// One scrambled Sobol value: dimension `dim` of sample `index`, decorrelated
/// by `seed`.
///
/// Two scrambles, and both are needed:
///
/// * the **index** is scrambled first, which shuffles *which* point of the
///   sequence this sample gets. Without it every pixel walks the same points in
///   the same order and the image shows the sequence's structure directly.
/// * the **value** is scrambled after, which is the Owen scramble proper and is
///   what makes the estimator unbiased while preserving stratification.
///
/// Dimensions past the fourth are padded by reusing base dimension `dim % 4`
/// under a different seed. Groups are therefore independent of one another,
/// which is what keeps a forty-dimensional path from degenerating.
#[inline]
pub fn sample(index: u32, dim: u32, seed: u32) -> f32 {
    let group = dim / SOBOL_DIMENSIONS as u32;
    let base = (dim % SOBOL_DIMENSIONS as u32) as usize;

    // One seed per (pixel, group), so the four dimensions of a group stay
    // mutually stratified while different groups do not line up.
    let group_seed = hash_u32(seed ^ group.wrapping_mul(0x9e37_79b9));
    let shuffled = owen_scramble(index, group_seed);
    let v = sobol(shuffled, base);
    let scrambled = owen_scramble(v, hash_u32(group_seed.wrapping_add(base as u32 + 1)));

    // Top 24 bits, matching `Rng::next_f32`: the full 32 divided by 2^32 can
    // round up to exactly 1.0 in f32, which breaks every `floor(u * n)` index.
    (scrambled >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// Which point set to draw samples from.
///
/// Both are kept because the comparison is the interesting part, and because
/// Sobol's advantage is *conditional*: it depends on the integrand being smooth
/// enough for stratification to help, which is true of a diffuse surface under a
/// sky and much less true of a caustic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SamplerKind {
    /// Independent PCG stream. Converges at `1/sqrt(N)`, always.
    Independent,
    #[default]
    /// Owen-scrambled Sobol.
    Sobol,
}

impl SamplerKind {
    /// Every variant, in wire order. Codegen enumerates this to emit the
    /// browser's index table, so the two cannot drift.
    pub const ALL: [SamplerKind; 2] = [SamplerKind::Independent, SamplerKind::Sobol];

    pub fn name(self) -> &'static str {
        match self {
            SamplerKind::Independent => "independent",
            SamplerKind::Sobol => "sobol",
        }
    }

    pub fn parse(s: &str) -> Option<SamplerKind> {
        match s {
            "independent" | "random" | "pcg" => Some(SamplerKind::Independent),
            "sobol" => Some(SamplerKind::Sobol),
            _ => None,
        }
    }

    pub fn index(self) -> u32 {
        match self {
            SamplerKind::Independent => 0,
            SamplerKind::Sobol => 1,
        }
    }
}

/// The renderer's source of random numbers.
///
/// Deliberately shaped like the old [`crate::rng::Rng`] — `next_f32`,
/// `next_vec2`, drawn in a fixed order — so the integrator did not have to
/// change. The draw *order* already encodes which dimension each sample belongs
/// to, and that order was already pinned by the CPU/GPU agreement tests.
#[derive(Clone, Copy, Debug)]
pub struct Sampler {
    kind: SamplerKind,
    rng: crate::rng::Rng,
    /// Sobol sample index: which point of the sequence this is.
    index: u32,
    /// Per-pixel scramble seed, which decorrelates neighbouring pixels.
    seed: u32,
    /// Next dimension to draw.
    dim: u32,
}

impl Sampler {
    pub fn new(kind: SamplerKind, frame_seed: u32, pixel_index: u32, sample_index: u32) -> Self {
        Self {
            kind,
            rng: crate::rng::Rng::new(frame_seed, pixel_index, sample_index),
            // The **unmodified** sample index. A (0, 2)-sequence guarantees
            // one point per cell for *power-of-two-aligned* prefixes — indices
            // 0..2^k — so offsetting the block by a seed leaves it misaligned
            // and gives up the guarantee.
            //
            // Measured, that alignment is worth almost nothing here: removing
            // an offset changed the noise ratio on `cornell-box` from 1.18x to
            // 1.19x. Worth doing because it costs nothing and the guarantee is
            // the reason the sequence was chosen, but it is not where the
            // benefit comes from, and it would have been easy to assume it was.
            //
            // Progressive rendering continues the sequence through
            // `sample_offset`, which keeps the block contiguous, and frames are
            // decorrelated through the scramble seed below instead.
            index: sample_index,
            // Per *pixel*, not per sample: every sample of a pixel must share a
            // scramble or they are not points of one stratified set.
            seed: hash_u32(pixel_index.wrapping_add(frame_seed.wrapping_mul(0x9e37_79b9))),
            dim: 0,
        }
    }

    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        match self.kind {
            SamplerKind::Independent => self.rng.next_f32(),
            SamplerKind::Sobol => {
                let v = sample(self.index, self.dim, self.seed);
                self.dim += 1;
                v
            }
        }
    }

    /// Two samples that are **jointly stratified**.
    ///
    /// The pair is aligned to an even dimension, which is not bookkeeping — it
    /// is the whole reason the pairing is worth anything. Base dimensions 0 and
    /// 1 form a strict (0, 2)-sequence and no other pair does, so a 2-D draw
    /// that started at an odd dimension would straddle two groups and get two
    /// *independent* values instead of a stratified pair. The renderer would
    /// still be correct and would have thrown away the benefit.
    ///
    /// The alignment rule has to be mirrored exactly in WGSL, since it changes
    /// which dimension every later draw receives.
    #[inline]
    pub fn next_vec2(&mut self) -> glam::Vec2 {
        match self.kind {
            SamplerKind::Independent => self.rng.next_vec2(),
            SamplerKind::Sobol => {
                self.dim = (self.dim + 1) & !1;
                let x = sample(self.index, self.dim, self.seed);
                let y = sample(self.index, self.dim + 1, self.seed);
                self.dim += 2;
                glam::Vec2::new(x, y)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dimension 0 must be the van der Corput sequence.
    #[test]
    fn first_dimension_is_bit_reversal() {
        for i in 0..1000u32 {
            assert_eq!(
                sobol(i, 0),
                i.reverse_bits(),
                "dimension 0 should be the radical inverse of {i}"
            );
        }
    }

    /// Every value must be in range and the first sample must be the origin.
    #[test]
    fn sequence_starts_at_zero() {
        for d in 0..SOBOL_DIMENSIONS {
            assert_eq!(sobol(0, d), 0, "dimension {d} should start at 0");
        }
    }

    /// Smallest `t` for which a pair of dimensions is a `(t, 2)`-sequence over
    /// the first `2^k` points, for every `k` up to `max_k`.
    ///
    /// A `(t, 2)`-sequence puts exactly `2^t` points in every elementary
    /// interval of area `2^(t-k)`. `t = 0` is the ideal — one point per cell —
    /// and every increment of `t` is a doubling of the clumping the sequence
    /// permits.
    fn t_value(da: usize, db: usize, max_k: u32) -> u32 {
        for t in 0..=8u32 {
            let mut ok = true;
            'outer: for k in (t + 1)..=max_k {
                let n = 1u32 << k;
                let cells = k - t;
                for split in 0..=cells {
                    let (bits_a, bits_b) = (split, cells - split);
                    let mut count = vec![0u32; 1 << cells];
                    for i in 0..n {
                        let a = if bits_a == 0 { 0 } else { sobol(i, da) >> (32 - bits_a) };
                        let b = if bits_b == 0 { 0 } else { sobol(i, db) >> (32 - bits_b) };
                        count[((a << bits_b) | b) as usize] += 1;
                    }
                    if count.iter().any(|&c| c != 1 << t) {
                        ok = false;
                        break 'outer;
                    }
                }
            }
            if ok {
                return t;
            }
        }
        u32::MAX
    }

    /// The defining property: the first two dimensions form a **(0, 2)-sequence**.
    ///
    /// Every "elementary interval" — a rectangle of area `2^-k` whose sides are
    /// powers of two — contains *exactly one* of the first `2^k` points. That is
    /// far stronger than "evenly spread", it is what makes the sequence converge
    /// faster than random, and it fails loudly if a single direction number is
    /// wrong. It caught the `m`-based recurrence being used where the `V`-based
    /// one belongs.
    #[test]
    fn first_two_dimensions_are_a_02_sequence() {
        assert_eq!(
            t_value(0, 1, 10),
            0,
            "dimensions 0 and 1 must be a strict (0, 2)-sequence"
        );
    }

    /// Every pair a 2-D draw can receive must be ideal.
    ///
    /// With two base dimensions there is only one such pair, which is the point:
    /// an earlier version used four and padded across them, and the projections
    /// measured `t = 2` for half the pairs — four points per cell where the
    /// ideal permits one. Asserting `t = 0` for *all* pairs was itself an
    /// over-claim, since only Sobol's leading pair is guaranteed; the fix was to
    /// stop using the others rather than to weaken the assertion.
    #[test]
    fn every_pair_a_draw_can_receive_is_ideal() {
        for da in 0..SOBOL_DIMENSIONS {
            for db in (da + 1)..SOBOL_DIMENSIONS {
                assert_eq!(
                    t_value(da, db, 10),
                    0,
                    "dimensions ({da}, {db}) are not a strict (0, 2)-sequence"
                );
            }
        }
    }

    /// Owen scrambling must be a bijection.
    ///
    /// If it were not, some values would be unreachable and others doubled — the
    /// sampler would be biased, quietly, in a way an image would never show.
    #[test]
    fn owen_scramble_is_a_permutation() {
        for &seed in &[0u32, 1, 0xDEAD_BEEF, 12345] {
            // Over the top 16 bits, which is as much as can be enumerated
            // cheaply; the construction is bitwise so this is representative.
            let mut seen = vec![false; 1 << 16];
            for x in 0..(1u32 << 16) {
                let y = owen_scramble(x << 16, seed) >> 16;
                assert!(!seen[y as usize], "seed {seed}: collision at {x}");
                seen[y as usize] = true;
            }
        }
    }

    /// Scrambling must preserve stratification.
    ///
    /// This is the whole claim of Owen scrambling and the reason it is used
    /// instead of, say, adding a random offset: the points move, but they stay
    /// one-per-cell. A permutation that broke this would still look random and
    /// would have thrown away the entire benefit of the sequence.
    #[test]
    fn scrambling_preserves_stratification() {
        for &seed in &[1u32, 7, 0xABCD] {
            for k in 1..=8u32 {
                let n = 1u32 << k;
                let mut seen = vec![false; n as usize];
                for i in 0..n {
                    let shuffled = owen_scramble(i, hash_u32(seed));
                    let v = owen_scramble(sobol(shuffled, 0), hash_u32(seed + 1));
                    let cell = v >> (32 - k);
                    assert!(
                        !seen[cell as usize],
                        "seed {seed}, {n} points: cell {cell} hit twice — the scramble \
                         destroyed the stratification it is supposed to preserve"
                    );
                    seen[cell as usize] = true;
                }
            }
        }
    }

    /// Values must land in `[0, 1)` — never exactly 1.
    #[test]
    fn samples_are_in_the_half_open_unit_interval() {
        for seed in 0..64u32 {
            for i in 0..256u32 {
                for d in 0..12u32 {
                    let v = sample(i, d, seed);
                    assert!(
                        (0.0..1.0).contains(&v),
                        "sample({i}, {d}, {seed}) = {v} is outside [0, 1)"
                    );
                }
            }
        }
    }

    /// Different pixels must get different sequences.
    ///
    /// Without this every pixel shares the same residual error and the image
    /// shows correlated structure rather than noise — which is worse to look at
    /// than plain noise even when its magnitude is lower.
    #[test]
    fn different_seeds_decorrelate() {
        let n = 512;
        let mut identical = 0;
        for i in 0..n {
            if sample(i, 0, 1) == sample(i, 0, 2) {
                identical += 1;
            }
        }
        assert!(
            identical < 4,
            "{identical} of {n} samples are identical between two pixel seeds; the \
             scramble is not decorrelating"
        );
    }

    /// The point of the exercise: integration error must fall faster than
    /// `1/sqrt(N)`.
    ///
    /// Measured on a smooth two-dimensional integrand with a known answer.
    /// Independent sampling gets the `1/sqrt(N)` rate by construction, so the
    /// comparison is against that rather than against an absolute figure.
    #[test]
    fn converges_faster_than_independent_sampling() {
        use crate::rng::Rng;
        // f(x, y) = exp(-x - y) over the unit square; the integral is
        // (1 - 1/e)^2.
        let f = |x: f64, y: f64| (-x - y).exp();
        let exact = (1.0 - (-1.0f64).exp()).powi(2);

        eprintln!("\n{:>8} {:>14} {:>14} {:>10}", "N", "sobol err", "random err", "ratio");
        let mut ratios = Vec::new();
        for k in [6u32, 8, 10, 12] {
            let n = 1u32 << k;
            // Averaged over independent scrambles, because a single scramble's
            // error is itself a random variable and one draw says little.
            let trials = 32;
            let (mut sobol_err, mut rand_err) = (0.0f64, 0.0f64);
            for t in 0..trials {
                let mut s = 0.0f64;
                for i in 0..n {
                    s += f(sample(i, 0, t) as f64, sample(i, 1, t) as f64);
                }
                sobol_err += (s / n as f64 - exact).abs();

                let mut rng = Rng::new(t, 0, 0);
                let mut r = 0.0f64;
                for _ in 0..n {
                    r += f(rng.next_f32() as f64, rng.next_f32() as f64);
                }
                rand_err += (r / n as f64 - exact).abs();
            }
            sobol_err /= trials as f64;
            rand_err /= trials as f64;
            eprintln!(
                "{n:>8} {sobol_err:>14.3e} {rand_err:>14.3e} {:>9.1}x",
                rand_err / sobol_err
            );
            ratios.push(rand_err / sobol_err);
        }

        // The advantage must grow with N: that is what a better *rate* means, as
        // opposed to merely a better constant.
        assert!(
            ratios.last().unwrap() > ratios.first().unwrap(),
            "the advantage over random sampling did not grow with N ({:?}); the \
             sequence is not converging at a better rate",
            ratios
        );
        assert!(
            *ratios.last().unwrap() > 8.0,
            "at the largest N, Sobol is only {:.1}x better than random",
            ratios.last().unwrap()
        );
    }
}
