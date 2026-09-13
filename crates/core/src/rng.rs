//! Counter-seeded PCG random number generator.
//!
//! # Why this specific generator
//!
//! WGSL has **no 64-bit integer type**, so the usual PCG32 (64-bit LCG state,
//! 32-bit output) cannot be ported to a shader. We use PCG-RXS-M-XS with a
//! 32-bit state instead, which is entirely `u32` arithmetic and therefore
//! expressible *identically* in Rust and WGSL.
//!
//! That identity is the whole point. Because the CPU reference tracer and the
//! GPU tracer draw from bit-identical random streams for the same
//! (pixel, sample) pair, their images should agree to within floating-point
//! reassociation (~1e-5 relative), **not** merely to within Monte Carlo noise
//! (~1e-1 at practical sample counts). That turns "do CPU and GPU agree?" from
//! a fuzzy statistical question into a sharp one, and is the single most
//! valuable testing property in the project.
//!
//! Seeding is *counter-based*: the state is derived by hashing
//! (frame_seed, pixel_index, sample_index) rather than by advancing one global
//! sequence. No atomics, no inter-thread ordering, and a pixel's stream does not
//! depend on how many other pixels were traced first.

/// One step of the PCG-RXS-M-XS 32-bit output function.
///
/// Also usable standalone as an integer hash — it is a bijection on `u32` with
/// good avalanche, which is what we want for mixing seed components.
#[inline(always)]
pub fn pcg_hash(input: u32) -> u32 {
    let state = input.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28).wrapping_add(4))) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

#[derive(Clone, Copy, Debug)]
pub struct Rng {
    state: u32,
}

impl Rng {
    /// Seed from a (frame, pixel, sample) coordinate.
    ///
    /// Hashing is *sequential* (`h(h(h(a) + b) + c)`) rather than
    /// `h(a) ^ h(b) ^ h(c)`: XOR-combining independent hashes leaves visible
    /// correlation when two coordinates differ by a small amount, which shows up
    /// as structured patterns in the image at low sample counts.
    #[inline(always)]
    pub fn new(frame_seed: u32, pixel_index: u32, sample_index: u32) -> Self {
        let mut h = pcg_hash(frame_seed);
        h = pcg_hash(h.wrapping_add(pixel_index));
        h = pcg_hash(h.wrapping_add(sample_index));
        Self { state: h }
    }

    #[inline(always)]
    pub fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(747_796_405)
            .wrapping_add(2_891_336_453);
        let word = ((self.state >> ((self.state >> 28).wrapping_add(4))) ^ self.state)
            .wrapping_mul(277_803_737);
        (word >> 22) ^ word
    }

    /// Uniform in `[0, 1)`.
    ///
    /// Takes the top 24 bits and scales by 2^-24. Using the full 32 bits and
    /// dividing by 2^32 can round up to exactly 1.0 in `f32`, which breaks
    /// samplers that assume a half-open interval (e.g. `sqrt(1 - u)` guards,
    /// and any `floor(u * n)` index computation).
    #[inline(always)]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / 16_777_216.0)
    }

    #[inline(always)]
    pub fn next_vec2(&mut self) -> glam::Vec2 {
        // Note the explicit temporaries: Rust evaluates function arguments
        // left-to-right, but making the order explicit keeps this obviously
        // matched to the WGSL, where argument evaluation order is what we say
        // it is only if we write it this way.
        let x = self.next_f32();
        let y = self.next_f32();
        glam::Vec2::new(x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_in_half_open_unit_interval() {
        let mut rng = Rng::new(0, 0, 0);
        for _ in 0..1_000_000 {
            let u = rng.next_f32();
            assert!((0.0..1.0).contains(&u), "u = {u}");
        }
    }

    /// Sanity check on the mean and variance of the stream. A uniform [0,1)
    /// variable has mean 1/2 and variance 1/12; with 1e6 samples the standard
    /// error of the mean is sqrt(1/12/1e6) ~= 2.9e-4, so a 5-sigma band is
    /// ~1.5e-3.
    #[test]
    fn stream_is_uniform() {
        let mut rng = Rng::new(12345, 7, 0);
        let n = 1_000_000;
        let (mut sum, mut sum_sq) = (0.0f64, 0.0f64);
        for _ in 0..n {
            let u = rng.next_f32() as f64;
            sum += u;
            sum_sq += u * u;
        }
        let mean = sum / n as f64;
        let var = sum_sq / n as f64 - mean * mean;
        assert!((mean - 0.5).abs() < 1.5e-3, "mean = {mean}");
        assert!((var - 1.0 / 12.0).abs() < 1e-3, "var = {var}");
    }

    /// Neighbouring pixels must not produce correlated streams — this is the
    /// failure mode that XOR-combined seeding exhibits and that shows up as
    /// visible structure in the render.
    #[test]
    fn adjacent_seeds_decorrelate() {
        let n = 100_000;
        let mut cov = 0.0f64;
        for i in 0..n {
            let a = Rng::new(0, i, 0).next_f32() as f64 - 0.5;
            let b = Rng::new(0, i + 1, 0).next_f32() as f64 - 0.5;
            cov += a * b;
        }
        cov /= n as f64;
        // Independent => covariance 0, with standard error ~ (1/12)/sqrt(n).
        assert!(
            cov.abs() < 5.0 * (1.0 / 12.0) / (n as f64).sqrt(),
            "cov = {cov}"
        );
    }
}
