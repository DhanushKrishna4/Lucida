//! Display transforms.
//!
//! The renderer works in **linear HDR** from the camera to here. This module is
//! the only place that maps unbounded scene radiance onto the [0, 1] a display
//! can show, and it is deliberately separate from the integrator: switching
//! operators or exposure re-runs a fragment shader, not the light transport.
//!
//! # The pipeline, stated once
//!
//! ```text
//!   linear HDR radiance
//!     -> * exposure                 (a pure scale, in scene-linear)
//!     -> tone map                   (HDR -> display-linear, still linear!)
//!     -> sRGB transfer function     (display-linear -> display code values)
//! ```
//!
//! Every operator below returns **display-linear** values, *not* encoded ones.
//! The sRGB encode happens once, afterwards, in `linear_to_srgb`. This matters:
//! several well-known ACES and AgX snippets on the web bake a gamma step into
//! the curve, and applying sRGB on top of those double-encodes and visibly
//! washes out the shadows. Where a published fit ends in display-encoded space
//! (AgX does), it is linearised here so the contract holds.
//!
//! Each function is mirrored in `shaders/common/tonemap.wgsl`, and
//! `crates/gpu/tests/tonemap_agreement.rs` runs the WGSL on the GPU and compares
//! it against these implementations value for value — matrix transcription
//! errors are otherwise almost impossible to spot by eye.

use glam::{Mat3, Vec3};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tonemap {
    /// No curve: clamp to [0, 1]. Blows out highlights, but it is the only
    /// operator that leaves mid-tones numerically untouched, which makes it the
    /// right choice when eyeballing a render against reference values.
    Clamp,
    Reinhard,
    Aces,
    Agx,
}

impl Tonemap {
    pub const ALL: [Tonemap; 4] = [
        Tonemap::Clamp,
        Tonemap::Reinhard,
        Tonemap::Aces,
        Tonemap::Agx,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Tonemap::Clamp => "clamp",
            Tonemap::Reinhard => "reinhard",
            Tonemap::Aces => "aces",
            Tonemap::Agx => "agx",
        }
    }

    /// Index passed to the shader. Must match the `TONEMAP_*` constants in
    /// `shaders/common/tonemap.wgsl`.
    pub fn index(self) -> u32 {
        match self {
            Tonemap::Clamp => 0,
            Tonemap::Reinhard => 1,
            Tonemap::Aces => 2,
            Tonemap::Agx => 3,
        }
    }

    pub fn parse(s: &str) -> Option<Tonemap> {
        Tonemap::ALL.into_iter().find(|t| t.name() == s)
    }

    pub fn apply(self, c: Vec3) -> Vec3 {
        match self {
            Tonemap::Clamp => c.clamp(Vec3::ZERO, Vec3::ONE),
            Tonemap::Reinhard => reinhard(c),
            Tonemap::Aces => aces(c),
            Tonemap::Agx => agx(c),
        }
    }
}

/// Reinhard: `x / (1 + x)`.
///
/// Applied per channel, which is the simple form. It never clips, but it also
/// desaturates bright colours toward white as each channel independently
/// approaches 1, and it drains contrast from the mid-tones. Included because it
/// is the reference point everyone knows, and because the difference between it
/// and a filmic curve is the clearest demonstration that "tone mapping" is a
/// creative choice rather than a formula.
#[inline]
pub fn reinhard(c: Vec3) -> Vec3 {
    (c / (Vec3::ONE + c)).max(Vec3::ZERO)
}

// ACES, via Stephen Hill's fit of the RRT + sRGB ODT.
//
// Deliberately *not* Narkowicz's one-liner `(x*(2.51x+0.03))/(x*(2.43x+0.59)+0.14)`.
// That fit was made against the ODT's *output*, which is already display-encoded,
// so whether a gamma step belongs after it is genuinely ambiguous — and the
// ambiguity is why so many implementations of it are subtly wrong. Hill's fit is
// unambiguous: linear sRGB in, display-linear sRGB out.
//
// Matrices are column-major, matching `Mat3::from_cols_array` and WGSL's
// `mat3x3<f32>` constructor, so the Rust and WGSL transcriptions are identical.

#[rustfmt::skip]
const ACES_INPUT: [f32; 9] = [
    // columns
    0.59719, 0.07600, 0.02840,
    0.35458, 0.90834, 0.13383,
    0.04823, 0.01566, 0.83777,
];

#[rustfmt::skip]
const ACES_OUTPUT: [f32; 9] = [
    // columns
     1.60475, -0.10208, -0.00327,
    -0.53108,  1.10813, -0.07276,
    -0.07367, -0.00605,  1.07602,
];

#[inline]
fn rrt_and_odt_fit(v: Vec3) -> Vec3 {
    let a = v * (v + 0.024_578_6) - Vec3::splat(0.000_090_537);
    let b = v * (0.983_729 * v + Vec3::splat(0.432_951)) + Vec3::splat(0.238_081);
    a / b
}

#[inline]
pub fn aces(c: Vec3) -> Vec3 {
    let m_in = Mat3::from_cols_array(&ACES_INPUT);
    let m_out = Mat3::from_cols_array(&ACES_OUTPUT);
    let v = rrt_and_odt_fit(m_in * c.max(Vec3::ZERO));
    (m_out * v).clamp(Vec3::ZERO, Vec3::ONE)
}

// AgX (Troy Sobotka), using the widely-adopted minimal approximation.
//
// The idea: work in log2 exposure, run a sigmoid that desaturates *toward the
// path a real film stock takes* rather than toward white, and come back. The
// visible payoff is that very bright saturated colours — a blown-out coloured
// light, a sunlit red — keep their hue instead of turning into white blobs the
// way per-channel Reinhard makes them.
//
// The published curve ends in a display-encoded space, so the final `pow(2.2)`
// below brings it back to display-linear to satisfy this module's contract.

#[rustfmt::skip]
const AGX_INSET: [f32; 9] = [
    // columns
    0.856_627_2,  0.137_318_97, 0.111_898_21,
    0.095_121_24, 0.761_242,    0.076_799_42,
    0.048_251_606, 0.101_439_04, 0.811_302_4,
];

#[rustfmt::skip]
const AGX_OUTSET: [f32; 9] = [
    // columns
     1.127_100_6,  -0.141_329_76, -0.141_329_76,
    -0.110_606_64,  1.157_823_7,  -0.110_606_64,
    -0.016_493_939, -0.016_493_939, 1.251_936_4,
];

/// The dynamic range AgX maps, in stops around middle grey.
const AGX_MIN_EV: f32 = -12.473_93;
const AGX_MAX_EV: f32 = 4.026_069;

/// Sixth-order polynomial fit of the AgX contrast curve on [0, 1].
#[inline]
fn agx_contrast(x: Vec3) -> Vec3 {
    let x2 = x * x;
    let x4 = x2 * x2;
    15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x
        - Vec3::splat(0.00232)
}

#[inline]
pub fn agx(c: Vec3) -> Vec3 {
    let inset = Mat3::from_cols_array(&AGX_INSET);
    let outset = Mat3::from_cols_array(&AGX_OUTSET);

    let v = inset * c.max(Vec3::ZERO);
    // Floor before log2: pure black is legitimate in a render and log2(0) is
    // -inf, which would propagate NaN through the polynomial.
    let v = v.max(Vec3::splat(1e-10));
    let v = Vec3::new(v.x.log2(), v.y.log2(), v.z.log2());
    let v =
        ((v - Vec3::splat(AGX_MIN_EV)) / (AGX_MAX_EV - AGX_MIN_EV)).clamp(Vec3::ZERO, Vec3::ONE);
    let v = agx_contrast(v);
    let v = (outset * v).max(Vec3::ZERO);
    // Back to display-linear; see the module docs.
    Vec3::new(v.x.powf(2.2), v.y.powf(2.2), v.z.powf(2.2)).clamp(Vec3::ZERO, Vec3::ONE)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Black must stay black under every operator. An operator with a non-zero
    /// intercept lifts the entire image off the floor and makes a correct render
    /// look foggy.
    #[test]
    fn black_maps_to_black() {
        for t in Tonemap::ALL {
            let out = t.apply(Vec3::ZERO);
            assert!(out.max_element() < 2e-3, "{} maps black to {out}", t.name());
        }
    }

    /// Every operator must be non-decreasing. A curve that dips would make a
    /// brighter scene render darker in places — visually baffling and, unlike
    /// most tone mapping choices, unambiguously a bug.
    #[test]
    fn operators_are_monotonic() {
        for t in Tonemap::ALL {
            let mut prev = -1.0f32;
            let mut x = 0.0f32;
            while x < 64.0 {
                let y = t.apply(Vec3::splat(x)).y;
                assert!(
                    y >= prev - 1e-5,
                    "{} is not monotonic: f({x}) = {y} < previous {prev}",
                    t.name()
                );
                prev = y;
                x += 0.01;
            }
        }
    }

    /// Output must stay inside [0, 1] for any physically possible input,
    /// including the very large radiances an emitter produces.
    #[test]
    fn output_is_bounded() {
        for t in Tonemap::ALL {
            for &v in &[0.0f32, 1e-6, 0.18, 1.0, 18.4, 1e3, 1e6] {
                let out = t.apply(Vec3::splat(v));
                assert!(out.is_finite(), "{} produced {out} for input {v}", t.name());
                assert!(
                    out.min_element() >= 0.0 && out.max_element() <= 1.0,
                    "{} produced out-of-range {out} for input {v}",
                    t.name()
                );
            }
        }
    }

    /// Middle grey (scene-linear 0.18) must land near perceptual mid grey.
    ///
    /// Asserted in **sRGB-encoded** terms, because that is the space in which
    /// "looks like mid grey" is a meaningful statement — a display-linear bound
    /// would be an arbitrary number. The spread between operators here is not
    /// error, it is the operators disagreeing about exposure, which is exactly
    /// the point of offering more than one:
    ///
    ///   clamp    0.461  — the reference; no curve at all
    ///   reinhard 0.427  — slightly darker, the curve starts compressing early
    ///   aces     0.359  — noticeably darker; ACES is well known for pulling
    ///                     mid-tones down, which is why people often expose up
    ///   agx      0.501  — brightest; AgX places 0.18 almost exactly at 0.5
    #[test]
    fn middle_grey_lands_in_a_sane_place() {
        use crate::image::linear_to_srgb;
        let expectations: &[(Tonemap, f32)] = &[
            (Tonemap::Clamp, 0.4614),
            (Tonemap::Reinhard, 0.4269),
            (Tonemap::Aces, 0.3585),
            (Tonemap::Agx, 0.5005),
        ];
        for &(t, expected) in expectations {
            let srgb = linear_to_srgb(t.apply(Vec3::splat(0.18)).y);
            assert!(
                (srgb - expected).abs() < 0.01,
                "{} maps middle grey to sRGB {srgb:.4}, expected {expected:.4}",
                t.name()
            );
            // ...and regardless of the exact value, it must still read as grey
            // rather than as black or white.
            assert!(
                (0.30..0.60).contains(&srgb),
                "{} puts middle grey at {srgb:.3}",
                t.name()
            );
        }
    }

    /// A neutral grey input must stay neutral. The matrix round trip in ACES and
    /// AgX is where a transcription error shows up as a colour cast, and a cast
    /// on greys is both the most visible symptom and the easiest to assert.
    #[test]
    fn greys_stay_neutral() {
        for t in [Tonemap::Aces, Tonemap::Agx] {
            for &v in &[0.05f32, 0.18, 0.5, 1.0, 4.0] {
                let out = t.apply(Vec3::splat(v));
                let spread = out.max_element() - out.min_element();
                assert!(
                    spread < 5e-3,
                    "{} tints grey {v}: {out} (spread {spread})",
                    t.name()
                );
            }
        }
    }

    /// HSV hue angle in degrees.
    fn hue(c: Vec3) -> f32 {
        let (max, min) = (c.max_element(), c.min_element());
        let d = max - min;
        if d < 1e-9 {
            return 0.0;
        }
        let h = if max == c.x {
            ((c.y - c.z) / d).rem_euclid(6.0)
        } else if max == c.y {
            (c.z - c.x) / d + 2.0
        } else {
            (c.x - c.y) / d + 4.0
        };
        h * 60.0
    }

    /// ACES skews hue on bright saturated colours noticeably more than AgX.
    ///
    /// This is the substantive difference between the two, and the reason AgX
    /// exists: ACES's highlight handling rotates a saturated red toward orange
    /// (the well-documented "ACES red shift"). Measured on scene-linear
    /// (12, 1.2, 0.8), whose input hue is 2.1 degrees:
    ///
    ///   reinhard  12.6 deg   (+10.5)
    ///   aces      23.2 deg   (+21.1)   <- roughly twice the rotation
    ///   agx       12.6 deg   (+10.5)
    ///
    /// Note that *every* operator desaturates this colour heavily (input
    /// saturation 0.93 -> 0.27..0.52), and AgX desaturates it the most. That is
    /// intentional: AgX takes a deliberate "path to white" in the highlights,
    /// the way film does. Desaturation is a design choice; hue rotation is the
    /// artifact.
    #[test]
    fn aces_skews_hue_more_than_agx() {
        let bright_red = Vec3::new(12.0, 1.2, 0.8);
        let input_hue = hue(bright_red);

        let aces_shift = (hue(Tonemap::Aces.apply(bright_red)) - input_hue).abs();
        let agx_shift = (hue(Tonemap::Agx.apply(bright_red)) - input_hue).abs();

        assert!(
            aces_shift > agx_shift * 1.5,
            "expected ACES to skew hue substantially more than AgX, \
             got ACES {aces_shift:.1} deg vs AgX {agx_shift:.1} deg"
        );
        // And AgX must not be wildly rotating it either.
        assert!(agx_shift < 15.0, "AgX shifted hue by {agx_shift:.1} deg");
    }

    /// Filmic operators must keep headroom above scene-linear 1.0, so that a
    /// value of 1.0 is not already maximum brightness. Clamp, by definition,
    /// does not — which is what makes it useful as a reference and useless as a
    /// display transform for an HDR render.
    #[test]
    fn filmic_operators_preserve_highlight_headroom() {
        assert_eq!(Tonemap::Clamp.apply(Vec3::ONE).y, 1.0);
        for t in [Tonemap::Reinhard, Tonemap::Aces, Tonemap::Agx] {
            let at_one = t.apply(Vec3::ONE).y;
            let at_four = t.apply(Vec3::splat(4.0)).y;
            assert!(
                at_one < 0.8,
                "{} maps 1.0 to {at_one:.3}, leaving no headroom",
                t.name()
            );
            assert!(
                at_four > at_one,
                "{} does not keep rising past 1.0 ({at_one:.3} -> {at_four:.3})",
                t.name()
            );
            assert!(at_four < 1.0, "{} already clips at 4.0", t.name());
        }
    }

    #[test]
    fn names_round_trip() {
        for t in Tonemap::ALL {
            assert_eq!(Tonemap::parse(t.name()), Some(t));
        }
        assert_eq!(Tonemap::parse("nope"), None);
    }
}

#[cfg(test)]
mod matrix_invariants {
    use super::*;

    /// Every colour matrix here maps neutral to neutral, which means each **row**
    /// sums to 1 (because `M * (1,1,1)` must be `(1,1,1)`).
    ///
    /// This is the invariant that catches a transposed transcription, and it is
    /// worth asserting precisely because a transposed matrix still produces a
    /// plausible-looking image — it just quietly tints everything. Both AgX
    /// matrices were transposed when first written here; their rows summed to
    /// 1.106 instead of 1.0, and the symptom was a blue cast on dark greys.
    #[test]
    fn colour_matrices_preserve_neutral() {
        for (name, cols) in [
            ("ACES_INPUT", &ACES_INPUT),
            ("ACES_OUTPUT", &ACES_OUTPUT),
            ("AGX_INSET", &AGX_INSET),
            ("AGX_OUTSET", &AGX_OUTSET),
        ] {
            let m = Mat3::from_cols_array(cols);
            let out = m * Vec3::ONE;
            for i in 0..3 {
                assert!(
                    (out[i] - 1.0).abs() < 1e-4,
                    "{name} row {i} sums to {} — the matrix is probably transposed",
                    out[i]
                );
            }
        }
    }
}
