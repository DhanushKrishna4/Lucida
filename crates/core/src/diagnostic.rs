//! Diagnostic render modes: showing what the renderer is doing rather than what
//! the scene looks like.
//!
//! # Why these are worth building
//!
//! Almost every bug in this project so far produced an image that still looked
//! like the scene. A transposed BVH count, a normal transformed by the matrix
//! instead of its inverse-transpose, a sampler reading a stale sample index —
//! each one renders something plausible, and finding them meant diffing against
//! a reference that happened to exist.
//!
//! A diagnostic mode is a reference that does not need a second implementation.
//! Showing the **normals** makes a wrong transform obvious at a glance; showing
//! the **traversal cost** makes a badly-built BVH obvious in a way a render time
//! never quite does, because it says *where* the cost is.
//!
//! # They are nearly free
//!
//! Everything except the heatmap is already being computed. The denoiser's guide
//! channels — albedo, normal and depth at the first hit — live in the
//! accumulation buffer, so displaying them is a branch in the display pass and
//! no work at all in the tracer. The heatmap needed one counter and the spare
//! word those guides left behind.

use glam::Vec3;

/// What to display.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RenderMode {
    /// The render.
    #[default]
    Beauty,
    /// First-hit shading normal, remapped from `[-1, 1]` to `[0, 1]`.
    Normal,
    /// First-hit surface albedo, as the denoiser sees it.
    Albedo,
    /// First-hit distance, normalised against the scene's own scale.
    Depth,
    /// BVH node visits per ray, on a heat ramp.
    TraversalHeat,
}

impl RenderMode {
    pub fn name(self) -> &'static str {
        match self {
            RenderMode::Beauty => "beauty",
            RenderMode::Normal => "normal",
            RenderMode::Albedo => "albedo",
            RenderMode::Depth => "depth",
            RenderMode::TraversalHeat => "heat",
        }
    }

    pub fn parse(s: &str) -> Option<RenderMode> {
        match s {
            "beauty" | "colour" | "color" => Some(RenderMode::Beauty),
            "normal" | "normals" => Some(RenderMode::Normal),
            "albedo" => Some(RenderMode::Albedo),
            "depth" => Some(RenderMode::Depth),
            "heat" | "traversal" | "bvh" => Some(RenderMode::TraversalHeat),
            _ => None,
        }
    }

    pub fn index(self) -> u32 {
        match self {
            RenderMode::Beauty => 0,
            RenderMode::Normal => 1,
            RenderMode::Albedo => 2,
            RenderMode::Depth => 3,
            RenderMode::TraversalHeat => 4,
        }
    }

    pub fn all() -> [RenderMode; 5] {
        [
            RenderMode::Beauty,
            RenderMode::Normal,
            RenderMode::Albedo,
            RenderMode::Depth,
            RenderMode::TraversalHeat,
        ]
    }

    /// Whether this mode's output is a measurement rather than radiance.
    ///
    /// Tone mapping and exposure are for radiance; applying a filmic curve to a
    /// normal map would make a diagnostic lie about its own values, which is the
    /// one thing a diagnostic cannot do.
    pub fn is_data(self) -> bool {
        self != RenderMode::Beauty
    }
}

/// A perceptually-ordered heat ramp: black, blue, green, yellow, red, white.
///
/// Ordered by *lightness* as well as hue, so it reads correctly in greyscale and
/// for the eight percent of men with red-green colour blindness. A plain
/// blue-to-red ramp fails both — its middle is a lightness plateau, so the
/// interesting transition is where the eye can least see it.
///
/// The last stop is white rather than saturated red so the very worst cells stay
/// distinguishable from merely bad ones.
pub fn heat_ramp(t: f32) -> Vec3 {
    const STOPS: [(f32, [f32; 3]); 6] = [
        (0.00, [0.00, 0.00, 0.05]),
        (0.20, [0.10, 0.15, 0.60]),
        (0.45, [0.05, 0.65, 0.35]),
        (0.68, [0.95, 0.85, 0.10]),
        (0.87, [0.90, 0.20, 0.10]),
        (1.00, [1.00, 1.00, 1.00]),
    ];
    let t = t.clamp(0.0, 1.0);
    for w in STOPS.windows(2) {
        let (t0, c0) = w[0];
        let (t1, c1) = w[1];
        if t <= t1 {
            let f = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
            return Vec3::from_array(c0).lerp(Vec3::from_array(c1), f);
        }
    }
    Vec3::ONE
}

/// Node visits per ray that map to the top of the ramp.
///
/// Fixed rather than auto-scaled to the image's own maximum. Auto-scaling makes
/// a heatmap that always looks the same — the worst pixel is always white — so
/// it can never say whether a change made things better, which is the only
/// question anyone asks of one. A fixed scale means two heatmaps are comparable.
///
/// 128 because a well-built BVH over the project's heaviest scene averages about
/// 31 node visits per ray and peaks around four times that; see
/// `crates/core/src/bvh4.rs` for the measurements.
pub const HEAT_SCALE: f32 = 128.0;

/// What the display shows for one pixel, given its accumulated channels.
///
/// Mirrored in `shaders/display.wgsl`. Kept here rather than only in the shader
/// so the CLI can write the same images the browser shows, and so the two can be
/// diffed.
pub fn shade(
    mode: RenderMode,
    radiance: Vec3,
    albedo: Vec3,
    normal: Vec3,
    depth: f32,
    traversal: f32,
    depth_scale: f32,
) -> Vec3 {
    match mode {
        RenderMode::Beauty => radiance,
        // Remapped rather than shown raw: half of a normal's range is negative,
        // and clamping it to zero would make the two hemispheres identical.
        RenderMode::Normal => normal.normalize_or(Vec3::ZERO) * 0.5 + Vec3::splat(0.5),
        // Clamped, because the stored albedo deliberately includes emission —
        // that is what keeps the denoiser from demodulating a light fixture into
        // a division by a near-black base colour — and an emitter's is far above
        // 1. A diagnostic bypasses tone mapping, so it has to arrive
        // display-ready; an emitter simply reads as white.
        RenderMode::Albedo => albedo.clamp(Vec3::ZERO, Vec3::ONE),
        // Inverted so near is bright, which reads as depth rather than as fog,
        // and normalised against the scene's own scale so one image works for a
        // Cornell box measured in hundreds and a sphere measured in ones.
        RenderMode::Depth => {
            if depth <= 0.0 {
                Vec3::ZERO
            } else {
                Vec3::splat((1.0 - (depth / depth_scale.max(1e-6)).clamp(0.0, 1.0)).powf(0.75))
            }
        }
        RenderMode::TraversalHeat => heat_ramp(traversal / HEAT_SCALE),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes_round_trip_through_their_names() {
        for m in RenderMode::all() {
            assert_eq!(RenderMode::parse(m.name()), Some(m), "{}", m.name());
        }
        assert_eq!(RenderMode::parse("nonsense"), None);
    }

    /// Indices are a wire format shared with the shader.
    #[test]
    fn indices_are_distinct_and_dense() {
        let mut seen: Vec<u32> = RenderMode::all().iter().map(|m| m.index()).collect();
        seen.sort_unstable();
        assert_eq!(seen, vec![0, 1, 2, 3, 4]);
        assert_eq!(RenderMode::Beauty.index(), 0, "beauty must be the default 0");
    }

    /// The ramp must be continuous and increase in lightness.
    ///
    /// Both matter for reading a heatmap: a discontinuity reads as a contour line
    /// that is not there, and a lightness plateau hides the transition it is
    /// meant to show.
    #[test]
    fn the_heat_ramp_is_continuous_and_monotone_in_lightness() {
        let lum = |c: Vec3| crate::math::luminance(c);
        let mut prev = heat_ramp(0.0);
        let mut prev_l = lum(prev);
        for i in 1..=500 {
            let t = i as f32 / 500.0;
            let c = heat_ramp(t);
            assert!(
                (c - prev).length() < 0.05,
                "the ramp jumps at t = {t}: {prev:?} -> {c:?}"
            );
            let l = lum(c);
            assert!(
                l > prev_l - 0.02,
                "lightness fell at t = {t}: {prev_l:.3} -> {l:.3}. A heat ramp \
                 that is not monotone in lightness is unreadable in greyscale \
                 and to a red-green colour-blind viewer."
            );
            prev = c;
            prev_l = l;
        }
        assert!(lum(heat_ramp(1.0)) > 0.9, "the top of the ramp should be near white");
    }

    /// Out-of-range input must clamp rather than wrap or explode.
    #[test]
    fn the_ramp_clamps() {
        assert_eq!(heat_ramp(-5.0), heat_ramp(0.0));
        assert_eq!(heat_ramp(5.0), heat_ramp(1.0));
        for t in [-1.0f32, 0.0, 0.5, 1.0, 2.0] {
            assert!(heat_ramp(t).is_finite());
        }
    }

    /// Normals must map into the unit cube, both hemispheres distinctly.
    #[test]
    fn normals_map_into_the_display_range() {
        for n in [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z] {
            let c = shade(RenderMode::Normal, Vec3::ZERO, Vec3::ZERO, n, 1.0, 0.0, 10.0);
            assert!(
                c.min_element() >= -1e-6 && c.max_element() <= 1.0 + 1e-6,
                "normal {n:?} mapped to {c:?}, outside [0, 1]"
            );
        }
        // The two hemispheres must be distinguishable, which is the whole point
        // of the remap.
        let up = shade(RenderMode::Normal, Vec3::ZERO, Vec3::ZERO, Vec3::Y, 1.0, 0.0, 10.0);
        let down = shade(RenderMode::Normal, Vec3::ZERO, Vec3::ZERO, -Vec3::Y, 1.0, 0.0, 10.0);
        assert!((up - down).length() > 0.5, "opposing normals map to {up:?} and {down:?}");
    }

    /// Depth must read as near-bright, and a background ray must be black.
    #[test]
    fn depth_is_inverted_and_backgrounds_are_black() {
        let at = |d: f32| shade(RenderMode::Depth, Vec3::ZERO, Vec3::ZERO, Vec3::Z, d, 0.0, 100.0);
        assert_eq!(at(0.0), Vec3::ZERO, "a background ray should be black");
        assert!(at(1.0).x > at(50.0).x, "near should be brighter than far");
        assert!(at(200.0).x <= 1e-6, "past the scale should clamp to black");
    }
}
