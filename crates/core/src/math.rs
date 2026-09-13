//! Shared math: orthonormal bases, hemisphere sampling, and robust ray offsets.
//!
//! Every function here has a bit-for-bit twin in `shaders/common/math.wgsl`.
//! If you change one, change both — `cargo test -p pt-gpu` compares them
//! numerically.

use glam::{Vec2, Vec3};

pub const PI: f32 = std::f32::consts::PI;
pub const INV_PI: f32 = std::f32::consts::FRAC_1_PI;

/// Build an orthonormal basis around a unit vector `n`, branchlessly.
///
/// Duff et al. 2017, "Building an Orthonormal Basis, Revisited". The naive
/// approach (cross `n` with whichever cardinal axis it is least aligned to)
/// branches and loses precision near the axis boundaries; this version is exact
/// to within a few ULP for every input including `n.z` very close to -1, where
/// the older Frisvad formulation catastrophically cancels.
///
/// Returns `(tangent, bitangent)` such that `(t, b, n)` is right-handed.
#[inline]
pub fn onb(n: Vec3) -> (Vec3, Vec3) {
    let sign = (1.0f32).copysign(n.z);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    (
        Vec3::new(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x),
        Vec3::new(b, sign + n.y * n.y * a, -n.y),
    )
}

/// Transform a direction from the local frame where `n` is +Z into world space.
#[inline]
pub fn to_world(local: Vec3, n: Vec3) -> Vec3 {
    let (t, b) = onb(n);
    t * local.x + b * local.y + n * local.z
}

/// Transform a world-space direction into the local frame where `n` is +Z.
///
/// The inverse of [`to_world`]. Because the basis is orthonormal, the inverse is
/// the transpose, which is three dot products — no matrix inverse required.
#[inline]
pub fn to_local(world: Vec3, n: Vec3) -> Vec3 {
    let (t, b) = onb(n);
    Vec3::new(world.dot(t), world.dot(b), world.dot(n))
}

/// Cosine-weighted sample on the hemisphere around +Z, via Malley's method.
///
/// # Derivation
///
/// We want `p(omega) = cos(theta) / pi` (normalised: the integral of
/// `cos(theta)/pi` over the hemisphere is 1). Malley's method says: sample a
/// point uniformly on the unit *disk* and project it up onto the hemisphere.
///
/// Uniform on the disk is `p(x, y) = 1/pi`. The projection
/// `z = sqrt(1 - x^2 - y^2)` has Jacobian `dA_disk / dA_hemisphere = cos(theta)`
/// (the disk is the hemisphere's shadow, foreshortened by exactly `cos theta`),
/// so the induced density on the hemisphere is `(1/pi) * cos(theta)`. Which is
/// what we wanted, with no rejection and no trig.
///
/// We use a concentric (Shirley–Chiu) disk map rather than the polar map
/// `(r, theta) = (sqrt(u1), 2*pi*u2)`: the polar map badly distorts area near
/// the centre, which degrades stratification and shows up later when we feed
/// this a low-discrepancy sequence instead of white noise.
#[inline]
pub fn sample_cosine_hemisphere(u: Vec2) -> Vec3 {
    let d = concentric_sample_disk(u);
    // Guard against the tiny negative that `1 - (x^2+y^2)` can produce at the
    // rim through rounding; `sqrt` of it would be NaN and would poison the pixel.
    let z = (1.0 - d.x * d.x - d.y * d.y).max(0.0).sqrt();
    Vec3::new(d.x, d.y, z)
}

/// PDF of [`sample_cosine_hemisphere`], with respect to **solid angle**.
#[inline]
pub fn cosine_hemisphere_pdf(cos_theta: f32) -> f32 {
    (cos_theta * INV_PI).max(0.0)
}

/// Shirley–Chiu concentric mapping from the unit square to the unit disk.
///
/// Maps squares to squares-turned-wedges rather than to annuli, preserving
/// relative area and adjacency far better than the polar map.
#[inline]
pub fn concentric_sample_disk(u: Vec2) -> Vec2 {
    // Map [0,1)^2 to [-1,1)^2.
    let o = 2.0 * u - Vec2::ONE;
    if o.x == 0.0 && o.y == 0.0 {
        return Vec2::ZERO;
    }
    let (r, theta) = if o.x.abs() > o.y.abs() {
        (o.x, (PI / 4.0) * (o.y / o.x))
    } else {
        (o.y, PI / 2.0 - (PI / 4.0) * (o.x / o.y))
    };
    r * Vec2::new(theta.cos(), theta.sin())
}

/// Robust ray-origin offset (Wächter & Binder, *Ray Tracing Gems* ch. 6).
///
/// # Why not just `p + n * epsilon`
///
/// The spacing between representable `f32` values scales with magnitude. A fixed
/// epsilon that clears self-intersection near the origin is thousands of ULP too
/// small out at `|p| = 1e4`, and thousands of times too large near zero (which
/// detaches shadows and leaks light through contact points). The error we need
/// to clear is itself proportional to `|p|`, because it comes from rounding in
/// the intersection arithmetic.
///
/// So: offset by a fixed number of **ULP** instead of a fixed distance, by
/// adding an integer to the float's bit pattern. `int_scale` = 256 ULP is
/// empirically enough to clear the rounding error of a typical ray/primitive
/// intersection. Near the origin, ULP get so fine that this stops being
/// meaningful, so below `origin` we fall back to a small absolute offset.
#[inline]
pub fn offset_ray_origin(p: Vec3, n: Vec3) -> Vec3 {
    const ORIGIN: f32 = 1.0 / 32.0;
    const FLOAT_SCALE: f32 = 1.0 / 65536.0;
    const INT_SCALE: f32 = 256.0;

    let of_i = [
        (INT_SCALE * n.x) as i32,
        (INT_SCALE * n.y) as i32,
        (INT_SCALE * n.z) as i32,
    ];

    // Step the bit pattern outward along the normal. Adding to the bit pattern
    // of a positive float moves it away from zero; for a negative float it moves
    // it toward zero, hence the sign flip.
    let p_i = [
        f32::from_bits((p.x.to_bits() as i32).wrapping_add(if p.x < 0.0 {
            -of_i[0]
        } else {
            of_i[0]
        }) as u32),
        f32::from_bits((p.y.to_bits() as i32).wrapping_add(if p.y < 0.0 {
            -of_i[1]
        } else {
            of_i[1]
        }) as u32),
        f32::from_bits((p.z.to_bits() as i32).wrapping_add(if p.z < 0.0 {
            -of_i[2]
        } else {
            of_i[2]
        }) as u32),
    ];

    Vec3::new(
        if p.x.abs() < ORIGIN {
            p.x + FLOAT_SCALE * n.x
        } else {
            p_i[0]
        },
        if p.y.abs() < ORIGIN {
            p.y + FLOAT_SCALE * n.y
        } else {
            p_i[1]
        },
        if p.z.abs() < ORIGIN {
            p.z + FLOAT_SCALE * n.z
        } else {
            p_i[2]
        },
    )
}

/// Rec. 709 relative luminance of a linear RGB colour.
#[inline]
pub fn luminance(c: Vec3) -> f32 {
    c.dot(Vec3::new(0.2126, 0.7152, 0.0722))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn onb_is_orthonormal_everywhere() {
        let mut rng = Rng::new(1, 2, 3);
        // Include the pole that the naive and Frisvad constructions break at.
        let mut dirs = vec![Vec3::Z, -Vec3::Z, Vec3::new(0.0, 0.0, -1.0 + 1e-7)];
        for _ in 0..10_000 {
            let v = Vec3::new(
                rng.next_f32() * 2.0 - 1.0,
                rng.next_f32() * 2.0 - 1.0,
                rng.next_f32() * 2.0 - 1.0,
            );
            if v.length_squared() > 1e-6 {
                dirs.push(v.normalize());
            }
        }
        for n in dirs {
            let (t, b) = onb(n);
            assert!(t.dot(b).abs() < 1e-5, "t.b = {} for n = {n}", t.dot(b));
            assert!(t.dot(n).abs() < 1e-5, "t.n = {} for n = {n}", t.dot(n));
            assert!(b.dot(n).abs() < 1e-5, "b.n = {} for n = {n}", b.dot(n));
            assert!((t.length() - 1.0).abs() < 1e-5);
            assert!((b.length() - 1.0).abs() < 1e-5);
            // Right-handed: t x b == n.
            assert!(
                (t.cross(b) - n).length() < 1e-4,
                "handedness wrong for n = {n}"
            );
        }
    }

    /// The cosine-hemisphere sampler and its PDF must agree. This is a crude
    /// precursor to the full chi-squared test (build step 7): integrate 1/pdf
    /// over the samples, which must converge to the measure of the hemisphere,
    /// 2*pi.
    #[test]
    fn cosine_hemisphere_pdf_integrates_to_hemisphere_area() {
        let mut rng = Rng::new(9, 9, 9);
        let n = 2_000_000;
        let mut sum = 0.0f64;
        for _ in 0..n {
            let d = sample_cosine_hemisphere(rng.next_vec2());
            let pdf = cosine_hemisphere_pdf(d.z);
            assert!(d.z >= 0.0, "sample below the hemisphere: {d}");
            assert!((d.length() - 1.0).abs() < 1e-4, "not unit length: {d}");
            if pdf > 0.0 {
                sum += 1.0 / pdf as f64;
            }
        }
        let est = sum / n as f64;
        assert!(
            (est - 2.0 * std::f64::consts::PI).abs() < 5e-3,
            "E[1/pdf] = {est}, expected 2*pi = {}",
            2.0 * std::f64::consts::PI
        );
    }

    /// `to_local` and `to_world` must be exact inverses, or every BSDF
    /// evaluation is done in a subtly rotated frame.
    #[test]
    fn local_and_world_frames_round_trip() {
        let mut rng = Rng::new(21, 0, 0);
        for _ in 0..20_000 {
            let n = Vec3::new(
                rng.next_f32() * 2.0 - 1.0,
                rng.next_f32() * 2.0 - 1.0,
                rng.next_f32() * 2.0 - 1.0,
            );
            if n.length_squared() < 1e-6 {
                continue;
            }
            let n = n.normalize();
            let v = Vec3::new(
                rng.next_f32() * 2.0 - 1.0,
                rng.next_f32() * 2.0 - 1.0,
                rng.next_f32() * 2.0 - 1.0,
            )
            .normalize_or(Vec3::Z);

            let round_trip = to_world(to_local(v, n), n);
            assert!(
                (round_trip - v).length() < 1e-5,
                "round trip {v} -> {round_trip} for normal {n}"
            );
            // The normal itself must map to +Z exactly, which is the property
            // every BSDF's `wi.z` and `wo.z` depend on.
            let n_local = to_local(n, n);
            assert!(
                (n_local - Vec3::Z).length() < 1e-5,
                "normal maps to {n_local}, not +Z"
            );
        }
    }

    #[test]
    fn offset_ray_moves_off_the_surface_at_every_scale() {
        for &scale in &[1e-3f32, 1.0, 555.0, 1e5] {
            for &sign in &[1.0f32, -1.0] {
                let p = Vec3::splat(scale * sign);
                let n = Vec3::new(0.0, 1.0, 0.0);
                let o = offset_ray_origin(p, n);
                assert!(
                    o.y > p.y,
                    "no offset at scale {scale}, sign {sign}: {p} -> {o}"
                );
                // ...but not so far that geometry visibly detaches.
                assert!((o - p).length() < 1e-3 * scale.max(1.0) + 1e-4);
            }
        }
    }
}
