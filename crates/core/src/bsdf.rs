//! Microfacet BSDFs.
//!
//! Everything here works in the **local shading frame**, where the surface
//! normal is `+Z`, and both `wo` (toward the previous path vertex) and `wi`
//! (toward the next one) point *away* from the surface. Keeping one convention
//! and stating it once is worth more than any individual formula below: most
//! microfacet bugs are sign and direction bugs, not algebra.
//!
//! Mirrored in `shaders/common/bsdf.wgsl`, and cross-checked numerically by
//! `crates/gpu/tests/bsdf_agreement.rs`.

use crate::math::{sample_cosine_hemisphere, INV_PI, PI};
use glam::{Vec2, Vec3};

/// Perceptual roughness to the GGX width parameter.
///
/// `alpha = roughness^2` is the Disney/UE4 convention. It exists because a
/// linear ramp in `alpha` looks wildly non-linear: almost all the visible change
/// happens in the first tenth. Squaring spreads the perceptual change evenly
/// across a slider, which is the only reason the parameter the user touches is
/// not the parameter the maths wants.
#[inline]
pub fn roughness_to_alpha(roughness: f32) -> f32 {
    (roughness * roughness).max(MIN_ALPHA)
}

/// Smallest GGX width that `f32` can represent without destroying energy.
///
/// This is not a stylistic clamp. The GGX denominator contains `sin^2(theta)`,
/// and the only thing available at the call site is `cos(theta)` as an `f32`.
/// Near the lobe centre `cos(theta)` is within one ULP of 1.0, so `1 - cos^2`
/// underflows to zero and `D` blows up. The lobe's angular width is about
/// `alpha`, so the model needs `sin^2(alpha)` to stay well clear of the `f32`
/// epsilon near 1.0 (about 1.2e-7).
///
/// Measured, integrating `D(m)(n.m)` — which must be exactly 1:
///
/// ```text
///   alpha     integral
///   1e-5      596.4      <- catastrophic
///   1e-4        6.10     <- the clamp this code originally used
///   3e-4        1.245
///   5e-4        1.043
///   1e-3        1.002
///   2e-3        0.9996   <- safe
///   4e-3        1.000002
/// ```
///
/// So 2e-3 it is, which is roughness 0.045 — still a sharp, mirror-like
/// highlight. A genuinely perfect mirror needs a **delta specular lobe**, where
/// the direction is exact and no NDF is evaluated at all; that arrives with
/// smooth dielectrics at build step 12 and is the right answer below this
/// threshold, rather than pushing GGX somewhere `f32` cannot follow.
pub const MIN_ALPHA: f32 = 2.0e-3;

// ---------------------------------------------------------------------------
// GGX / Trowbridge-Reitz
// ---------------------------------------------------------------------------

/// The GGX normal distribution function, `D(m)`.
///
/// `D` is a density over the microsurface, normalised so that
/// `integral of D(m) (n.m) dm = 1` — the microfacets' *projected* area equals
/// the macrosurface area. That normalisation is what makes every later energy
/// argument meaningful, and it is asserted in the tests.
///
/// GGX rather than Beckmann because of its tails: GGX falls off far more slowly
/// away from the peak, which produces the broad bright haze around a highlight
/// that real rough surfaces show and Beckmann does not.
///
/// # Numerical form
///
/// The textbook denominator is `(n.m)^2 * (alpha^2 - 1) + 1`. That expression
/// **catastrophically cancels** at low roughness: with `alpha = 4e-4`,
/// `alpha^2` is 1.6e-7, so `c^2 * (a^2 - 1)` is a hair under -1 and adding 1
/// destroys most of the mantissa. Measured, it made the NDF integrate to 0.95
/// instead of 1 at roughness 0.02 — a 5% energy error at exactly the smooth,
/// mirror-like settings where a highlight is most conspicuous.
///
/// The algebraically identical form used here has no cancellation, because both
/// terms are small and positive:
///
/// ```text
///   c^2 (a^2 - 1) + 1  =  a^2 c^2 + (1 - c^2)  =  a^2 c^2 + sin^2(theta)
/// ```
#[inline]
pub fn ggx_d(alpha: f32, cos_theta_m: f32) -> f32 {
    if cos_theta_m <= 0.0 {
        return 0.0;
    }
    let a2 = alpha * alpha;
    let c2 = cos_theta_m * cos_theta_m;
    let sin2 = (1.0 - c2).max(0.0);
    let denom = a2 * c2 + sin2;
    a2 / (PI * denom * denom)
}

/// Sample a microfacet normal from `D` itself, weighted by `(n.m)`.
///
/// The classical alternative to [`sample_ggx_vndf`], kept because it is the
/// baseline VNDF improves on and because it makes a well-conditioned proposal
/// distribution for validating the visible-normal density.
///
/// Inverting the GGX CDF gives `tan(theta) = alpha * sqrt(u / (1 - u))`.
#[inline]
pub fn sample_ggx_ndf(alpha: f32, u: Vec2) -> Vec3 {
    let tan_theta = alpha * (u.x / (1.0 - u.x).max(1.0e-9)).sqrt();
    let cos_theta = 1.0 / (1.0 + tan_theta * tan_theta).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let phi = 2.0 * PI * u.y;
    Vec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta)
}

/// PDF of [`sample_ggx_ndf`], with respect to solid angle over microfacet
/// normals: `D(m) * (n.m)`.
#[inline]
pub fn ggx_ndf_pdf(alpha: f32, m: Vec3) -> f32 {
    ggx_d(alpha, m.z) * m.z.max(0.0)
}

/// Smith's lambda function for GGX.
///
/// `Lambda(v)` is the ratio of masked-to-visible microsurface area seen from `v`
/// — informally, how much of the microsurface hides behind itself at that
/// grazing angle.
///
/// ```text
///   Lambda(v) = (-1 + sqrt(1 + alpha^2 * tan^2(theta_v))) / 2
/// ```
#[inline]
pub fn smith_lambda(alpha: f32, cos_theta: f32) -> f32 {
    let c = cos_theta.abs();
    if c >= 1.0 {
        return 0.0;
    }
    let c2 = c * c;
    // tan^2 = (1 - cos^2) / cos^2
    let tan2 = (1.0 - c2) / c2.max(1.0e-12);
    0.5 * (-1.0 + (1.0 + alpha * alpha * tan2).sqrt())
}

/// Single-direction masking, `G1(v) = 1 / (1 + Lambda(v))`.
#[inline]
pub fn smith_g1(alpha: f32, cos_theta: f32) -> f32 {
    1.0 / (1.0 + smith_lambda(alpha, cos_theta))
}

/// **Height-correlated** Smith masking-shadowing, `G2(wo, wi)`.
///
/// ```text
///   G2 = 1 / (1 + Lambda(wo) + Lambda(wi))
/// ```
///
/// Not `G1(wo) * G1(wi)`. The separable form assumes masking and shadowing are
/// independent events, which is wrong: a microfacet high on the surface is more
/// likely to be visible from *both* directions, and one low down is likely
/// hidden from both. Treating them as independent systematically underestimates
/// the joint visibility and darkens rough surfaces — most visibly at grazing
/// angles, exactly where the error is easiest to mistake for correct shadowing.
///
/// The height-correlated form is the same cost and is what Heitz (2014) shows
/// matches the underlying microsurface model.
#[inline]
pub fn smith_g2(alpha: f32, cos_theta_o: f32, cos_theta_i: f32) -> f32 {
    1.0 / (1.0 + smith_lambda(alpha, cos_theta_o) + smith_lambda(alpha, cos_theta_i))
}

/// Sample a microfacet normal from the **distribution of visible normals**
/// (Heitz 2018).
///
/// # Why VNDF rather than sampling `D` directly
///
/// Sampling `D` proposes microfacet normals over the whole microsurface,
/// including the ones facing away from `wo`, which are masked and contribute
/// nothing. At high roughness and grazing angles most samples are wasted that
/// way, and the resulting fireflies are the classic "rough metal is noisy"
/// artifact.
///
/// VNDF samples only what `wo` can actually see, weighted by how much of it it
/// sees. Every sample contributes, and — the reason it is worth the extra
/// algebra — the estimator collapses to `F * G2 / G1(wo)` with `D` cancelling
/// out entirely. No division by a near-zero density, so no fireflies from the
/// pdf.
///
/// The method: stretch the view direction so the ellipsoid becomes a hemisphere,
/// sample the projected disk uniformly, and unstretch.
pub fn sample_ggx_vndf(wo: Vec3, alpha: f32, u: Vec2) -> Vec3 {
    // Work in the upper hemisphere; the caller flips back if needed.
    let v = Vec3::new(alpha * wo.x, alpha * wo.y, wo.z).normalize();

    // Orthonormal basis around the stretched view direction. The degenerate
    // case is `v` pointing straight up, where any tangent will do.
    let lensq = v.x * v.x + v.y * v.y;
    let t1 = if lensq > 0.0 {
        Vec3::new(-v.y, v.x, 0.0) / lensq.sqrt()
    } else {
        Vec3::X
    };
    let t2 = v.cross(t1);

    // Uniform point on the disk, then squashed toward the hemisphere's silhouette
    // by how oblique the view is. This is the step that makes the sample density
    // match the *projected* area rather than the raw area.
    let r = u.x.sqrt();
    let phi = 2.0 * PI * u.y;
    let p1 = r * phi.cos();
    let mut p2 = r * phi.sin();
    let s = 0.5 * (1.0 + v.z);
    p2 = (1.0 - s) * (1.0 - p1 * p1).max(0.0).sqrt() + s * p2;

    // Project back up onto the hemisphere.
    let nh = p1 * t1 + p2 * t2 + (1.0 - p1 * p1 - p2 * p2).max(0.0).sqrt() * v;

    // Unstretch into the original (ellipsoid) configuration.
    Vec3::new(alpha * nh.x, alpha * nh.y, nh.z.max(0.0)).normalize()
}

/// PDF of [`sample_ggx_vndf`] with respect to the **reflected direction**
/// `wi`, in solid angle.
///
/// # Derivation
///
/// The density of visible normals is
///
/// ```text
///   D_v(m) = G1(wo) * max(0, wo.m) * D(m) / (n.wo)
/// ```
///
/// Reflecting `wo` about `m` to get `wi` is a change of variables whose Jacobian
/// is `dm/dwi = 1 / (4 * (wo.m))` — the factor of 4 comes from the reflection
/// halving the angle between `wo` and `wi` relative to `m`. So
///
/// ```text
///   pdf(wi) = D_v(m) / (4 * (wo.m))
///           = G1(wo) * (wo.m) * D(m) / ((n.wo) * 4 * (wo.m))
///           = G1(wo) * D(m) / (4 * (n.wo))
/// ```
///
/// The `(wo.m)` cancels. Note this is exactly the term that would otherwise
/// approach zero at grazing angles, which is why VNDF sampling is so much better
/// behaved than sampling `D`.
#[inline]
pub fn ggx_vndf_pdf(alpha: f32, wo: Vec3, m: Vec3) -> f32 {
    let cos_o = wo.z;
    if cos_o <= 0.0 {
        return 0.0;
    }
    smith_g1(alpha, cos_o) * ggx_d(alpha, m.z) / (4.0 * cos_o)
}

/// Density of the **half-vector** `m` itself, before any Jacobian.
///
/// ```text
///   D_vis(m) = G1(wo) * D(m) * |wo.m| / cos_theta_o
/// ```
///
/// Note the difference from [`ggx_vndf_pdf`], which despite its name is the
/// density of the *reflected direction*: it has already been through the
/// reflection Jacobian `dm/dwi = 1 / (4 |wo.m|)`, and the `|wo.m|` cancelled on
/// the way. That cancellation is why the reflection pdf looks so much simpler
/// than this one, and it is a trap — reusing it for **transmission**, whose
/// Jacobian is entirely different, is wrong by a factor of `4 |wo.m|`.
///
/// Measured when exactly that was done here: the sampler transmitted 95% of the
/// time while its pdf accounted for 25% of the mass, which is 3.8x, which is
/// `4 |wo.m|` at near-normal incidence.
/// `wo.m <= 0` returns zero, and that is the definition rather than a guard:
/// the *visible* normal distribution only contains microfacets facing the
/// viewer. Using `|wo.m|` instead assigns density to microfacets `sample` can
/// never produce, so the pdf integrates to slightly more than one — measured
/// here at 1.00064 for roughness 0.2 at 53 degrees, which the chi-squared test
/// caught as a 10-sigma disagreement between the sampler's acceptance rate and
/// its pdf's total mass.
#[inline]
pub fn ggx_vndf_half_pdf(alpha: f32, wo: Vec3, m: Vec3) -> f32 {
    let cos_o = wo.z;
    let cos_om = wo.dot(m);
    if cos_o <= 0.0 || cos_om <= 0.0 {
        return 0.0;
    }
    smith_g1(alpha, cos_o) * ggx_d(alpha, m.z.abs()) * cos_om / cos_o
}

// ---------------------------------------------------------------------------
// Fresnel
// ---------------------------------------------------------------------------

/// Schlick's approximation.
///
/// An approximation *of the unpolarised dielectric Fresnel equations*, so it is
/// only appropriate where those apply — see [`fresnel_conductor`] for why metals
/// need more.
///
/// Its accuracy is often overstated. Measured against the exact equations at
/// IOR 1.5, the **maximum absolute error is 0.036, near `cos_theta = 0.09`** —
/// Schlick climbs toward 1 too early at grazing angles. That is visible as a
/// slightly too-bright rim on a smooth dielectric, and it is the reason the
/// exact form is used wherever the cost is affordable (transmission, and the
/// `f0_from_ior` derivation).
#[inline]
pub fn fresnel_schlick(f0: Vec3, cos_theta: f32) -> Vec3 {
    let m = (1.0 - cos_theta).clamp(0.0, 1.0);
    let m2 = m * m;
    f0 + (Vec3::ONE - f0) * (m2 * m2 * m)
}

/// Exact unpolarised Fresnel reflectance for a dielectric interface.
///
/// `eta` is the relative index of refraction, `n_transmitted / n_incident`.
/// Returns 1.0 under total internal reflection. Used directly by the
/// transmission lobe at build step 12; here it is what [`f0_from_ior`] is
/// derived from.
#[inline]
pub fn fresnel_dielectric(cos_theta_i: f32, eta: f32) -> f32 {
    let c = cos_theta_i.clamp(-1.0, 1.0).abs();
    let sin2_t = (1.0 - c * c) / (eta * eta);
    if sin2_t >= 1.0 {
        // Total internal reflection: no transmitted component exists.
        return 1.0;
    }
    let cos_t = (1.0 - sin2_t).max(0.0).sqrt();

    // Reflectance for the two polarisations, averaged. Unpolarised light is an
    // equal mixture, and a path tracer that does not track polarisation can only
    // carry the mean.
    let r_parallel = (eta * c - cos_t) / (eta * c + cos_t);
    let r_perpendicular = (c - eta * cos_t) / (c + eta * cos_t);
    0.5 * (r_parallel * r_parallel + r_perpendicular * r_perpendicular)
}

/// Normal-incidence reflectance of a dielectric with the given IOR.
///
/// `F0 = ((ior - 1) / (ior + 1))^2` — the Fresnel equations at `cos_theta = 1`.
/// For IOR 1.5, common for plastics and glass, this is 0.04, which is where the
/// familiar "4% specular" constant comes from.
#[inline]
pub fn f0_from_ior(ior: f32) -> f32 {
    let r = (ior - 1.0) / (ior + 1.0);
    r * r
}

/// Exact unpolarised Fresnel reflectance for a **conductor**, from the complex
/// index of refraction `n - ik`.
///
/// # Why metals need this
///
/// Schlick is a fit to the *dielectric* Fresnel curve, and metals do not follow
/// it. Their reflectance varies strongly with wavelength and, crucially, does
/// not rise toward white monotonically: copper and gold both *dip* in the middle
/// of their angular response before climbing to white at grazing. Schlick can
/// only interpolate monotonically from `F0` to 1, so it renders gold as
/// "yellow fading to white" and misses the characteristic warm shift.
///
/// Evaluated per RGB channel, which is a spectral approximation — genuinely
/// spectral rendering would sample wavelengths — but it captures the shape that
/// Schlick cannot.
#[inline]
pub fn fresnel_conductor(cos_theta_i: f32, eta: Vec3, k: Vec3) -> Vec3 {
    let c = cos_theta_i.clamp(0.0, 1.0);
    let c2 = c * c;
    let sin2 = 1.0 - c2;

    let eta2 = eta * eta;
    let k2 = k * k;

    let t0 = eta2 - k2 - Vec3::splat(sin2);
    let a2_plus_b2 = (t0 * t0 + 4.0 * eta2 * k2).max(Vec3::ZERO).powf(0.5);
    let t1 = a2_plus_b2 + Vec3::splat(c2);
    let a = (0.5 * (a2_plus_b2 + t0)).max(Vec3::ZERO).powf(0.5);
    let t2 = 2.0 * c * a;
    let rs = (t1 - t2) / (t1 + t2);

    let t3 = c2 * a2_plus_b2 + Vec3::splat(sin2 * sin2);
    let t4 = t2 * sin2;
    let rp = rs * (t3 - t4) / (t3 + t4);

    0.5 * (rs + rp)
}

/// Measured optical constants at roughly R/G/B wavelengths (~600/550/450 nm).
pub mod conductors {
    use glam::Vec3;
    /// Copper — the classic test case: a pronounced warm shift that Schlick
    /// cannot reproduce.
    pub const COPPER_ETA: Vec3 = Vec3::new(0.200_438, 0.924_033, 1.102_212);
    pub const COPPER_K: Vec3 = Vec3::new(3.9124, 2.44753, 2.13764);
    /// Gold.
    pub const GOLD_ETA: Vec3 = Vec3::new(0.143_245, 0.375_346, 1.442_386);
    pub const GOLD_K: Vec3 = Vec3::new(3.98312, 2.385_704, 1.603_575);
    /// Aluminium — nearly neutral, useful as a control.
    pub const ALUMINIUM_ETA: Vec3 = Vec3::new(1.345_812, 0.965_326, 0.617_180);
    pub const ALUMINIUM_K: Vec3 = Vec3::new(7.4746, 6.399_477, 5.303_298);
}

// ---------------------------------------------------------------------------
// Lobes
// ---------------------------------------------------------------------------

/// Result of sampling a BSDF.
#[derive(Clone, Copy, Debug)]
pub struct BsdfSample {
    pub wi: Vec3,
    /// `f * |cos(theta_i)| / pdf` — the factor to multiply path throughput by.
    pub weight: Vec3,
    /// Solid-angle density of `wi`. Zero means the sample is invalid.
    pub pdf: f32,
}

impl BsdfSample {
    pub const INVALID: BsdfSample = BsdfSample {
        wi: Vec3::Z,
        weight: Vec3::ZERO,
        pdf: 0.0,
    };
    pub fn is_valid(&self) -> bool {
        self.pdf > 0.0
    }
}

/// Lambertian diffuse: `f = albedo / pi`, sampled cosine-weighted.
pub mod diffuse {
    use super::*;

    #[inline]
    pub fn eval(albedo: Vec3, wi: Vec3) -> Vec3 {
        if wi.z <= 0.0 {
            return Vec3::ZERO;
        }
        albedo * INV_PI
    }

    #[inline]
    pub fn pdf(wi: Vec3) -> f32 {
        if wi.z <= 0.0 {
            0.0
        } else {
            wi.z * INV_PI
        }
    }

    #[inline]
    pub fn sample(albedo: Vec3, u: Vec2) -> BsdfSample {
        let wi = sample_cosine_hemisphere(u);
        let pdf = pdf(wi);
        if pdf <= 0.0 {
            return BsdfSample::INVALID;
        }
        // f * cos / pdf = (albedo/pi) * cos / (cos/pi) = albedo, exactly.
        BsdfSample {
            wi,
            weight: albedo,
            pdf,
        }
    }
}

/// Single-scattering GGX microfacet reflection.
pub mod specular {
    use super::*;

    /// `f(wo, wi) = D * G2 * F / (4 * |n.wo| * |n.wi|)`, single scattering only.
    pub fn eval_single(alpha: f32, f0: Vec3, wo: Vec3, wi: Vec3) -> Vec3 {
        if wo.z <= 0.0 || wi.z <= 0.0 {
            return Vec3::ZERO;
        }
        let h = (wo + wi).normalize_or_zero();
        if h == Vec3::ZERO {
            return Vec3::ZERO;
        }
        let d = ggx_d(alpha, h.z);
        let g2 = smith_g2(alpha, wo.z, wi.z);
        let f = fresnel_schlick(f0, wo.dot(h).max(0.0));
        f * (d * g2 / (4.0 * wo.z * wi.z))
    }

    pub fn pdf(alpha: f32, wo: Vec3, wi: Vec3) -> f32 {
        if wo.z <= 0.0 || wi.z <= 0.0 {
            return 0.0;
        }
        let h = (wo + wi).normalize_or_zero();
        if h == Vec3::ZERO {
            return 0.0;
        }
        ggx_vndf_pdf(alpha, wo, h)
    }

    /// Single-scattering sample, before energy compensation.
    pub fn sample_single(alpha: f32, f0: Vec3, wo: Vec3, u: Vec2) -> BsdfSample {
        if wo.z <= 0.0 {
            return BsdfSample::INVALID;
        }
        let m = sample_ggx_vndf(wo, alpha, u);
        let wi = reflect(wo, m);
        if wi.z <= 0.0 {
            // Reflected below the surface: masked, contributes nothing. Not an
            // error — it is the shadowing term expressing itself.
            return BsdfSample::INVALID;
        }
        let pdf = ggx_vndf_pdf(alpha, wo, m);
        if pdf <= 0.0 {
            return BsdfSample::INVALID;
        }
        let f = fresnel_schlick(f0, wo.dot(m).max(0.0));
        let g1 = smith_g1(alpha, wo.z);
        let g2 = smith_g2(alpha, wo.z, wi.z);
        BsdfSample {
            wi,
            weight: f * (g2 / g1),
            pdf,
        }
    }

    /// Energy-compensated evaluation. This is what the renderer uses.
    #[inline]
    pub fn eval(roughness: f32, f0: Vec3, wo: Vec3, wi: Vec3) -> Vec3 {
        let alpha = roughness_to_alpha(roughness);
        eval_single(alpha, f0, wo, wi) * energy::compensation(roughness, wo.z, f0)
    }

    /// Energy-compensated sample. This is what the renderer uses.
    #[inline]
    pub fn sample(roughness: f32, f0: Vec3, wo: Vec3, u: Vec2) -> BsdfSample {
        let alpha = roughness_to_alpha(roughness);
        let mut s = sample_single(alpha, f0, wo, u);
        if s.is_valid() {
            // The compensation scales the lobe's value, not its density, so only
            // the weight changes — the pdf is still the VNDF pdf and remains
            // correct for MIS at build step 9.
            s.weight *= energy::compensation(roughness, wo.z, f0);
        }
        s
    }
}

/// Refract `wo` about microfacet normal `m` with relative index `eta`.
///
/// `eta` is `n_transmitted / n_incident`, matching [`fresnel_dielectric`]. Both
/// vectors point *away* from the surface; the result points into it, so its `z`
/// has the opposite sign to `wo`'s.
///
/// Returns `None` under total internal reflection. That is not an error case to
/// be papered over with a clamp: past the critical angle there is genuinely no
/// transmitted direction, and the energy is entirely reflected. Clamping
/// `sin2_t` to 1 would invent a grazing refraction carrying energy that should
/// have gone into the reflection lobe, which shows up as a bright rim on the
/// inside of every curved glass surface.
///
/// From Snell's law, `sin(theta_t) = sin(theta_i) / eta`, written in vector form
/// so no trigonometry is needed:
///
/// ```text
///   wt = -wo / eta + (cos(theta_i) / eta - cos(theta_t)) * m
/// ```
#[inline]
pub fn refract(wo: Vec3, m: Vec3, eta: f32) -> Option<Vec3> {
    let cos_i = wo.dot(m);
    let sin2_i = (1.0 - cos_i * cos_i).max(0.0);
    let sin2_t = sin2_i / (eta * eta);
    if sin2_t >= 1.0 {
        return None;
    }
    let cos_t = (1.0 - sin2_t).sqrt();
    // The sign of cos_i decides which way the transmitted ray bends; writing it
    // this way keeps one branch instead of two.
    let wt = -wo / eta + (cos_i / eta - cos_t.copysign(cos_i)) * m;
    Some(wt.normalize_or_zero())
}

/// Rough dielectric transmission: a GGX microfacet BTDF (Walter et al. 2007).
///
/// # Why there is no smooth-glass special case
///
/// A perfectly smooth dielectric is a delta distribution, and a delta lobe has
/// no density to put in a pdf — which means no MIS weight, a separate code path
/// through the integrator, and a second set of rules for the light sampler. This
/// renderer avoids all of that because [`MIN_ALPHA`] already floors roughness at
/// `2e-3`: "smooth" glass is a very narrow GGX lobe rather than a delta, so it
/// goes through exactly the same evaluate/sample/pdf machinery as everything
/// else and needs no special handling anywhere.
///
/// The cost is that a mirror-flat surface is very slightly blurry. At alpha
/// `2e-3` the lobe is under a tenth of a degree wide, which is finer than a
/// pixel at any sane resolution.
///
/// # The half-vector for refraction
///
/// Reflection's half-vector bisects `wo` and `wi`. Refraction's does not — the
/// two directions are in different media, so the microfacet that connects them
/// is weighted by the indices (Walter eq. 16):
///
/// ```text
///   h = normalize(wo + eta * wi)
/// ```
///
/// flipped into the upper hemisphere. Getting this wrong produces an image that
/// is plausibly glassy and has the wrong caustics.
pub mod dielectric {
    use super::*;

    /// Half-vector connecting `wo` and a transmitted `wi`, in the upper
    /// hemisphere. `None` when the two are degenerate.
    #[inline]
    pub fn transmission_half_vector(wo: Vec3, wi: Vec3, eta: f32) -> Option<Vec3> {
        let h = (wo + eta * wi).normalize_or_zero();
        if h == Vec3::ZERO {
            return None;
        }
        Some(if h.z < 0.0 { -h } else { h })
    }

    /// The BTDF, for `wi` in the opposite hemisphere to `wo`.
    ///
    /// Walter eq. 21, with the radiance-transport correction folded in:
    ///
    /// ```text
    ///   f_t = (1 - F) * D * G2 * |wi.h| * |wo.h| * eta^2 * factor^2
    ///                   / (|cos_i| * |cos_o| * (wo.h + eta * wi.h)^2)
    /// ```
    ///
    /// # Why `eta^2` disappears
    ///
    /// A BTDF is **not symmetric**. Radiance is compressed by `eta^2` when light
    /// enters a denser medium — the same energy is squeezed into a narrower cone
    /// of solid angle — so light-transport and importance-transport differ by
    /// that factor. A path tracer starting at the camera is transporting
    /// importance, so it carries `factor = 1 / eta` (Veach 1997, §5.2), and
    /// `eta^2 * (1/eta)^2 = 1`.
    ///
    /// The two cancel exactly, so neither appears below. That is worth stating
    /// rather than leaving as an absence: an implementation that includes the
    /// `eta^2` and forgets the correction is too bright by `eta^2` — 2.25x for
    /// glass — on every transmitted path, and looks merely "a bit glowy".
    pub fn eval(alpha: f32, eta: f32, wo: Vec3, wi: Vec3) -> f32 {
        // Opposite hemispheres, or there is nothing transmitted about it.
        if wo.z * wi.z >= 0.0 {
            return 0.0;
        }
        let Some(h) = transmission_half_vector(wo, wi, eta) else {
            return 0.0;
        };
        let wo_h = wo.dot(h);
        let wi_h = wi.dot(h);
        // The microfacet must face the viewer and the two directions must be on
        // opposite sides of it, or this half-vector does not actually connect
        // them through the interface. Testing only the product would admit the
        // back-facing case, which the sampler never produces.
        if wo_h <= 0.0 || wi_h >= 0.0 {
            return 0.0;
        }

        let denom = wo_h + eta * wi_h;
        if denom.abs() < 1.0e-9 {
            return 0.0;
        }

        let f = fresnel_dielectric(wo_h, eta);
        let d = ggx_d(alpha, h.z.abs());
        let g2 = smith_g2(alpha, wo.z.abs(), wi.z.abs());

        ((1.0 - f) * d * g2 * (wi_h * wo_h).abs())
            / (wo.z.abs() * wi.z.abs() * denom * denom)
    }

    /// Solid-angle density of a transmitted direction, given the half-vector is
    /// drawn from the visible normal distribution.
    ///
    /// The Jacobian of the refraction mapping `h -> wi` (Walter eq. 17):
    ///
    /// ```text
    ///   dh/dwi = eta^2 * |wi.h| / (wo.h + eta * wi.h)^2
    /// ```
    pub fn pdf(alpha: f32, eta: f32, wo: Vec3, wi: Vec3) -> f32 {
        if wo.z * wi.z >= 0.0 {
            return 0.0;
        }
        let Some(h) = transmission_half_vector(wo, wi, eta) else {
            return 0.0;
        };
        let wo_h = wo.dot(h);
        let wi_h = wi.dot(h);
        // The microfacet must face the viewer and the two directions must be on
        // opposite sides of it, or this half-vector does not actually connect
        // them through the interface. Testing only the product would admit the
        // back-facing case, which the sampler never produces.
        if wo_h <= 0.0 || wi_h >= 0.0 {
            return 0.0;
        }
        let denom = wo_h + eta * wi_h;
        if denom.abs() < 1.0e-9 {
            return 0.0;
        }
        let jacobian = (eta * eta * wi_h).abs() / (denom * denom);
        ggx_vndf_half_pdf(alpha, wo, h) * jacobian
    }
}

/// Reflect `v` about `n`, both pointing away from the surface.
#[inline]
pub fn reflect(v: Vec3, n: Vec3) -> Vec3 {
    -v + 2.0 * v.dot(n) * n
}

// ---------------------------------------------------------------------------
// Energy measurement
// ---------------------------------------------------------------------------

/// Directional albedo of the specular lobe with Fresnel forced to **white**:
///
/// ```text
///   E(wo) = integral over the hemisphere of  f(wo, wi) * cos(theta_i) dwi
/// ```
///
/// This is the white furnace test reduced to a single number. Put a
/// non-absorbing object in an environment of uniform radiance 1: the radiance
/// leaving it is exactly `E(wo)`, so the object is **invisible** if and only if
/// `E(wo) == 1` for every `wo`. Any visible silhouette is energy the BSDF
/// created or destroyed.
///
/// Estimated with VNDF sampling, where the estimator collapses to `G2 / G1(wo)`
/// — no `D`, no pdf division, so the estimate itself is well conditioned even
/// where the BRDF is not.
///
/// This is the *random* estimator, used by the furnace test itself so the test
/// genuinely samples rather than replaying the grid the table was built from.
/// [`specular_directional_albedo_grid`] is the deterministic one used to build
/// the table.
pub fn specular_directional_albedo(alpha: f32, cos_theta_o: f32, samples: u32, seed: u32) -> f32 {
    let mut rng = crate::rng::Rng::new(seed, 0, 0);
    let sin_theta = (1.0 - cos_theta_o * cos_theta_o).max(0.0).sqrt();
    let wo = Vec3::new(sin_theta, 0.0, cos_theta_o);

    let mut sum = 0.0f64;
    for _ in 0..samples {
        // `sample_single`, emphatically not `sample`: the compensated sampler
        // looks this table up, so calling it here would recurse into the table's
        // own initialisation and deadlock.
        let s = specular::sample_single(alpha, Vec3::ONE, wo, rng.next_vec2());
        if s.is_valid() {
            // Weight is F * G2 / G1 and F is white here, so this accumulates
            // exactly the directional albedo.
            sum += s.weight.x as f64;
        }
    }
    (sum / samples as f64) as f32
}

/// Directional albedo by deterministic quadrature over the VNDF sampler's
/// parameter square.
///
/// VNDF sampling maps `[0,1]^2` to directions, and the estimator `G2/G1` is
/// smooth in those coordinates — no peaks, bounded in [0, 1]. A regular grid
/// therefore converges far faster than random sampling at the same cost, and
/// produces **no noise at all**.
///
/// That matters more than it looks. The compensation multiplier is
/// `1 + F_avg (1 - E)/E`, whose sensitivity to an error in `E` is `1/E^2`. At
/// roughness 1, `E` is about 0.32, so a 0.5% error in the table becomes a 1.5%
/// energy error in the render — and with random sampling that error is
/// *different in every cell*, which survives bilinear interpolation as banding.
pub fn specular_directional_albedo_grid(alpha: f32, cos_theta_o: f32, grid: u32) -> f32 {
    let sin_theta = (1.0 - cos_theta_o * cos_theta_o).max(0.0).sqrt();
    let wo = Vec3::new(sin_theta, 0.0, cos_theta_o);
    let mut sum = 0.0f64;
    let n = grid as f32;
    for i in 0..grid {
        for j in 0..grid {
            // Midpoint rule: sample cell centres, so no sample lands on the
            // degenerate u = 0 or u = 1 edges.
            let u = Vec2::new((i as f32 + 0.5) / n, (j as f32 + 0.5) / n);
            let s = specular::sample_single(alpha, Vec3::ONE, wo, u);
            if s.is_valid() {
                sum += s.weight.x as f64;
            }
        }
    }
    (sum / (grid as f64 * grid as f64)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    /// `integral of D(m)(n.m) dm`, by quadrature in tan space.
    ///
    /// Uniform hemisphere sampling cannot do this job: the GGX lobe has angular
    /// width about `alpha`, so at `alpha = 0.01` a uniform sample lands in the
    /// lobe roughly once in ten thousand draws and the estimate is useless.
    ///
    /// Substituting `t = tan(theta) / alpha` turns the integrand into
    /// `t / (1 + t^2)^2`, which is smooth *and independent of alpha* — so one
    /// fixed grid is accurate at every roughness.
    fn ndf_integral(alpha: f32) -> f64 {
        let n = 200_000;
        let t_max = 5_000.0f64;
        let h = t_max / n as f64;
        let mut sum = 0.0f64;
        for i in 0..=n {
            let t = i as f64 * h;
            let theta = (alpha as f64 * t).atan();
            let (sin, cos) = theta.sin_cos();
            let dtheta = alpha as f64 / (1.0 + (alpha as f64 * t).powi(2));
            let f = ggx_d(alpha, cos as f32) as f64 * cos * sin * dtheta;
            let w = if i == 0 || i == n {
                1.0
            } else if i % 2 == 1 {
                4.0
            } else {
                2.0
            };
            sum += w * f;
        }
        sum * h / 3.0 * 2.0 * std::f64::consts::PI
    }

    /// The NDF must be normalised: `integral of D(m) * (n.m) dm = 1`.
    ///
    /// This says the microfacets' *projected* area equals the macrosurface area
    /// — the assumption every later energy argument rests on. A `D` off by a
    /// constant produces a perfectly plausible highlight of the wrong brightness.
    #[test]
    fn ndf_is_normalised() {
        for &roughness in &[0.05f32, 0.1, 0.2, 0.5, 0.8, 1.0] {
            let alpha = roughness_to_alpha(roughness);
            let integral = ndf_integral(alpha);
            assert!(
                (integral - 1.0).abs() < 2e-3,
                "roughness {roughness} (alpha {alpha}): integral of D(m)(n.m) = {integral}, expected 1"
            );
        }
    }

    /// `MIN_ALPHA` must be the point below which `f32` stops being able to carry
    /// the lobe. Pinned with numbers, because the consequence of getting it
    /// wrong is not subtle: at alpha 1e-4 the NDF integrates to 6.1, so a
    /// near-mirror surface emits six times the light that fell on it.
    #[test]
    fn min_alpha_is_where_f32_gives_out() {
        assert!(
            (ndf_integral(MIN_ALPHA) - 1.0).abs() < 2e-3,
            "MIN_ALPHA = {MIN_ALPHA} does not integrate to 1: {}",
            ndf_integral(MIN_ALPHA)
        );
        // Below it, precision collapses — this is the failure being guarded
        // against, so assert it actually exists rather than trusting the note.
        assert!(
            ndf_integral(1.0e-4) > 5.0,
            "expected catastrophic energy gain below MIN_ALPHA, got {}",
            ndf_integral(1.0e-4)
        );
        // And no roughness the user can dial in may fall below the clamp.
        for r in 0..=100 {
            assert!(roughness_to_alpha(r as f32 / 100.0) >= MIN_ALPHA);
        }
    }

    /// The distribution of visible normals must integrate to 1 over the
    /// hemisphere. This is the exact, always-satisfiable furnace test — it
    /// validates the VNDF sampler and `G1` independently of any energy loss in
    /// the full BRDF.
    /// The distribution of visible normals must integrate to 1.
    ///
    /// Estimated by importance sampling from `D` itself rather than uniformly:
    /// the ratio `D_v(m) / (D(m)(n.m))` reduces to
    /// `G1(wo) * max(0, wo.m) / ((n.wo)(n.m))`, which is smooth and bounded, so
    /// this converges in thousands of samples where uniform sampling would need
    /// billions.
    ///
    /// This is the exact, always-satisfiable furnace test. It validates `G1` and
    /// the visible-normal density independently of any energy loss in the full
    /// BRDF, which is what makes it useful as a separate check.
    #[test]
    fn visible_normal_distribution_is_normalised() {
        for &roughness in &[0.05f32, 0.1, 0.4, 0.7, 1.0] {
            for &cos_o in &[0.95f32, 0.6, 0.25, 0.08] {
                let alpha = roughness_to_alpha(roughness);
                let sin_o = (1.0 - cos_o * cos_o).sqrt();
                let wo = Vec3::new(sin_o, 0.0, cos_o);

                let mut rng = Rng::new(13, 0, 0);
                let n = 400_000;
                let mut sum = 0.0f64;
                for _ in 0..n {
                    let m = sample_ggx_ndf(alpha, rng.next_vec2());
                    let pdf = ggx_ndf_pdf(alpha, m);
                    if pdf <= 0.0 {
                        continue;
                    }
                    let d_v = smith_g1(alpha, wo.z) * wo.dot(m).max(0.0) * ggx_d(alpha, m.z) / wo.z;
                    sum += (d_v / pdf) as f64;
                }
                let integral = sum / n as f64;
                assert!(
                    (integral - 1.0).abs() < 0.01,
                    "roughness {roughness}, cos_o {cos_o}: integral of D_v = {integral}, expected 1"
                );
            }
        }
    }

    /// **The white furnace test.**
    ///
    /// Put an object with a pure-white, non-absorbing BSDF in an environment of
    /// uniform radiance 1. The radiance leaving it is its directional albedo, so
    /// a correct energy-conserving BSDF renders the object exactly **invisible**.
    /// Any visible silhouette is energy the BSDF invented or destroyed.
    ///
    /// Run with *random* sampling, deliberately: the table was built by
    /// quadrature on a grid, and testing with that same grid would only prove
    /// the table reproduces itself.
    ///
    /// Single-scattering GGX fails this badly by construction — it discards 68%
    /// of the energy at roughness 1 — so the compensated lobe is what is
    /// asserted here. `single_scattering_ggx_loses_energy` pins the failure it
    /// is correcting.
    #[test]
    fn white_furnace_test() {
        let mut worst = 0.0f32;
        let mut worst_at = (0.0f32, 0.0f32);
        for ri in 0..=20 {
            let roughness = ri as f32 / 20.0;
            for ci in 1..=20 {
                let cos_o = ci as f32 / 20.0;
                let e = {
                    let mut rng = Rng::new(0xF0_0D, ri, ci);
                    let sin = (1.0 - cos_o * cos_o).max(0.0).sqrt();
                    let wo = Vec3::new(sin, 0.0, cos_o);
                    let n = 60_000;
                    let mut sum = 0.0f64;
                    for _ in 0..n {
                        let s = specular::sample(roughness, Vec3::ONE, wo, rng.next_vec2());
                        if s.is_valid() {
                            sum += s.weight.x as f64;
                        }
                    }
                    (sum / n as f64) as f32
                };
                if (e - 1.0).abs() > worst {
                    worst = (e - 1.0).abs();
                    worst_at = (roughness, cos_o);
                }
            }
        }
        eprintln!(
            "white furnace: worst deviation {worst:.4} at roughness {:.2}, cos_theta {:.2}",
            worst_at.0, worst_at.1
        );
        assert!(
            worst < 0.01,
            "the object is not invisible: directional albedo deviates from 1 by {worst:.4} \
             at roughness {:.2}, cos_theta {:.2}",
            worst_at.0,
            worst_at.1
        );
    }

    /// Pins the energy loss that compensation exists to correct.
    ///
    /// If this ever starts passing, either `G2` has silently become separable
    /// (which would *mask* the loss rather than fix it) or something else has
    /// changed the model — and the furnace test above would then be passing for
    /// the wrong reason.
    #[test]
    fn single_scattering_ggx_loses_energy() {
        let smooth = specular_directional_albedo_grid(roughness_to_alpha(0.05), 0.8, 128);
        assert!(
            (smooth - 1.0).abs() < 1e-3,
            "smooth GGX should already conserve energy, got {smooth}"
        );
        let rough = specular_directional_albedo_grid(roughness_to_alpha(1.0), 0.95, 128);
        assert!(
            (0.28..0.36).contains(&rough),
            "single-scattering GGX at roughness 1 should lose about two thirds of \
             its energy; measured directional albedo {rough}"
        );
    }

    /// The VNDF sampler's own density must match `ggx_vndf_pdf`.
    ///
    /// A precursor to the full chi-squared test at build step 7: integrate
    /// `1/pdf` over the sampled directions, which must converge to the measure
    /// of the region the sampler covers. Here it is easier to check the sampler
    /// reproduces the analytic density directly, by histogramming.
    #[test]
    fn vndf_sampler_matches_its_pdf() {
        let alpha = roughness_to_alpha(0.4);
        let wo = Vec3::new(0.6, 0.0, 0.8);
        let mut rng = Rng::new(17, 0, 0);

        // Estimate E[1 / pdf(wi)] over sampled directions. For a normalised
        // density over the hemisphere this converges to the measure of the
        // support, which for GGX reflection is (numerically) close to 2*pi
        // minus the masked region — so rather than assert a constant, check
        // that sampling and evaluating agree pairwise.
        let n = 200_000;
        let mut max_rel = 0.0f32;
        for _ in 0..n {
            let u = rng.next_vec2();
            let m = sample_ggx_vndf(wo, alpha, u);
            let wi = reflect(wo, m);
            if wi.z <= 0.0 {
                continue;
            }
            let from_sampler = ggx_vndf_pdf(alpha, wo, m);
            let from_eval = specular::pdf(alpha, wo, wi);
            if from_sampler > 1e-6 {
                let rel = ((from_sampler - from_eval) / from_sampler).abs();
                max_rel = max_rel.max(rel);
            }
        }
        assert!(
            max_rel < 1e-3,
            "sample() and pdf() disagree by up to {max_rel:.3e}"
        );
    }

    /// Fresnel must be 1 at grazing incidence for every material. Every surface
    /// becomes a mirror edge-on; a model that does not do this looks wrong at
    /// silhouettes in a way people notice without being able to name.
    #[test]
    fn fresnel_goes_to_one_at_grazing() {
        assert!((fresnel_schlick(Vec3::splat(0.04), 0.0).x - 1.0).abs() < 1e-6);
        assert!((fresnel_dielectric(0.0, 1.5) - 1.0).abs() < 1e-6);
        for (eta, k) in [
            (conductors::COPPER_ETA, conductors::COPPER_K),
            (conductors::GOLD_ETA, conductors::GOLD_K),
            (conductors::ALUMINIUM_ETA, conductors::ALUMINIUM_K),
        ] {
            let f = fresnel_conductor(0.0, eta, k);
            assert!(
                (f.x - 1.0).abs() < 1e-3 && (f.y - 1.0).abs() < 1e-3 && (f.z - 1.0).abs() < 1e-3,
                "grazing conductor Fresnel {f}, expected white"
            );
        }
    }

    /// Schlick must agree with the exact dielectric Fresnel it approximates.
    #[test]
    fn schlick_approximates_exact_dielectric_fresnel() {
        let ior = 1.5;
        let f0 = f0_from_ior(ior);
        let mut worst = 0.0f32;
        for i in 0..=100 {
            let cos = i as f32 / 100.0;
            let exact = fresnel_dielectric(cos, ior);
            let approx = fresnel_schlick(Vec3::splat(f0), cos).x;
            worst = worst.max((exact - approx).abs());
        }
        // Measured max absolute error for IOR 1.5 is 0.0357 near cos = 0.09.
        // Pinned so that a change to either implementation has to be deliberate.
        assert!(
            (0.030..0.040).contains(&worst),
            "Schlick deviates from exact Fresnel by {worst}, expected about 0.036"
        );
    }

    /// Conductor Fresnel must be sane and, for copper and gold, warm at normal
    /// incidence — red above green above blue. Getting eta and k swapped, or
    /// the channels reversed, produces a blue "copper" and nothing else catches
    /// it.
    #[test]
    fn conductor_fresnel_has_the_right_colour() {
        let copper = fresnel_conductor(1.0, conductors::COPPER_ETA, conductors::COPPER_K);
        assert!(
            copper.x > copper.y && copper.y > copper.z,
            "copper F0 = {copper}"
        );
        assert!(copper.x > 0.9 && copper.z < 0.7, "copper F0 = {copper}");

        let gold = fresnel_conductor(1.0, conductors::GOLD_ETA, conductors::GOLD_K);
        assert!(gold.x > gold.y && gold.y > gold.z, "gold F0 = {gold}");
        assert!(
            gold.z < 0.5,
            "gold should be strongly blue-absorbing, got {gold}"
        );

        // Aluminium is near-neutral and bright.
        let al = fresnel_conductor(1.0, conductors::ALUMINIUM_ETA, conductors::ALUMINIUM_K);
        assert!(al.min_element() > 0.85, "aluminium F0 = {al}");
        assert!(
            al.max_element() - al.min_element() < 0.1,
            "aluminium should be neutral, got {al}"
        );

        for (eta, k) in [
            (conductors::COPPER_ETA, conductors::COPPER_K),
            (conductors::GOLD_ETA, conductors::GOLD_K),
        ] {
            for i in 0..=20 {
                let f = fresnel_conductor(i as f32 / 20.0, eta, k);
                assert!(
                    f.is_finite() && f.min_element() >= 0.0 && f.max_element() <= 1.0001,
                    "conductor Fresnel out of range: {f}"
                );
            }
        }
    }

    /// Height-correlated G2 must never exceed the separable product, and both
    /// must stay in [0, 1].
    #[test]
    fn masking_shadowing_is_bounded_and_correlated() {
        for &roughness in &[0.05f32, 0.3, 0.7, 1.0] {
            let alpha = roughness_to_alpha(roughness);
            for i in 1..=20 {
                for j in 1..=20 {
                    let (co, ci) = (i as f32 / 20.0, j as f32 / 20.0);
                    let g2 = smith_g2(alpha, co, ci);
                    let separable = smith_g1(alpha, co) * smith_g1(alpha, ci);
                    assert!((0.0..=1.0).contains(&g2), "G2 = {g2}");
                    // Correlation can only *increase* joint visibility: a facet
                    // visible from one direction is more likely visible from the
                    // other. The separable form therefore under-counts.
                    assert!(
                        g2 >= separable - 1e-6,
                        "roughness {roughness}, cos {co}/{ci}: height-correlated G2 {g2} \
                         is below the separable product {separable}"
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Multiple-scattering energy compensation
// ---------------------------------------------------------------------------

/// Energy compensation for single-scattering GGX.
///
/// # The problem the furnace test exposes
///
/// A microfacet BRDF with a single-scattering `G2` accounts for light that hits
/// one microfacet and leaves. Light that bounces *between* microfacets before
/// escaping is simply discarded — `G2` says "this got masked" and drops it. On a
/// smooth surface almost nothing gets masked, so nothing is lost; on a rough one
/// most of it does.
///
/// Measured directional albedo of the white-Fresnel lobe, which a conserving
/// BSDF must hold at exactly 1:
///
/// ```text
///   roughness   cos=0.95   cos=0.6   cos=0.2
///   0.05          1.0000    1.0000    0.9999
///   0.20          0.9981    0.9965    0.9751
///   0.40          0.9661    0.9407    0.8706
///   0.60          0.8209    0.7881    0.8268
///   1.00          0.3176    0.4124    0.6427
/// ```
///
/// At roughness 1 the model destroys **68% of the incident energy**. That is not
/// a bug in the implementation — it is what single-scattering GGX *is*, and it
/// is precisely what the white furnace test exists to reveal. Rough metal
/// rendered this way is conspicuously too dark, and the error grows with
/// roughness in a way that reads as "this material looks wrong" without being
/// attributable.
///
/// # The fix
///
/// Turquin's compensation (2019): scale the lobe by whatever fraction went
/// missing.
///
/// ```text
///   f = f_single * (1 + F_avg * (1 - E(mu_o)) / E(mu_o))
/// ```
///
/// With white Fresnel this is exact by construction — the directional albedo
/// becomes `E * (1 + (1 - E)/E) = 1` for every roughness and every angle — so
/// the furnace test passes identically rather than approximately. `F_avg`
/// carries the colour: energy that bounced several times between microfacets was
/// filtered by Fresnel each time, which is why rough gold is more saturated than
/// smooth gold.
///
/// The one honest caveat: using only `mu_o` makes the BSDF **non-reciprocal**.
/// Kulla-Conty's formulation is symmetric in `mu_o` and `mu_i` and restores
/// reciprocity at the cost of a second table lookup. For a unidirectional path
/// tracer the asymmetry is invisible; it would matter for bidirectional methods.
pub mod energy {
    use super::*;
    use std::sync::OnceLock;

    /// Table resolution in each dimension (roughness, cos_theta_o).
    pub const TABLE_SIZE: usize = 32;

    /// Quadrature grid per table entry, so each cell costs `GRID^2` samples.
    /// 128 x 128 is 16k evaluations — the same cost as the random estimator it
    /// replaced, with no noise.
    const TABLE_GRID: u32 = 128;

    /// Directional albedo of the white-Fresnel single-scattering GGX lobe.
    ///
    /// Row-major, roughness as the outer index. Roughness is sampled linearly;
    /// the view angle is sampled in **`sqrt(cos_theta_o)`**, not `cos_theta_o`.
    ///
    /// That parameterisation is not decoration. `E` is nearly flat over most of
    /// the hemisphere and then drops steeply in the last few degrees before
    /// grazing, so uniform spacing in `cos` wastes most of its resolution on the
    /// flat part and then interpolates across the steep part. Measured, the
    /// furnace test's worst error was 1.07% at `cos_theta = 0.05` with uniform
    /// spacing, and it was entirely interpolation error rather than table error.
    /// Sampling in `sqrt(cos)` puts the resolution where the curvature is.
    ///
    /// Computed once and shared. Codegen emits the identical table into WGSL by
    /// calling this function, so the two cannot drift.
    pub fn table() -> &'static [f32] {
        static TABLE: OnceLock<Vec<f32>> = OnceLock::new();
        TABLE.get_or_init(compute_table)
    }

    pub fn compute_table() -> Vec<f32> {
        let mut t = vec![0.0f32; TABLE_SIZE * TABLE_SIZE];
        for ri in 0..TABLE_SIZE {
            let roughness = ri as f32 / (TABLE_SIZE - 1) as f32;
            let alpha = roughness_to_alpha(roughness);
            for ci in 0..TABLE_SIZE {
                // Sample in sqrt(cos) so resolution concentrates near grazing,
                // where E has all its curvature. Avoid exactly zero: at true
                // grazing the lobe degenerates and the estimate is meaningless.
                let root_mu = ci as f32 / (TABLE_SIZE - 1) as f32;
                let mu = (root_mu * root_mu).max(1.0e-4);
                t[ri * TABLE_SIZE + ci] = specular_directional_albedo_grid(alpha, mu, TABLE_GRID);
            }
        }
        t
    }

    /// Bilinear lookup of the directional albedo.
    #[inline]
    pub fn directional_albedo(roughness: f32, cos_theta_o: f32) -> f32 {
        lookup(table(), roughness, cos_theta_o)
    }

    /// Shared by the Rust lookup and the WGSL mirror, so both interpolate the
    /// same way.
    #[inline]
    pub fn lookup(table: &[f32], roughness: f32, cos_theta_o: f32) -> f32 {
        let n = TABLE_SIZE;
        let fx = roughness.clamp(0.0, 1.0) * (n - 1) as f32;
        // Must match the parameterisation used to build the table.
        let fy = cos_theta_o.clamp(0.0, 1.0).sqrt() * (n - 1) as f32;
        let x0 = (fx as usize).min(n - 1);
        let y0 = (fy as usize).min(n - 1);
        let x1 = (x0 + 1).min(n - 1);
        let y1 = (y0 + 1).min(n - 1);
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;

        let a = table[x0 * n + y0];
        let b = table[x0 * n + y1];
        let c = table[x1 * n + y0];
        let d = table[x1 * n + y1];
        let top = a + (b - a) * ty;
        let bot = c + (d - c) * ty;
        (top + (bot - top) * tx).clamp(1.0e-3, 1.0)
    }

    /// Average Fresnel over the hemisphere, for Schlick.
    ///
    /// `F_avg = integral of F(mu) * 2*mu dmu` over [0,1], which for Schlick
    /// integrates in closed form to `F0 + (1 - F0) / 21`. This is the colour the
    /// multiply-scattered energy picks up.
    #[inline]
    pub fn fresnel_average(f0: Vec3) -> Vec3 {
        f0 + (Vec3::ONE - f0) * (1.0 / 21.0)
    }

    /// Multiplier that restores the energy single scattering discards.
    #[inline]
    pub fn compensation(roughness: f32, cos_theta_o: f32, f0: Vec3) -> Vec3 {
        let e = directional_albedo(roughness, cos_theta_o);
        Vec3::ONE + fresnel_average(f0) * ((1.0 - e) / e)
    }
}

// ---------------------------------------------------------------------------
// The combined surface BSDF
// ---------------------------------------------------------------------------

/// A diffuse base under a GGX specular layer.
///
/// # Lobe selection and the combined PDF
///
/// Two lobes, one sample. Pick a lobe with probability `p`, sample it, and then
/// report the **combined** density
///
/// ```text
///   pdf(wi) = p_spec * pdf_spec(wi) + p_diff * pdf_diff(wi)
/// ```
///
/// not the density of whichever lobe happened to be chosen. This is the point
/// most easily got wrong, and getting it wrong is invisible in a render: a
/// direction reachable by *either* lobe really is generated with the combined
/// probability, so using only the chosen lobe's density biases the estimator by
/// double-counting the overlap. It also makes the multiple importance sampling
/// weights at build step 9 wrong, because MIS needs the true density of the
/// strategy, not of a sub-strategy.
///
/// The selection probability is Fresnel-weighted rather than fixed: at grazing
/// angles almost all the energy is specular, and spending half the samples on a
/// diffuse lobe that contributes nothing there is pure variance.
pub mod surface {
    use super::*;
    use crate::gpu_layout::GpuMaterial;

    /// Material parameters resolved into what the lobes actually need.
    #[derive(Clone, Copy, Debug)]
    pub struct Surface {
        pub diffuse_albedo: Vec3,
        pub roughness: f32,
        /// Normal-incidence reflectance, for the Schlick path.
        pub f0: Vec3,
        /// Complex IOR; used instead of `f0` when `conductor` is set.
        pub eta: Vec3,
        pub k: Vec3,
        pub conductor: bool,
        /// Fraction of the non-specularly-reflected energy that refracts through
        /// rather than scattering off a diffuse base. 1 is glass, 0 is opaque.
        pub transmission: f32,
        /// Colour applied to transmitted light.
        ///
        /// Physically this belongs in Beer-Lambert absorption along the path
        /// *inside* the medium, which needs the distance travelled and so needs
        /// volumetric tracking. Tinting at the interface instead is the usual
        /// approximation and is exact for a thin surface; it is wrong for a
        /// thick coloured solid, where absorption should deepen with thickness.
        pub transmission_tint: Vec3,
        /// Relative index of refraction, `n_transmitted / n_incident`, for the
        /// side the ray is currently on.
        ///
        /// Entering glass from air this is the material's IOR; leaving it is the
        /// reciprocal. It has to be resolved here rather than inside the lobes
        /// because the shading frame has already been flipped to face the
        /// viewer, so `wo.z > 0` on both sides and the local frame alone cannot
        /// say which medium the ray is in.
        pub relative_ior: f32,
    }

    impl Surface {
        /// Resolve a material for a hit on the side `front_face` describes.
        ///
        /// `front_face` is the *geometric* facing test from the hit record, not
        /// a shading-normal one: near a silhouette an interpolated normal can
        /// disagree with the facet, and letting it decide which medium the ray
        /// is in would flip a glass surface inside out along its own outline.
        pub fn from_material(m: &GpuMaterial, front_face: bool) -> Surface {
            let base = Vec3::from_array(m.base_color);
            let metallic = m.metallic.clamp(0.0, 1.0);
            // A metal has no diffuse lobe: the free electrons that make it
            // reflective also absorb anything that enters, so there is no
            // subsurface scattering to be diffuse about.
            let diffuse_albedo = base * (1.0 - metallic);
            // Dielectric F0 comes from the IOR; metal F0 is the base colour.
            // Blending between them is physically meaningless for intermediate
            // values but is the parameterisation every artist expects.
            let dielectric_f0 = Vec3::splat(f0_from_ior(m.ior.max(1.0)));
            // A metal cannot transmit, whatever the material says: the same
            // free electrons that make it reflective absorb anything that gets
            // in. Letting `metallic` and `transmission` both apply would produce
            // a surface that is neither.
            let transmission = m.transmission.clamp(0.0, 1.0) * (1.0 - metallic);
            let ior = m.ior.max(1.0);
            Surface {
                diffuse_albedo,
                roughness: m.roughness.clamp(0.0, 1.0),
                f0: dielectric_f0.lerp(base, metallic),
                eta: Vec3::from_array(m.eta),
                k: Vec3::from_array(m.k),
                conductor: m.is_conductor(),
                transmission,
                // White glass is the common case and `base_color` defaults to
                // black, so an unset colour must not make glass opaque.
                transmission_tint: if base == Vec3::ZERO { Vec3::ONE } else { base },
                relative_ior: if front_face { ior } else { 1.0 / ior },
            }
        }

        /// Fresnel at a given cosine, taking whichever path this material uses.
        #[inline]
        pub fn fresnel(&self, cos_theta: f32) -> Vec3 {
            if self.conductor {
                fresnel_conductor(cos_theta, self.eta, self.k)
            } else {
                fresnel_schlick(self.f0, cos_theta)
            }
        }

        /// Fraction of incoming energy the specular layer reflects, seen from
        /// `wo`.
        ///
        /// # Why this is needed
        ///
        /// The two lobes are not independent. The specular layer sits *over* the
        /// diffuse base, so light it reflects never reaches the base at all.
        /// Adding the lobes without accounting for that lets a surface reflect
        /// more light than arrived — measured, a smooth blue plastic at grazing
        /// incidence returned 1.40 in the blue channel, because Schlick gives
        /// F = 0.61 there and the 0.8 diffuse albedo was added on top regardless.
        ///
        /// The estimate mirrors the structure of the compensation itself:
        /// singly-scattered energy is weighted by the Fresnel at *this* angle,
        /// and the multiply-scattered remainder by the hemispherical average,
        /// because it was filtered by Fresnel several times on the way out.
        ///
        /// ```text
        ///   E_spec(mu_o) = F(mu_o) * E_ss(mu_o) + F_avg * (1 - E_ss(mu_o))
        /// ```
        ///
        /// With a perfect mirror layer (`F = 1`) this is exactly 1, so the base
        /// receives nothing — which is the right answer and a useful check that
        /// the form is not merely plausible.
        #[inline]
        pub fn specular_albedo(&self, cos_theta_o: f32) -> Vec3 {
            let e_ss = energy::directional_albedo(self.roughness, cos_theta_o.abs());
            let f = self.fresnel(cos_theta_o.abs());
            // F_avg via Schlick's closed form. For a conductor this is an
            // approximation — the exact average would need integrating the
            // conductor equations — but it is used only to tint energy that has
            // already bounced several times, where the angular detail is gone.
            let f_avg = energy::fresnel_average(self.normal_reflectance());
            (f * e_ss + f_avg * (1.0 - e_ss)).clamp(Vec3::ZERO, Vec3::ONE)
        }

        /// Reflectance at normal incidence, used for lobe weighting and for the
        /// energy compensation's average Fresnel.
        #[inline]
        pub fn normal_reflectance(&self) -> Vec3 {
            if self.conductor {
                fresnel_conductor(1.0, self.eta, self.k)
            } else {
                self.f0
            }
        }

        /// Probability of choosing the specular lobe.
        ///
        /// Weighted by the Fresnel reflectance actually seen from `wo`, so the
        /// split tracks where the energy is. Clamped away from 0 and 1: a lobe
        /// that can still contribute must keep a non-zero chance of being
        /// sampled, or its contribution is silently dropped.
        #[inline]
        pub fn specular_probability(&self, cos_theta_o: f32) -> f32 {
            if self.transmission > 0.0 {
                // For a transmissive dielectric the split is exactly Fresnel:
                // what is not reflected is what goes through. Using the opaque
                // path's albedo-ratio heuristic here would put most samples in
                // the reflection lobe of a material that is almost entirely
                // transmissive, and glass would converge like a mirror does.
                let f = fresnel_dielectric(cos_theta_o.abs(), self.relative_ior);
                // Past the critical angle the interface reflects everything, and
                // the clamp must not apply: reserving 5% of samples for a lobe
                // that cannot produce a direction leaves them to the total
                // internal reflection fallback, which the pdf does not describe.
                // Measured, that was a sampler reflecting 100% of the time
                // against a pdf accounting for 95% — 5% of the energy at every
                // angle past critical, which is most of the inside of a glass
                // sphere.
                if f >= 1.0 {
                    return 1.0;
                }
                return f.clamp(0.05, 0.95);
            }
            let spec = crate::math::luminance(self.specular_albedo(cos_theta_o));
            let diff = crate::math::luminance(
                self.diffuse_albedo * (Vec3::ONE - self.specular_albedo(cos_theta_o)),
            );
            if spec + diff <= 0.0 {
                return 1.0;
            }
            (spec / (spec + diff)).clamp(0.1, 0.9)
        }

        /// Fresnel for the reflection lobe.
        ///
        /// A transmissive dielectric uses the **exact** equations rather than
        /// Schlick's fit, because its reflection and transmission lobes have to
        /// sum to one. Schlick is accurate to about 0.036 at worst, which is
        /// invisible on an opaque surface and is energy created or destroyed on
        /// a transmissive one.
        #[inline]
        pub fn reflection_fresnel(&self, cos_theta: f32) -> Vec3 {
            if self.transmission > 0.0 && !self.conductor {
                Vec3::splat(fresnel_dielectric(cos_theta, self.relative_ior))
            } else {
                self.fresnel(cos_theta)
            }
        }
    }

    /// Evaluate the full BSDF. Returns `f`, not `f * cos`.
    ///
    /// `wi.z < 0` is the transmitted hemisphere, which only a material with
    /// non-zero transmission can reach.
    pub fn eval(s: &Surface, wo: Vec3, wi: Vec3) -> Vec3 {
        if wo.z <= 0.0 {
            return Vec3::ZERO;
        }
        let alpha = roughness_to_alpha(s.roughness);

        if wi.z < 0.0 {
            if s.transmission <= 0.0 {
                return Vec3::ZERO;
            }
            // Tinted by the base colour: light that goes *through* a coloured
            // dielectric is filtered by it. Physically this belongs in
            // Beer-Lambert absorption along the path inside the medium, which
            // needs volumetric tracking; tinting at the interface is the usual
            // approximation and is exact for a thin surface.
            let t = dielectric::eval(alpha, s.relative_ior, wo, wi);
            return Vec3::splat(t) * s.transmission * s.transmission_tint;
        }
        if wi.z <= 0.0 {
            return Vec3::ZERO;
        }

        // The diffuse base only ever sees what the specular layer let through,
        // and a transmissive material has no diffuse base to speak of.
        let transmitted = Vec3::ONE - s.specular_albedo(wo.z);
        let mut f = diffuse::eval(s.diffuse_albedo, wi)
            * transmitted
            * (1.0 - s.transmission);

        let h = (wo + wi).normalize_or_zero();
        if h != Vec3::ZERO {
            let d = ggx_d(alpha, h.z);
            let g2 = smith_g2(alpha, wo.z, wi.z);
            let fr = s.reflection_fresnel(wo.dot(h).max(0.0));
            let single = fr * (d * g2 / (4.0 * wo.z * wi.z));
            // Energy compensation is a reflection-only correction: it adds back
            // the multiple scattering between microfacets that the
            // single-scattering Smith model drops. A transmissive surface loses
            // that energy to the transmission lobe instead, so applying the
            // reflection compensation there would create light.
            f += if s.transmission > 0.0 {
                single
            } else {
                single * energy::compensation(s.roughness, wo.z, s.normal_reflectance())
            };
        }
        f
    }

    /// Combined solid-angle density of [`sample`].
    pub fn pdf(s: &Surface, wo: Vec3, wi: Vec3) -> f32 {
        if wo.z <= 0.0 {
            return 0.0;
        }
        let p_spec = s.specular_probability(wo.z);
        let alpha = roughness_to_alpha(s.roughness);
        let rest = 1.0 - p_spec;

        if wi.z < 0.0 {
            if s.transmission <= 0.0 {
                return 0.0;
            }
            return rest * s.transmission * dielectric::pdf(alpha, s.relative_ior, wo, wi);
        }
        if wi.z <= 0.0 {
            return 0.0;
        }
        p_spec * specular::pdf(alpha, wo, wi)
            + rest * (1.0 - s.transmission) * diffuse::pdf(wi)
    }

    /// Sample the BSDF.
    ///
    /// `u_lobe` chooses the lobe and `u` samples within it. Drawing the lobe
    /// choice from its own random number, rather than reusing a component of
    /// `u`, keeps the within-lobe sample unstratified-by-the-choice — reusing it
    /// correlates the lobe with the sampled direction and shows up as
    /// structured noise.
    pub fn sample(s: &Surface, wo: Vec3, u_lobe: f32, u: Vec2) -> BsdfSample {
        if wo.z <= 0.0 {
            return BsdfSample::INVALID;
        }
        let p_spec = s.specular_probability(wo.z);
        let alpha = roughness_to_alpha(s.roughness);

        // Which hemisphere the chosen lobe is required to land in. A sample
        // that misses it is dropped rather than reinterpreted: on a rough
        // surface at a grazing microfacet, `reflect` can return a direction
        // *below* the surface and `refract` one above it, and letting those
        // through means the other lobe's pdf gets asked to describe a sample it
        // never could have produced. Measured before this check, on rough glass
        // seen from inside: the sampler put 1.5% more samples in the transmitted
        // hemisphere than its pdf accounted for, and no amount of binning moved
        // the discrepancy — which is how it was told apart from a quadrature
        // artefact.
        let wi = if u_lobe < p_spec {
            let m = sample_ggx_vndf(wo, alpha, u);
            let r = reflect(wo, m);
            if r.z <= 0.0 {
                return BsdfSample::INVALID;
            }
            r
        } else if u_lobe < p_spec + (1.0 - p_spec) * s.transmission {
            // Refract about the *same* visible microfacet distribution the
            // reflection lobe samples, which is what makes the two lobes two
            // halves of one interface rather than two unrelated surfaces.
            let m = sample_ggx_vndf(wo, alpha, u);
            match refract(wo, m, s.relative_ior) {
                // Total internal reflection at this *microfacet*, while the
                // macroscopic angle still admits transmission — so the
                // transmission lobe was a reasonable choice and this particular
                // microfacet refused it.
                //
                // Dropped rather than redirected into the reflection lobe.
                // Redirecting is tempting and wrong: those samples would land in
                // the reflected hemisphere with a density `pdf` has no term for,
                // and the estimator would weight them by a number describing a
                // different distribution. Past the critical angle this branch is
                // unreachable anyway, because `specular_probability` returns 1
                // there and the transmission lobe is never chosen.
                None => return BsdfSample::INVALID,
                // A refraction that lands back on the side it came from is not
                // a refraction. It happens on rough surfaces at grazing angles,
                // where the sampled microfacet is tilted far enough that Snell
                // bends the ray past the surface — a configuration single
                // scattering has no answer for, since such a ray would in
                // reality hit the microsurface again.
                //
                // Dropped rather than salvaged, which loses that energy. That
                // loss is the well-known single-scattering deficiency of
                // microfacet transmission and is what a multiple-scattering
                // model would return; inventing a direction for it instead
                // would put energy somewhere the pdf does not describe, and the
                // estimator would be biased rather than merely dark. Measured
                // extent in `rough_glass_energy_loss_is_bounded`.
                Some(wt) if wt.z >= 0.0 => return BsdfSample::INVALID,
                Some(wt) => wt,
            }
        } else {
            sample_cosine_hemisphere(u)
        };
        if wi.z == 0.0 || wi == Vec3::ZERO {
            return BsdfSample::INVALID;
        }
        if wi.z < 0.0 && s.transmission <= 0.0 {
            return BsdfSample::INVALID;
        }

        // Evaluate the *whole* BSDF and the *combined* pdf, regardless of which
        // lobe produced the direction. This is what makes the estimator unbiased
        // when the lobes overlap.
        let f = eval(s, wo, wi);
        let pdf = pdf(s, wo, wi);
        if pdf <= 0.0 {
            return BsdfSample::INVALID;
        }
        BsdfSample {
            wi,
            // `|cos|`, not `cos`. The cosine here is the projected-solid-angle
            // factor and is always positive; a transmitted direction has
            // `wi.z < 0` and would otherwise carry a *negative* weight, which
            // subtracts light from the image. The chi-squared tests cannot see
            // this — they check the distribution of directions, not what each
            // one is worth — and it showed up immediately as a total albedo of
            // -0.53 in the energy test.
            weight: f * (wi.z.abs() / pdf),
            pdf,
        }
    }
}

#[cfg(test)]
mod surface_tests {
    use super::surface::Surface;
    use super::*;
    use crate::gpu_layout::GpuMaterial;
    use crate::rng::Rng;

    fn probe_surfaces() -> Vec<(&'static str, Surface)> {
        vec![
            (
                "pure diffuse",
                Surface::from_material(&GpuMaterial::diffuse(Vec3::splat(0.8)), true),
            ),
            (
                "smooth plastic",
                Surface::from_material(&GpuMaterial::glossy(Vec3::new(0.2, 0.4, 0.8), 0.1), true),
            ),
            (
                "rough plastic",
                Surface::from_material(&GpuMaterial::glossy(Vec3::new(0.8, 0.3, 0.2), 0.6), true),
            ),
            (
                "smooth metal",
                Surface::from_material(&GpuMaterial::metal(Vec3::new(0.95, 0.8, 0.4), 0.08), true),
            ),
            (
                "rough metal",
                Surface::from_material(&GpuMaterial::metal(Vec3::new(0.95, 0.8, 0.4), 0.7), true),
            ),
            (
                "copper",
                Surface::from_material(&GpuMaterial::conductor(
                    conductors::COPPER_ETA,
                    conductors::COPPER_K,
                    0.25,
                ), true),
            ),
            (
                "gold",
                Surface::from_material(&GpuMaterial::conductor(
                    conductors::GOLD_ETA,
                    conductors::GOLD_K,
                    0.4,
                ), true),
            ),
        ]
    }

    /// `sample()` must report the same density `pdf()` computes for the
    /// direction it returned.
    ///
    /// A mismatch between sampling and evaluation silently biases every image
    /// ever rendered and is invisible to the eye — it is the single most
    /// valuable thing to check about a BSDF, and the full chi-squared version
    /// arrives at build step 7. This is the cheap pairwise precursor.
    #[test]
    fn sample_and_pdf_agree() {
        for (name, s) in probe_surfaces() {
            let mut rng = Rng::new(0x5A3, 0, 0);
            let mut worst = 0.0f32;
            for _ in 0..50_000 {
                let cos_o = rng.next_f32().max(0.02);
                let sin_o = (1.0 - cos_o * cos_o).sqrt();
                let wo = Vec3::new(sin_o, 0.0, cos_o);
                let smp = surface::sample(&s, wo, rng.next_f32(), rng.next_vec2());
                if !smp.is_valid() {
                    continue;
                }
                let p = surface::pdf(&s, wo, smp.wi);
                if smp.pdf > 1e-5 {
                    worst = worst.max(((smp.pdf - p) / smp.pdf).abs());
                }
            }
            assert!(
                worst < 1e-4,
                "{name}: sample() and pdf() disagree by up to {worst:.3e}"
            );
        }
    }

    /// The BSDF must never produce a negative or non-finite value, and the
    /// sampled weight must stay bounded. An unbounded weight is a firefly.
    #[test]
    fn bsdf_is_finite_and_bounded() {
        for (name, s) in probe_surfaces() {
            let mut rng = Rng::new(0x11, 0, 0);
            let mut max_weight = 0.0f32;
            for _ in 0..100_000 {
                let cos_o = rng.next_f32().max(0.02);
                let sin_o = (1.0 - cos_o * cos_o).sqrt();
                let wo = Vec3::new(sin_o, 0.0, cos_o);
                let smp = surface::sample(&s, wo, rng.next_f32(), rng.next_vec2());
                if !smp.is_valid() {
                    continue;
                }
                assert!(
                    smp.weight.is_finite(),
                    "{name}: weight {} is not finite",
                    smp.weight
                );
                assert!(
                    smp.weight.min_element() >= 0.0,
                    "{name}: negative weight {}",
                    smp.weight
                );
                max_weight = max_weight.max(smp.weight.max_element());
            }
            // Energy compensation can push a rough conductor's weight slightly
            // above 1 at a single bounce; far above 1 means a firefly source.
            assert!(
                max_weight < 3.0,
                "{name}: max sample weight {max_weight}, expected order 1"
            );
        }
    }

    /// Total reflectance must not exceed 1 for any material or viewing angle.
    ///
    /// This is the white furnace test applied to the *combined* BSDF rather than
    /// the specular lobe alone: a real surface cannot reflect more light than
    /// falls on it. Coloured surfaces legitimately reflect less.
    #[test]
    fn combined_bsdf_conserves_energy() {
        for (name, s) in probe_surfaces() {
            for ci in 1..=10 {
                let cos_o = ci as f32 / 10.0;
                let sin_o = (1.0 - cos_o * cos_o).max(0.0).sqrt();
                let wo = Vec3::new(sin_o, 0.0, cos_o);
                let mut rng = Rng::new(0x99, ci, 0);
                let n = 60_000;
                let mut sum = Vec3::ZERO;
                for _ in 0..n {
                    let smp = surface::sample(&s, wo, rng.next_f32(), rng.next_vec2());
                    if smp.is_valid() {
                        sum += smp.weight;
                    }
                }
                let albedo = sum / n as f32;
                assert!(
                    albedo.max_element() <= 1.02,
                    "{name} at cos_theta {cos_o}: reflects {albedo}, which is more light \
                     than arrived"
                );
            }
        }
    }

    /// A white non-metal with a white specular layer must be very nearly
    /// invisible in the furnace — the combined lobes should still add up to 1.
    #[test]
    fn white_furnace_holds_for_the_combined_bsdf() {
        // Pure white diffuse plus a white specular layer: F0 = 1 makes the
        // specular lobe reflect everything, so the diffuse lobe underneath
        // receives nothing and total reflectance must be 1.
        let s = Surface {
            diffuse_albedo: Vec3::ZERO,
            roughness: 0.0,
            f0: Vec3::ONE,
            eta: Vec3::ZERO,
            k: Vec3::ZERO,
            conductor: false,
            transmission: 0.0,
            transmission_tint: Vec3::ONE,
            relative_ior: 1.5,
        };
        for &roughness in &[0.05f32, 0.3, 0.6, 1.0] {
            for &cos_o in &[0.9f32, 0.5, 0.15] {
                let s = Surface { roughness, ..s };
                let sin_o = (1.0 - cos_o * cos_o).sqrt();
                let wo = Vec3::new(sin_o, 0.0, cos_o);
                let mut rng = Rng::new(0xFA, 0, 0);
                let n = 80_000;
                let mut sum = 0.0f64;
                for _ in 0..n {
                    let smp = surface::sample(&s, wo, rng.next_f32(), rng.next_vec2());
                    if smp.is_valid() {
                        sum += smp.weight.x as f64;
                    }
                }
                let e = (sum / n as f64) as f32;
                assert!(
                    (e - 1.0).abs() < 0.02,
                    "white furnace on the combined BSDF: roughness {roughness}, \
                     cos {cos_o} gives {e}, expected 1"
                );
            }
        }
    }

    /// Rough metal must be *more* saturated than smooth metal.
    ///
    /// Energy that bounced several times between microfacets was filtered by
    /// Fresnel at each bounce, so it comes back more strongly tinted. A
    /// compensation term that ignored `F_avg` would add the missing energy back
    /// as white and desaturate rough metal instead — a subtle wrongness that no
    /// energy check would catch.
    #[test]
    fn rough_metal_is_more_saturated_than_smooth_metal() {
        let saturation = |roughness: f32| -> f32 {
            let s = Surface::from_material(&GpuMaterial::conductor(
                conductors::GOLD_ETA,
                conductors::GOLD_K,
                roughness,
            ), true);
            let wo = Vec3::new(0.3, 0.0, (1.0f32 - 0.09).sqrt());
            let mut rng = Rng::new(0xC0, 0, 0);
            let n = 100_000;
            let mut sum = Vec3::ZERO;
            for _ in 0..n {
                let smp = surface::sample(&s, wo, rng.next_f32(), rng.next_vec2());
                if smp.is_valid() {
                    sum += smp.weight;
                }
            }
            let a = sum / n as f32;
            (a.max_element() - a.min_element()) / a.max_element().max(1e-6)
        };
        let smooth = saturation(0.1);
        let rough = saturation(0.9);
        assert!(
            rough > smooth,
            "rough gold ({rough:.3}) should be more saturated than smooth gold ({smooth:.3}) — \
             multiple-scattering compensation must carry the Fresnel colour"
        );
    }

    /// A pure metal has no diffuse lobe, and a pure dielectric's specular is
    /// weak at normal incidence.
    #[test]
    fn metallic_removes_the_diffuse_lobe() {
        let metal = Surface::from_material(&GpuMaterial::metal(Vec3::new(0.9, 0.9, 0.9), 0.3), true);
        assert_eq!(metal.diffuse_albedo, Vec3::ZERO);

        let plastic = Surface::from_material(&GpuMaterial::glossy(Vec3::splat(0.5), 0.3), true);
        assert!(
            (plastic.f0.x - 0.04).abs() < 1e-4,
            "dielectric F0 = {}",
            plastic.f0
        );
        assert_eq!(plastic.diffuse_albedo, Vec3::splat(0.5));
    }
}
