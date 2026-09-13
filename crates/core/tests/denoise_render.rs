//! Build step 16: denoising, measured on real renders.
//!
//! The unit tests in `denoise.rs` check the filter on synthetic images where the
//! right answer is known exactly. This checks the thing that actually matters and
//! that a synthetic test cannot: on a real render, does filtering get you
//! *closer to the converged image* than not filtering?
//!
//! That framing is the whole point. A denoiser trades variance for bias, and
//! "less noisy" is not the same as "more correct" — a heavy blur is very quiet
//! and very wrong. So every measurement here is against a converged reference.

use pt_core::denoise::{denoise, DenoiseParams};
use pt_core::integrator::{self, Film, RenderParams};
use pt_core::scenes;

const W: u32 = 96;
const H: u32 = 96;

fn params(spp: u32) -> RenderParams {
    RenderParams {
        width: W,
        height: H,
        samples: spp,
        max_depth: 5,
        ..Default::default()
    }
}

/// Root-mean-square error against a reference.
fn rmse(a: &Film, b: &Film) -> f64 {
    let n = a.data.len() as f64;
    (a.data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| (*x - *y).length_squared() as f64)
        .sum::<f64>()
        / n)
        .sqrt()
}

/// Sorted per-pixel distances from a reference.
fn sorted_errors(a: &Film, b: &Film) -> Vec<f64> {
    let mut e: Vec<f64> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| (*x - *y).length() as f64)
        .collect();
    e.sort_by(|p, q| p.partial_cmp(q).unwrap());
    e
}

/// Denoising must move the **typical** pixel toward the converged answer.
///
/// # Why the median and not RMSE
///
/// Measured on the Cornell box at 4 spp, the worst **1% of pixels hold 98% of
/// the squared error** — median error 0.019 against an RMSE of 0.196. RMSE is
/// therefore almost entirely a firefly metric, and an edge-avoiding filter
/// cannot move it by design: its luminance weight sees an outlier as a different
/// surface and refuses to blend it away. Judging the denoiser by RMSE measures
/// how well it suppresses fireflies, which is a *different technique* with its
/// own bias trade, and reports ~1.0x however well the filter is working.
///
/// So the claim is stated on the median, where the filter's actual job is, and
/// the tail is reported alongside rather than hidden.
#[test]
fn denoising_improves_the_typical_pixel_at_low_sample_counts() {
    eprintln!(
        "\n{:<14} {:>5} {:>10} {:>10} {:>8} {:>9}",
        "scene", "spp", "raw med", "denoised", "ratio", "p99 ratio"
    );
    for def in [scenes::cornell_box(), scenes::cornell_mesh()] {
        let reference = integrator::render(&def, &params(2048));
        for spp in [4u32, 16] {
            let (noisy, guides) = integrator::render_with_guides(&def, &params(spp));
            let filtered = denoise(&noisy, &guides, &DenoiseParams::default());
            let (re, de) = (
                sorted_errors(&noisy, &reference),
                sorted_errors(&filtered, &reference),
            );
            let n = re.len();
            let med = re[n / 2] / de[n / 2];
            let p99 = re[n * 99 / 100] / de[n * 99 / 100];
            eprintln!(
                "{:<14} {spp:>5} {:>10.5} {:>10.5} {med:>7.2}x {p99:>8.2}x",
                def.name,
                re[n / 2],
                de[n / 2]
            );
            assert!(
                med > 1.3,
                "{} at {spp} spp: the median pixel only improved {med:.2}x. The \
                 filter is not doing its job — check the edge-stopping weights \
                 are not rejecting every tap.",
                def.name
            );
        }
    }
}

/// And it must be honest about the sample count past which it stops helping.
///
/// A denoiser trades variance for bias. The bias is a fixed cost, so once the
/// render is converged enough the trade reverses and filtering makes the image
/// *worse*. Knowing roughly where that crossover sits is what separates a useful
/// default from one that quietly degrades every long render.
///
/// Pinned rather than merely documented: this is the number a UI would key its
/// automatic-disable on.
#[test]
fn the_benefit_reverses_once_the_render_converges() {
    let def = scenes::cornell_box();
    let reference = integrator::render(&def, &params(4096));

    eprintln!("\n{:>6} {:>10} {:>10} {:>8}", "spp", "raw med", "denoised", "ratio");
    let mut ratios = Vec::new();
    for spp in [4u32, 32, 256] {
        let (noisy, guides) = integrator::render_with_guides(&def, &params(spp));
        let filtered = denoise(&noisy, &guides, &DenoiseParams::default());
        let (re, de) = (
            sorted_errors(&noisy, &reference),
            sorted_errors(&filtered, &reference),
        );
        let n = re.len();
        let ratio = re[n / 2] / de[n / 2];
        eprintln!("{spp:>6} {:>10.5} {:>10.5} {ratio:>7.2}x", re[n / 2], de[n / 2]);
        ratios.push(ratio);
    }
    assert!(
        ratios[0] > ratios[2],
        "the advantage should shrink as the render converges, but it went {ratios:?}"
    );
    assert!(
        ratios[2] < 1.2,
        "at 256 spp the filter still claims a {:.2}x improvement; either the \
         reference is not converged or the filter has stopped adapting to the \
         noise level",
        ratios[2]
    );
}

/// Denoising must not change what the render converges to.
///
/// The accumulation buffer is untouched and the filter is a display-time
/// product, so this is really a check that nothing has quietly started writing
/// back into the render. Cheap insurance against the worst possible regression:
/// a denoiser that biases the integrator itself.
#[test]
fn denoising_does_not_touch_the_accumulation() {
    let def = scenes::cornell_box();
    let (a, guides) = integrator::render_with_guides(&def, &params(32));
    let before = a.data.clone();
    let _ = denoise(&a, &guides, &DenoiseParams::default());
    assert_eq!(
        a.data, before,
        "denoising modified the film it was handed; it must return a new image"
    );
}

/// The guides themselves must be sane.
///
/// A denoiser silently degrades to a plain blur if its guides are wrong, and a
/// plain blur still reduces RMSE at low sample counts — so the test above would
/// pass with the edge-stopping entirely broken. These check the inputs directly.
#[test]
fn guides_describe_the_first_hit() {
    let def = scenes::cornell_box();
    let (_, guides) = integrator::render_with_guides(&def, &params(4));
    let n = (W * H) as usize;

    let mut lit = 0;
    let mut unit_normals = 0;
    for i in 0..n {
        assert!(
            guides.albedo[i].min_element() >= 0.0 && guides.albedo[i].is_finite(),
            "albedo {i} is {:?}",
            guides.albedo[i]
        );
        assert!(guides.depth[i] >= 0.0 && guides.depth[i].is_finite());
        if guides.depth[i] > 0.0 {
            lit += 1;
            let len = guides.normal[i].length();
            assert!(
                len <= 1.001,
                "normal {i} has length {len}; an average of unit vectors cannot \
                 exceed one"
            );
            if len > 0.9 {
                unit_normals += 1;
            }
        }
    }
    // The Cornell box fills the frame, so essentially every pixel is a hit.
    assert!(
        lit > n * 9 / 10,
        "only {lit} of {n} pixels recorded a first hit"
    );
    // Most pixels see one surface across all their samples, so their averaged
    // normal stays near unit. A silhouette pixel's does not, and that shortening
    // is a deliberate signal rather than a defect — it is what tells the filter
    // the pixel straddles an edge — so this is a majority claim, not a universal
    // one. Asserting it universally is how this test first failed, on a pixel at
    // the sphere's outline.
    assert!(
        unit_normals > lit * 9 / 10,
        "only {unit_normals} of {lit} hit pixels have a near-unit averaged \
         normal; that many silhouettes would mean the guides are averaging \
         across surfaces they should not"
    );
}
