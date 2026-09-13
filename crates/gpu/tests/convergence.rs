//! Build step 18: the measured convergence readout.
//!
//! The number the UI shows for "how converged is this" is a *measurement* —
//! per-pixel variance from the accumulated sum and sum of squares — rather than
//! the `1/sqrt(N)` a model would predict. This checks it actually measures that,
//! because a convergence meter that is wrong is worse than none: it is trusted
//! precisely when something else is being judged.

use glam::Vec3;
use pt_core::integrator::{RenderParams, SamplingMode};
use pt_core::math::luminance;
use pt_core::sobol::SamplerKind;
use pt_core::scenes;
use pt_gpu::{measure_convergence, render_native_on, Architecture, Gpu};

fn gpu() -> Option<Gpu> {
    match Gpu::new() {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("\n*** SKIPPING GPU TEST: {e} ***\n");
            None
        }
    }
}

fn params(spp: u32) -> RenderParams {
    RenderParams {
        width: 96,
        height: 96,
        samples: spp,
        max_depth: 5,
        ..Default::default()
    }
}

/// The reported error must fall as samples accumulate, and roughly as
/// `1/sqrt(N)` on a well-behaved scene.
///
/// "Roughly" is the honest word: the rate is only `1/sqrt(N)` when the estimator
/// has finite, well-sampled variance, which is exactly the assumption this
/// project has repeatedly found broken on heavy-tailed scenes. So the assertion
/// is that it *falls substantially*, with the measured ratios printed so the
/// departure from the model is visible rather than assumed away.
#[test]
fn the_convergence_readout_falls_with_sample_count() {
    let Some(gpu) = gpu() else { return };
    let def = scenes::cornell_box();

    eprintln!("\n{:>7} {:>14} {:>12}", "spp", "rel. std err", "vs 1/sqrt(N)");
    let mut first: Option<(f64, f64)> = None;
    let mut prev = f64::INFINITY;
    for spp in [16u32, 64, 256, 1024] {
        let e = measure_convergence(&gpu, &def, &params(spp)).expect("convergence") as f64;
        assert!(e > 0.0, "at {spp} spp the readout returned {e}, which means it \
                          found no measurable pixels");
        let model = match first {
            Some((n0, e0)) => e0 * (n0 / spp as f64).sqrt(),
            None => {
                first = Some((spp as f64, e));
                e
            }
        };
        eprintln!("{spp:>7} {e:>14.5} {:>11.2}x", e / model);
        assert!(
            e < prev,
            "the readout rose from {prev:.5} to {e:.5} going from fewer samples \
             to more; it is not measuring convergence"
        );
        prev = e;
    }

    let (n0, e0) = first.unwrap();
    let expected = e0 * (n0 / 1024.0).sqrt();
    // Within a factor of two of the model. Wider than the model's own precision
    // on purpose — the point is to catch an error that is constant, or rising,
    // or off by orders of magnitude, not to assert the Cornell box is textbook.
    assert!(
        prev > expected * 0.5 && prev < expected * 2.0,
        "at 1024 spp the readout is {prev:.5} against a 1/sqrt(N) projection of \
         {expected:.5}. That is far enough from the model to suspect the variance \
         is being accumulated from the running mean rather than per sample."
    );
}

/// A strictly worse sampling strategy must report a larger error.
///
/// This is the directional test, and the pair is chosen so the ordering is
/// **structural** rather than aesthetic: the Cornell box's light is 130 x 105 in
/// a 555-unit box, so a BSDF-only walk finds it about once in a thousand
/// bounces, while MIS connects to it at every vertex. Whatever the meter is
/// measuring, if it does not separate those two it is not measuring noise.
///
/// An earlier version of this test compared two *scenes* — cornell-box against
/// sunset, on the reasoning that sunset's mirror and glass ball make caustics.
/// It failed, and the meter was right: sunset measures 0.039 against the box's
/// 0.115 at 64 spp. Nearly every pixel in sunset is directly lit by an
/// environment that light sampling covers well, and the caustic is a small
/// minority of pixels that a mean over the image barely feels. The box is
/// mostly multiply-bounced light. "Has caustics" is not the same claim as "is
/// noisier on average", and only one of them is testable.
#[test]
fn a_worse_sampling_strategy_reports_more_error() {
    let Some(gpu) = gpu() else { return };
    let def = scenes::cornell_box();
    let mut mis = params(64);
    mis.sampling = SamplingMode::Mis;
    let mut bsdf = params(64);
    bsdf.sampling = SamplingMode::BsdfOnly;

    let good = measure_convergence(&gpu, &def, &mis).expect("mis");
    let bad = measure_convergence(&gpu, &def, &bsdf).expect("bsdf-only");
    eprintln!("\nmis {good:.5}, bsdf-only {bad:.5}  ({:.1}x)", bad / good);
    assert!(
        bad > good * 1.5,
        "BSDF-only sampling of a small area light reported {bad:.5} against MIS's \
         {good:.5}. The meter is not separating a strategy that finds the light \
         from one that stumbles into it."
    );
}

/// The two architectures must report the same figure.
///
/// They already agree on the image to 8e-9, so agreeing on its noise is not much
/// of a leap — but the sum of squares is accumulated in two different places
/// (`shaders/trace/megakernel.wgsl` and `shaders/wavefront/resolve.wgsl`), and
/// the wavefront's copy of the shading logic has drifted from the megakernel's
/// three separate times in this project. This is the test that would catch a
/// fourth.
#[test]
fn both_architectures_report_the_same_convergence() {
    let Some(gpu) = gpu() else { return };
    let def = scenes::cornell_box();
    let p = params(64);
    let (_, mega) = pt_gpu::render_with_measured(&gpu, &def, &p, Architecture::Megakernel)
        .expect("megakernel");
    let (_, wave) =
        pt_gpu::render_with_measured(&gpu, &def, &p, Architecture::Wavefront).expect("wavefront");
    eprintln!("\nmegakernel {mega:.5}, wavefront {wave:.5}");
    assert!(
        (mega - wave).abs() < 0.02 * mega.max(wave),
        "the architectures disagree about how noisy the same image is: \
         {mega:.5} against {wave:.5}. The sum of squares is accumulated in two \
         places and one of them has drifted."
    );
}

/// Independent renders per seed, for a ground truth to check the meter against.
fn spread_of_independent_renders(
    gpu: &Gpu,
    def: &scenes::SceneDef,
    base: &RenderParams,
    runs: u32,
) -> f64 {
    let n = (base.width * base.height) as usize;
    let mut sum = vec![Vec3::ZERO; n];
    let mut sum_sq = vec![Vec3::ZERO; n];
    for k in 0..runs {
        let mut p = *base;
        // A distinct stream per run. Without this every run is the same render
        // and the measured spread is exactly zero.
        p.frame_seed = base.frame_seed.wrapping_add(k.wrapping_mul(0x9e37_79b9));
        let film = render_native_on(gpu, def, &p).expect("render");
        for (i, v) in film.data.iter().enumerate() {
            sum[i] += *v;
            sum_sq[i] += *v * *v;
        }
    }

    // E[s] = c4(n) * sigma: the square root of an unbiased variance is itself
    // biased low, by about 1.1% at 16 runs. Small, but it is a known bias in the
    // reference and correcting it costs one constant.
    let k = runs as f32;
    let c4 = (2.0 / (k - 1.0)).sqrt() * gamma_ratio(runs);

    let mut total = 0.0f64;
    let mut count = 0u32;
    for i in 0..n {
        let mean = sum[i] / k;
        let mean_l = luminance(mean);
        // The same floor the shader applies, so the two averages are taken over
        // the same set of pixels.
        if mean_l <= 1.0e-3 {
            continue;
        }
        // Unbiased per-channel variance across runs, then luminance-weighted the
        // way the shader does it.
        let var = ((sum_sq[i] / k - mean * mean) * (k / (k - 1.0))).max(Vec3::ZERO);
        total += (luminance(var).max(0.0).sqrt() / c4 / mean_l) as f64;
        count += 1;
    }
    total / count.max(1) as f64
}

/// `gamma(n/2) / gamma((n-1)/2)`, for the c4 correction above.
fn gamma_ratio(n: u32) -> f32 {
    // Only ever called with the run counts below, and a lgamma is not worth
    // pulling in a dependency for.
    match n {
        8 => 0.9650 / (2.0f32 / 7.0).sqrt(),
        16 => 0.9823 / (2.0f32 / 15.0).sqrt(),
        _ => panic!("no c4 tabulated for {n} runs"),
    }
}

/// The headline check: the reported figure must match the actual spread of
/// independent renders.
///
/// Everything else here tests that the number moves the right way. This tests
/// that it is *the right number*. The meter derives the standard error from the
/// sum and sum of squares within a single render; the reference re-renders the
/// scene sixteen times with independent seeds and measures how much the answer
/// actually moved. If the meter squared the running mean instead of accumulating
/// per-sample squares, or divided by N once instead of twice, this is off by a
/// factor of eight or more.
///
/// Forced onto the independent sampler, because the identity being checked —
/// `Var[mean] = Var[sample] / N` — assumes the samples are independent. Sobol's
/// whole purpose is to break that assumption in the useful direction.
#[test]
fn the_readout_matches_the_spread_of_independent_renders() {
    let Some(gpu) = gpu() else { return };
    let def = scenes::cornell_box();
    let mut p = params(64);
    p.sampler = SamplerKind::Independent;

    let reported = measure_convergence(&gpu, &def, &p).expect("meter") as f64;
    let actual = spread_of_independent_renders(&gpu, &def, &p, 16);
    let ratio = reported / actual;
    eprintln!("\nindependent sampler: meter {reported:.5}, measured spread {actual:.5}  ({ratio:.3}x)");
    assert!(
        (0.80..1.25).contains(&ratio),
        "the meter reports {reported:.5} where sixteen independent renders \
         actually spread by {actual:.5} ({ratio:.3}x). The estimator is wrong, \
         not merely imprecise."
    );
}

/// With Sobol the readout is an honest **upper bound**, not an equality.
///
/// Stratification is what makes Sobol worth having: the samples within a pixel
/// are deliberately correlated so they cover the domain more evenly than chance
/// would, so the true spread of the mean is *below* `sqrt(var/N)`. The meter
/// computes the i.i.d. figure regardless, which means it over-reports — and
/// over-reporting is the right failure direction for a convergence readout,
/// since the alternative is telling someone an image is done when it is not.
///
/// This pins that down rather than leaving it as a claim in a comment.
#[test]
fn stratification_makes_the_readout_conservative() {
    let Some(gpu) = gpu() else { return };
    let def = scenes::cornell_box();
    let mut p = params(64);
    p.sampler = SamplerKind::Sobol;

    let reported = measure_convergence(&gpu, &def, &p).expect("meter") as f64;
    let actual = spread_of_independent_renders(&gpu, &def, &p, 16);
    eprintln!(
        "\nsobol: meter {reported:.5}, measured spread {actual:.5}  ({:.3}x)",
        reported / actual
    );
    assert!(
        reported >= actual * 0.95,
        "with Sobol the meter reported {reported:.5} but the true spread is \
         {actual:.5} — it is under-reporting, which would tell a user an image \
         has converged further than it has."
    );
}
