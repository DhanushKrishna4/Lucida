//! Build step 16: the WGSL denoiser against its Rust twin.
//!
//! A filtered image looks plausible whatever the weights are doing — too much
//! blur reads as "well denoised" and too little as "still converging" — so
//! nothing about the picture tells you whether the shader implements the same
//! filter as the reference. Diffing them is the only way to know.

use pt_core::denoise::{denoise, DenoiseParams};
use pt_core::image::compare;
use pt_core::integrator::{self, RenderParams};
use pt_core::scenes;
use pt_gpu::{render_native_denoised, Gpu};

fn gpu() -> Option<Gpu> {
    match Gpu::new() {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("\n*** SKIPPING GPU TEST: {e} ***\n");
            None
        }
    }
}

#[test]
fn gpu_denoiser_matches_the_cpu_reference() {
    let Some(gpu) = gpu() else { return };
    let params = RenderParams {
        width: 96,
        height: 96,
        samples: 16,
        max_depth: 5,
        ..Default::default()
    };
    let dn = DenoiseParams::default();

    for def in [scenes::cornell_box(), scenes::cornell_mesh()] {
        let (noisy, guides) = integrator::render_with_guides(&def, &params);
        let cpu = denoise(&noisy, &guides, &dn);
        let gpu_film = render_native_denoised(&gpu, &def, &params, &dn).expect("gpu denoise");

        let d = compare(&cpu, &gpu_film).expect("same size");
        eprintln!(
            "{:<14} denoised: mean rel {:.3e}, energy ratio {:.6}",
            def.name,
            d.mean_rel,
            d.mean_b / d.mean_a
        );
        assert!(
            d.mean_rel < 4.0e-3,
            "{}: the GPU denoiser differs from the CPU one by {:.3e} (max {:.3e} \
             at {:?}).\nCheck, in order:\n\
               1. the deviation is measured on the demodulated signal, not the raw\n\
               2. all four edge-stopping weights are present and multiplied\n\
               3. the ping-pong binds the buffer the last pass wrote\n\
               4. the guides are divided by the sample count on both sides",
            def.name,
            d.mean_rel,
            d.max_abs,
            d.max_abs_at
        );
    }
}

/// Denoising must not disturb the render it filters.
///
/// The accumulation buffer is read-only to the denoiser, so a denoised and an
/// undenoised render of the same scene must produce the *same accumulation* —
/// checked by rendering both ways and comparing the undenoised output.
#[test]
fn denoising_does_not_change_the_render() {
    let Some(gpu) = gpu() else { return };
    let def = scenes::cornell_box();
    let params = RenderParams {
        width: 64,
        height: 64,
        samples: 16,
        max_depth: 4,
        ..Default::default()
    };
    let plain = pt_gpu::render_native_on(&gpu, &def, &params).expect("gpu render");
    // Run the denoiser, then render again the normal way. If the denoiser had
    // written into the accumulation, this second render would differ.
    let _ = render_native_denoised(&gpu, &def, &params, &DenoiseParams::default());
    let again = pt_gpu::render_native_on(&gpu, &def, &params).expect("gpu render");

    let d = compare(&plain, &again).expect("same size");
    assert_eq!(
        d.mean_rel, 0.0,
        "the render changed after denoising ran; the denoiser must treat the \
         accumulation as read-only"
    );
}
