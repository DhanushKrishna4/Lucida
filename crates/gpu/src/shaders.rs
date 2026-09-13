//! WGSL sources, embedded at compile time, with a tiny `//!include` resolver.
//!
//! WGSL has no preprocessor and no module system, so shared code has to be
//! textually spliced. The same resolution happens on the web side in a Vite
//! plugin (`web/vite-wgsl.ts`) — both implement the identical directive so the
//! native harness and the browser compile *byte-identical* shader source. That
//! matters: "works natively, breaks in the browser" is much easier to diagnose
//! when you can rule out the source differing.
//!
//! Sources are embedded rather than read from disk so the harness works from any
//! working directory and inside `cargo test`.

use std::collections::HashSet;

macro_rules! shader_files {
    ($($path:literal),* $(,)?) => {
        pub const FILES: &[(&str, &str)] = &[
            $(($path, include_str!(concat!("../../../shaders/", $path))),)*
        ];
    };
}

shader_files! {
    "common/generated.wgsl",
    "common/math.wgsl",
    "common/rng.wgsl",
    "common/ggx_energy.wgsl",
    "common/bsdf.wgsl",
    "common/bvh.wgsl",
    "common/ray.wgsl",
    "common/triangle_shading.wgsl",
    "common/primitives.wgsl",
    "common/scene.wgsl",
    "common/diagnostic.wgsl",
    "common/envmap.wgsl",
    "common/light.wgsl",
    "common/camera.wgsl",
    "common/tonemap.wgsl",
    "trace/megakernel.wgsl",
    "wavefront/generate.wgsl",
    "wavefront/extend.wgsl",
    "wavefront/shade.wgsl",
    "wavefront/connect.wgsl",
    "wavefront/reset.wgsl",
    "wavefront/resolve.wgsl",
    "gradient.wgsl",
    "display.wgsl",
    "denoise/atrous.wgsl",
    "stats/convergence.wgsl",
    "lbvh/prepare.wgsl",
    "lbvh/hierarchy.wgsl",
    "lbvh/fit.wgsl",
    "lbvh/radix.wgsl",
    "test/eval_diagnostic.wgsl",
    "test/eval_vec3.wgsl",
    "test/eval_srgb.wgsl",
    "test/eval_bsdf.wgsl",
}

pub fn raw(path: &str) -> Option<&'static str> {
    FILES.iter().find(|(p, _)| *p == path).map(|(_, s)| *s)
}

/// Resolve `//!include "path"` directives, depth-first, including each file at
/// most once.
///
/// Include-once is required, not a nicety: WGSL rejects a duplicate function or
/// struct definition, so a diamond include (megakernel includes both scene.wgsl
/// and camera.wgsl, each of which wants math.wgsl) would otherwise fail to
/// compile.
pub fn resolve(entry: &str) -> Result<String, String> {
    let mut seen = HashSet::new();
    let mut out = String::new();
    resolve_into(entry, &mut seen, &mut out, 0)?;
    Ok(out)
}

fn resolve_into(
    path: &str,
    seen: &mut HashSet<String>,
    out: &mut String,
    depth: usize,
) -> Result<(), String> {
    if depth > 16 {
        return Err(format!(
            "include depth exceeded at `{path}` — is there a cycle?"
        ));
    }
    if !seen.insert(path.to_string()) {
        return Ok(());
    }
    let src = raw(path).ok_or_else(|| {
        format!(
            "no such shader `{path}`; known files: {}",
            FILES.iter().map(|(p, _)| *p).collect::<Vec<_>>().join(", ")
        )
    })?;

    for line in src.lines() {
        if let Some(inc) = parse_include(line) {
            out.push_str(&format!("// ---- begin {inc} (included by {path}) ----\n"));
            resolve_into(inc, seen, out, depth + 1)?;
            out.push_str(&format!("// ---- end {inc} ----\n"));
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    Ok(())
}

/// Parse `//!include "some/path.wgsl"`. Returns the path, or `None`.
pub fn parse_include(line: &str) -> Option<&str> {
    let rest = line.trim().strip_prefix("//!include")?;
    let rest = rest.trim();
    let rest = rest.strip_prefix('"')?;
    rest.strip_suffix('"')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_include_directives() {
        assert_eq!(
            parse_include("//!include \"common/math.wgsl\""),
            Some("common/math.wgsl")
        );
        assert_eq!(parse_include("  //!include  \"a.wgsl\"  "), Some("a.wgsl"));
        assert_eq!(parse_include("// include \"a.wgsl\""), None);
        assert_eq!(parse_include("let x = 1; // !include"), None);
        assert_eq!(parse_include("//!include common/math.wgsl"), None);
    }

    #[test]
    fn every_include_target_exists() {
        for (path, src) in FILES {
            for line in src.lines() {
                if let Some(inc) = parse_include(line) {
                    assert!(raw(inc).is_some(), "{path} includes missing file `{inc}`");
                }
            }
        }
    }

    /// The diamond: megakernel pulls in scene.wgsl and camera.wgsl, both of which
    /// depend on math.wgsl. If include-once regressed, WGSL would reject the
    /// duplicate definitions — catch it here with a cheap textual check instead.
    #[test]
    fn shared_code_is_included_exactly_once() {
        let src = resolve("trace/megakernel.wgsl").unwrap();
        assert_eq!(
            src.matches("fn offset_ray_origin(").count(),
            1,
            "math.wgsl included twice"
        );
        assert_eq!(
            src.matches("fn pcg_hash(").count(),
            1,
            "rng.wgsl included twice"
        );
        assert_eq!(
            src.matches("struct Uniforms").count(),
            1,
            "generated.wgsl included twice"
        );
    }

    #[test]
    fn resolved_megakernel_has_everything_it_needs() {
        let src = resolve("trace/megakernel.wgsl").unwrap();
        for needed in [
            "struct Material",
            "struct Primitive",
            "struct Light",
            "struct Uniforms",
            "fn scene_intersect(",
            "fn generate_ray(",
            "fn sample_cosine_hemisphere(",
            "fn rng_init(",
            "fn trace_path(",
            "fn direct_light(",
            "fn surface_sample(",
            "fn intersect_triangles(",
            "fn main(",
        ] {
            assert!(
                src.contains(needed),
                "resolved megakernel is missing `{needed}`"
            );
        }
        assert!(!src.contains("//!include"), "unresolved include remains");
    }

    #[test]
    fn missing_include_is_a_clear_error() {
        let e = resolve("nope.wgsl").unwrap_err();
        assert!(e.contains("no such shader"), "{e}");
    }
}
