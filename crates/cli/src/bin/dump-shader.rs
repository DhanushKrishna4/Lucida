//! Print a shader with its `//!include` directives resolved, numbered.
//!
//! WGSL compile errors report a line number in the *resolved* source, which does
//! not exist on disk anywhere. This prints it.
//!
//! ```text
//! cargo run -p pt-cli --bin dump-shader -- trace/megakernel.wgsl
//! ```

fn main() {
    let Some(entry) = std::env::args().nth(1) else {
        eprintln!("usage: dump-shader <path-under-shaders/>\n");
        eprintln!("available:");
        for (p, _) in pt_gpu::shaders::FILES {
            eprintln!("  {p}");
        }
        std::process::exit(1);
    };
    match pt_gpu::shaders::resolve(&entry) {
        Ok(src) => {
            for (i, line) in src.lines().enumerate() {
                println!("{:5} | {line}", i + 1);
            }
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}
