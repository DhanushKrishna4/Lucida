//! HDR image I/O and comparison metrics.
//!
//! Everything the renderer produces is **linear HDR** and stays that way until
//! display. Reference images are written as PFM (Portable FloatMap): raw
//! `f32` RGB with a 3-line ASCII header, no compression, no colour management,
//! no dependency. Exactly what a numerical oracle should round-trip through.
//!
//! PPM output exists only for eyeballing; it is lossy by construction and is
//! never used for comparison.

use crate::integrator::Film;
use glam::Vec3;
use std::io::{self, Read, Write};
use std::path::Path;

// ---------------------------------------------------------------------------
// PFM
// ---------------------------------------------------------------------------

/// Write a PFM file.
///
/// Note the row order: PFM stores rows **bottom-to-top**, while [`Film`] stores
/// row 0 as the top row. We flip on write and flip back on read, so a
/// write/read round trip is the identity and external viewers show the image the
/// right way up.
pub fn write_pfm(path: impl AsRef<Path>, film: &Film) -> io::Result<()> {
    let mut f = io::BufWriter::new(std::fs::File::create(path)?);
    write!(f, "PF\n{} {}\n-1.0\n", film.width, film.height)?;
    for y in (0..film.height).rev() {
        for x in 0..film.width {
            let p = film.pixel(x, y);
            f.write_all(&p.x.to_le_bytes())?;
            f.write_all(&p.y.to_le_bytes())?;
            f.write_all(&p.z.to_le_bytes())?;
        }
    }
    f.flush()
}

pub fn read_pfm(path: impl AsRef<Path>) -> io::Result<Film> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut bytes)?;

    let err = |m: &str| io::Error::new(io::ErrorKind::InvalidData, m.to_string());

    // Three whitespace-terminated header tokens, then exactly one byte of
    // whitespace, then the payload.
    let mut pos = 0usize;
    let mut token = || -> io::Result<String> {
        while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        let start = pos;
        while pos < bytes.len() && !bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if start == pos {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "truncated PFM header",
            ));
        }
        let t = String::from_utf8_lossy(&bytes[start..pos]).into_owned();
        pos += 1; // consume the single separating whitespace byte
        Ok(t)
    };

    let magic = token()?;
    if magic != "PF" {
        return Err(err("only colour PFM ('PF') is supported"));
    }
    let width: u32 = token()?.parse().map_err(|_| err("bad width"))?;
    let height: u32 = token()?.parse().map_err(|_| err("bad height"))?;
    let scale: f32 = token()?.parse().map_err(|_| err("bad scale"))?;
    if scale >= 0.0 {
        return Err(err(
            "big-endian PFM is not supported (scale must be negative)",
        ));
    }

    let expected = (width as usize) * (height as usize) * 3 * 4;
    let payload = &bytes[pos..];
    if payload.len() < expected {
        return Err(err("PFM payload is shorter than the header claims"));
    }

    let mut film = Film::new(width, height);
    let mut i = 0usize;
    for y in (0..height).rev() {
        for x in 0..width {
            let mut c = [0f32; 3];
            for ch in &mut c {
                *ch = f32::from_le_bytes([
                    payload[i],
                    payload[i + 1],
                    payload[i + 2],
                    payload[i + 3],
                ]);
                i += 4;
            }
            film.data[(y * width + x) as usize] = Vec3::from_array(c);
        }
    }
    Ok(film)
}

// ---------------------------------------------------------------------------
// Display transform + PPM
// ---------------------------------------------------------------------------

/// The sRGB opto-electronic transfer function.
///
/// Not `powf(1/2.2)`: sRGB has a linear segment near black, and the exponent on
/// the curved segment is 2.4, not 2.2. Using the gamma approximation shifts
/// shadow values by a couple of percent, which is exactly the magnitude of
/// difference a CPU-vs-GPU comparison is trying to detect — so the real curve
/// goes in from the start.
#[inline]
pub fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Write an 8-bit PPM for quick visual inspection.
///
/// `exposure` scales linear radiance *before* the curve, so it behaves like a
/// camera stop rather than a brightness slider.
pub fn write_ppm(
    path: impl AsRef<Path>,
    film: &Film,
    exposure: f32,
    tonemap: crate::tonemap::Tonemap,
) -> io::Result<()> {
    let mut f = io::BufWriter::new(std::fs::File::create(path)?);
    write!(f, "P6\n{} {}\n255\n", film.width, film.height)?;
    let mut row = Vec::with_capacity(film.width as usize * 3);
    for y in 0..film.height {
        row.clear();
        for x in 0..film.width {
            let p = tonemap.apply(film.pixel(x, y) * exposure);
            for ch in [p.x, p.y, p.z] {
                row.push((linear_to_srgb(ch) * 255.0 + 0.5) as u8);
            }
        }
        f.write_all(&row)?;
    }
    f.flush()
}

// ---------------------------------------------------------------------------
// Comparison metrics
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default)]
pub struct ImageDiff {
    /// Root mean squared error over all channels, in linear radiance units.
    pub rmse: f64,
    /// Largest absolute per-channel difference.
    pub max_abs: f64,
    /// Location of `max_abs`.
    pub max_abs_at: (u32, u32),
    /// Mean **relative** error: `|a - b| / (|a| + |b| + eps)`, averaged.
    ///
    /// This is the metric that matters for CPU-vs-GPU agreement. Absolute error
    /// is dominated by the few very bright pixels that see the light source
    /// directly, so RMSE alone can hide a large proportional error everywhere
    /// else.
    pub mean_rel: f64,
    /// Mean absolute radiance of each image — a gross energy check. A systematic
    /// difference here means one renderer is losing or gaining energy.
    pub mean_a: f64,
    pub mean_b: f64,
}

pub fn compare(a: &Film, b: &Film) -> Result<ImageDiff, String> {
    if a.width != b.width || a.height != b.height {
        return Err(format!(
            "size mismatch: {}x{} vs {}x{}",
            a.width, a.height, b.width, b.height
        ));
    }
    let n = (a.data.len() * 3) as f64;
    let mut sum_sq = 0.0f64;
    let mut sum_rel = 0.0f64;
    let (mut sum_a, mut sum_b) = (0.0f64, 0.0f64);
    let mut max_abs = 0.0f64;
    let mut max_at = (0u32, 0u32);

    for i in 0..a.data.len() {
        let (pa, pb) = (a.data[i], b.data[i]);
        for ch in 0..3 {
            let (va, vb) = (pa[ch] as f64, pb[ch] as f64);
            let d = (va - vb).abs();
            sum_sq += d * d;
            // The epsilon keeps the denominator away from zero in black regions,
            // where a tiny absolute difference is not meaningfully "relative".
            sum_rel += d / (va.abs() + vb.abs() + 1e-4);
            sum_a += va;
            sum_b += vb;
            if d > max_abs {
                max_abs = d;
                max_at = ((i as u32) % a.width, (i as u32) / a.width);
            }
        }
    }

    Ok(ImageDiff {
        rmse: (sum_sq / n).sqrt(),
        max_abs,
        max_abs_at: max_at,
        mean_rel: sum_rel / n,
        mean_a: sum_a / n,
        mean_b: sum_b / n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_film() -> Film {
        let mut f = Film::new(7, 5);
        for y in 0..5 {
            for x in 0..7 {
                f.data[(y * 7 + x) as usize] =
                    Vec3::new(x as f32 * 0.25, y as f32 * 1.5, (x * y) as f32 * 0.125);
            }
        }
        f
    }

    #[test]
    fn pfm_round_trips_exactly() {
        let dir = std::env::temp_dir().join("pt-core-pfm-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rt.pfm");
        let a = test_film();
        write_pfm(&path, &a).unwrap();
        let b = read_pfm(&path).unwrap();
        assert_eq!(a.width, b.width);
        assert_eq!(a.height, b.height);
        // Bit-exact: PFM is raw f32 and we must not be rescaling anything.
        assert_eq!(a.data, b.data, "PFM round trip is not lossless");
        std::fs::remove_file(&path).ok();
    }

    /// A round trip must also preserve orientation. Writing bottom-to-top and
    /// reading back top-to-bottom (or forgetting one of the flips) produces a
    /// vertically mirrored image that the equality test above would still catch
    /// only because the test image is asymmetric — so assert it directly.
    #[test]
    fn pfm_preserves_orientation() {
        let dir = std::env::temp_dir().join("pt-core-pfm-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("orient.pfm");
        let mut a = Film::new(2, 2);
        a.data[0] = Vec3::new(1.0, 0.0, 0.0); // top-left
        a.data[3] = Vec3::new(0.0, 0.0, 1.0); // bottom-right
        write_pfm(&path, &a).unwrap();
        let b = read_pfm(&path).unwrap();
        assert_eq!(b.pixel(0, 0), Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(b.pixel(1, 1), Vec3::new(0.0, 0.0, 1.0));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn srgb_transfer_matches_the_standard() {
        assert!((linear_to_srgb(0.0) - 0.0).abs() < 1e-9);
        assert!((linear_to_srgb(1.0) - 1.0).abs() < 1e-6);
        // Mid-grey: linear 0.2140 is sRGB 0.5.
        assert!((linear_to_srgb(0.2140) - 0.5).abs() < 2e-3);
        // The two segments must meet at the breakpoint.
        let b = 0.003_130_8;
        assert!((linear_to_srgb(b) - 12.92 * b).abs() < 1e-6);
        assert!((linear_to_srgb(b + 1e-7) - 12.92 * b).abs() < 1e-5);
    }

    #[test]
    fn identical_images_compare_as_zero() {
        let a = test_film();
        let d = compare(&a, &a).unwrap();
        assert_eq!(d.rmse, 0.0);
        assert_eq!(d.max_abs, 0.0);
        assert_eq!(d.mean_rel, 0.0);
    }

    #[test]
    fn compare_reports_a_known_difference() {
        let a = Film {
            width: 1,
            height: 1,
            data: vec![Vec3::splat(1.0)],
        };
        let b = Film {
            width: 1,
            height: 1,
            data: vec![Vec3::splat(3.0)],
        };
        let d = compare(&a, &b).unwrap();
        assert!((d.rmse - 2.0).abs() < 1e-12);
        assert!((d.max_abs - 2.0).abs() < 1e-12);
        assert!((d.mean_rel - 2.0 / 4.0001).abs() < 1e-4);
        assert!((d.mean_a - 1.0).abs() < 1e-12);
        assert!((d.mean_b - 3.0).abs() < 1e-12);
    }

    #[test]
    fn size_mismatch_is_an_error() {
        let a = Film::new(2, 2);
        let b = Film::new(3, 2);
        assert!(compare(&a, &b).is_err());
    }
}

// ---------------------------------------------------------------------------
// PNG
// ---------------------------------------------------------------------------
//
// A minimal, dependency-free PNG encoder. PPM is fine as a debug dump but no
// ordinary image viewer opens it, and being able to glance at a render without
// a conversion step matters more than file size for preview output.
//
// Compression is deflate in "stored" mode: valid zlib, no compression, ~1.0x the
// raw size. A real DEFLATE implementation would be a lot of code for a preview
// path, and every PNG decoder handles stored blocks.

fn crc32(data: &[u8]) -> u32 {
    // Table-free bitwise CRC-32 (IEEE 802.3 polynomial, reflected form 0xEDB88320).
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

fn png_chunk(out: &mut Vec<u8>, kind: &[u8; 4], payload: &[u8]) {
    out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    let mut crc_input = Vec::with_capacity(4 + payload.len());
    crc_input.extend_from_slice(kind);
    crc_input.extend_from_slice(payload);
    out.extend_from_slice(&crc_input);
    out.extend_from_slice(&crc32(&crc_input).to_be_bytes());
}

/// Encode 8-bit RGB rows (tightly packed, `width * height * 3` bytes) as a PNG.
pub fn encode_png_rgb8(width: u32, height: u32, rgb: &[u8]) -> Vec<u8> {
    assert_eq!(rgb.len(), (width * height * 3) as usize);

    // PNG scanlines are each prefixed with a filter-type byte; 0 = None.
    let mut raw = Vec::with_capacity(rgb.len() + height as usize);
    for y in 0..height as usize {
        raw.push(0);
        let start = y * width as usize * 3;
        raw.extend_from_slice(&rgb[start..start + width as usize * 3]);
    }

    // zlib wrapper: CMF=0x78 (deflate, 32K window), FLG=0x01 makes the header
    // checksum (CMF*256 + FLG) a multiple of 31, which decoders verify.
    let mut z = vec![0x78, 0x01];
    for (i, block) in raw.chunks(65_535).enumerate() {
        let is_last = (i + 1) * 65_535 >= raw.len();
        z.push(if is_last { 1 } else { 0 }); // BFINAL, BTYPE=00 (stored)
        z.extend_from_slice(&(block.len() as u16).to_le_bytes());
        z.extend_from_slice(&(!(block.len() as u16)).to_le_bytes()); // one's complement
        z.extend_from_slice(block);
    }
    z.extend_from_slice(&adler32(&raw).to_be_bytes());

    let mut out = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8 bits/channel, colour type 2 (RGB)
    png_chunk(&mut out, b"IHDR", &ihdr);
    png_chunk(&mut out, b"IDAT", &z);
    png_chunk(&mut out, b"IEND", &[]);
    out
}

/// Tone map, encode to sRGB, and write a PNG.
///
/// The same pipeline the browser's display pass runs — exposure in scene-linear,
/// then the operator, then the sRGB transfer function — so a CLI render and a
/// browser render of the same film are pixel-comparable.
pub fn write_png(
    path: impl AsRef<Path>,
    film: &Film,
    exposure: f32,
    tonemap: crate::tonemap::Tonemap,
) -> io::Result<()> {
    let mut rgb = Vec::with_capacity((film.width * film.height * 3) as usize);
    for y in 0..film.height {
        for x in 0..film.width {
            let p = tonemap.apply(film.pixel(x, y) * exposure);
            for ch in [p.x, p.y, p.z] {
                rgb.push((linear_to_srgb(ch) * 255.0 + 0.5) as u8);
            }
        }
    }
    std::fs::write(path, encode_png_rgb8(film.width, film.height, &rgb))
}

#[cfg(test)]
mod png_tests {
    use super::*;

    /// Known CRC-32 check value for the ASCII string "123456789".
    #[test]
    fn crc32_matches_the_reference_vector() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn adler32_matches_the_reference_vector() {
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
    }

    #[test]
    fn png_has_a_valid_signature_and_chunk_structure() {
        let png = encode_png_rgb8(2, 2, &[255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255]);
        assert_eq!(&png[..8], &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        assert_eq!(&png[12..16], b"IHDR");
        assert_eq!(&png[png.len() - 8..png.len() - 4], b"IEND");

        // Walk the chunks and verify every CRC.
        let mut i = 8;
        let mut kinds = Vec::new();
        while i + 8 <= png.len() {
            let len = u32::from_be_bytes(png[i..i + 4].try_into().unwrap()) as usize;
            let kind = &png[i + 4..i + 8];
            let body = &png[i + 4..i + 8 + len];
            let stored = u32::from_be_bytes(png[i + 8 + len..i + 12 + len].try_into().unwrap());
            assert_eq!(
                crc32(body),
                stored,
                "bad CRC on chunk {:?}",
                std::str::from_utf8(kind)
            );
            kinds.push(String::from_utf8_lossy(kind).into_owned());
            i += 12 + len;
        }
        assert_eq!(kinds, ["IHDR", "IDAT", "IEND"]);
        assert_eq!(i, png.len());
    }

    /// A stored-block zlib stream larger than one 64 KiB block must still set
    /// BFINAL exactly once, on the last block.
    #[test]
    fn multi_block_zlib_is_well_formed() {
        let (w, h) = (256u32, 256u32);
        let png = encode_png_rgb8(w, h, &vec![128u8; (w * h * 3) as usize]);
        // Locate IDAT and count final-block flags.
        let idat_len = u32::from_be_bytes(png[33..37].try_into().unwrap()) as usize;
        let z = &png[41..41 + idat_len];
        let mut i = 2; // skip the zlib header
        let mut finals = 0;
        loop {
            let hdr = z[i];
            assert_eq!(hdr & 0x06, 0, "expected a stored block");
            if hdr & 1 == 1 {
                finals += 1;
            }
            let len = u16::from_le_bytes(z[i + 1..i + 3].try_into().unwrap()) as usize;
            let nlen = u16::from_le_bytes(z[i + 3..i + 5].try_into().unwrap());
            assert_eq!(nlen, !(len as u16), "NLEN is not the complement of LEN");
            i += 5 + len;
            if hdr & 1 == 1 {
                break;
            }
        }
        assert_eq!(finals, 1);
        assert_eq!(i + 4, z.len(), "trailing bytes after the final block");
    }
}
