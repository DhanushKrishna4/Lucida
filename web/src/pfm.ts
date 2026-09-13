/**
 * PFM (Portable FloatMap) export.
 *
 * This is the bridge between the browser and the native test tooling: export the
 * accumulated HDR buffer here, then run
 *
 *   cargo run --release -p pt-cli --bin compare -- out/cornell-cpu.pfm browser.pfm
 *
 * to check the browser against the CPU reference at the same precision the
 * native GPU comparison uses. A PNG would not do — it is 8-bit and tone mapped,
 * so it discards exactly the information the comparison depends on.
 *
 * Mirrors `write_pfm` in `crates/core/src/image.rs`.
 */

/**
 * @param data linear RGB triples, row-major, **row 0 is the top row**
 */
export function encodePFM(
  width: number,
  height: number,
  data: Float32Array,
): Uint8Array<ArrayBuffer> {
  if (data.length !== width * height * 3) {
    throw new Error(`expected ${width * height * 3} floats, got ${data.length}`);
  }
  // A negative scale declares little-endian samples.
  const header = new TextEncoder().encode(`PF\n${width} ${height}\n-1.0\n`);
  const out = new Uint8Array(header.length + width * height * 3 * 4);
  out.set(header, 0);

  const view = new DataView(out.buffer, header.length);
  let o = 0;
  // PFM stores rows bottom-to-top, while our film has row 0 at the top.
  for (let y = height - 1; y >= 0; y--) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 3;
      view.setFloat32(o, data[i], true);
      view.setFloat32(o + 4, data[i + 1], true);
      view.setFloat32(o + 8, data[i + 2], true);
      o += 12;
    }
  }
  return out;
}

export function downloadBlob(filename: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  // Revoking immediately can cancel the download in some browsers; one turn of
  // the event loop is enough for the navigation to be picked up.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
