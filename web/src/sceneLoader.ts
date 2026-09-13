/**
 * Fetch and slice a packed scene asset.
 *
 * Scenes ship as a single binary blob per scene — every array concatenated at
 * 16-byte alignment, with the offsets generated from Rust alongside it. The
 * browser therefore never lays out a scene struct and cannot disagree with the
 * WGSL about where a field lives; it only takes views over bytes Rust already
 * arranged.
 *
 * One fetch per scene rather than one per array: a Cornell box is 736 bytes and
 * a mesh scene is under a megabyte, so the round trip dominates and batching
 * matters more than granularity.
 */
import type { CameraDef, SceneManifest } from './generated/scenes';

export interface SceneData {
  name: string;
  description: string;
  background: [number, number, number];
  camera: CameraDef;

  materials: Uint8Array<ArrayBuffer>;
  /// Spheres and quads in one tagged array; see `GpuPrimitive` in Rust.
  primitives: Uint8Array<ArrayBuffer>;
  lights: Uint8Array<ArrayBuffer>;
  positions: Uint8Array<ArrayBuffer>;
  vertexAttrs: Uint8Array<ArrayBuffer>;
  triangles: Uint8Array<ArrayBuffer>;
  bvhNodes: Uint8Array<ArrayBuffer>;
  /// Equirectangular radiance, rgba32float. Empty when the scene has no sky.
  envRadiance: Uint8Array<ArrayBuffer>;
  /// Packed sampling distribution: rows 0..h-1 are each row's conditional CDF
  /// over columns, row h is the marginal over rows. Laid out in Rust so the
  /// browser and the native harness read the identical rectangle.
  envCdf: Uint8Array<ArrayBuffer>;
  numInstances: number;
  tlasRoot: number;
  /** Scene extent, for normalising the depth diagnostic. */
  depthScale: number;
  envWidth: number;
  envHeight: number;
  envTotalWeight: number;

  numPrimitives: number;
  numLights: number;
  numTriangles: number;
  numBvhNodes: number;
}

/** Element sizes, mirroring `crates/core/src/gpu_layout.rs`. */
const STRIDE = {
  materials: 80,
  primitives: 64,
  lights: 64,
  positions: 16,
  vertexAttrs: 32,
  triangles: 16,
  bvhNodes: 32,
  // rgba32float radiance, and r32float for the packed CDF rectangle.
  envRadiance: 16,
  envCdf: 4,
} as const;

const cache = new Map<string, Promise<SceneData>>();

export function loadScene(manifest: SceneManifest, baseUrl: string): Promise<SceneData> {
  const url = baseUrl + manifest.asset;
  let pending = cache.get(url);
  if (!pending) {
    pending = fetchScene(manifest, url);
    // Cache the promise, not the result, so two concurrent requests for the same
    // scene share one fetch rather than racing.
    cache.set(url, pending);
    pending.catch(() => cache.delete(url));
  }
  return pending;
}

async function fetchScene(manifest: SceneManifest, url: string): Promise<SceneData> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`fetching ${url}: ${res.status} ${res.statusText}`);
  }
  const buf = await res.arrayBuffer();

  // A stale asset paired with a fresh manifest would produce garbage geometry
  // that renders as noise rather than as an error. Catch it here instead.
  if (buf.byteLength !== manifest.byteLength) {
    throw new Error(
      `${url} is ${buf.byteLength} bytes, manifest expects ${manifest.byteLength}. ` +
        `Re-run: cargo run -p pt-cli --bin codegen`,
    );
  }

  const slice = (key: keyof typeof STRIDE): Uint8Array<ArrayBuffer> => {
    const s = manifest.sections[key];
    return new Uint8Array(buf, s.byteOffset, s.count * STRIDE[key]);
  };

  const sections = manifest.sections;
  return {
    name: manifest.name,
    description: manifest.description,
    background: manifest.background,
    camera: manifest.camera,
    materials: slice('materials'),
    primitives: slice('primitives'),
    lights: slice('lights'),
    positions: slice('positions'),
    vertexAttrs: slice('vertexAttrs'),
    triangles: slice('triangles'),
    bvhNodes: slice('bvhNodes'),
    envRadiance: slice('envRadiance'),
    envCdf: slice('envCdf'),
    envWidth: manifest.env.width,
    envHeight: manifest.env.height,
    envTotalWeight: manifest.env.totalWeight,
    // The **analytic** count, not the section's: instances are appended to the
    // same array (they overlay `GpuPrimitive` exactly), and the shader's
    // brute-force analytic loop must stop before them.
    numPrimitives: manifest.instancing.analyticPrimitives,
    numInstances: manifest.instancing.count,
    tlasRoot: manifest.instancing.tlasRoot,
    depthScale: manifest.depthScale,
    numLights: sections.lights.count,
    numTriangles: sections.triangles.count,
    numBvhNodes: sections.bvhNodes.count,
  };
}
