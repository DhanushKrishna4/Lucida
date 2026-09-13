/**
 * Fetch large scene assets once, then serve them from IndexedDB.
 *
 * # Why these are not just committed
 *
 * The BVH stress scene packs to 27 MB. Putting that in the repository would
 * charge every clone, every CI run and every Pages deploy for a file almost
 * nobody opens — and git stores it forever, so a later re-pack adds a second
 * copy rather than replacing the first. Codegen keeps anything over 2 MB out of
 * `web/public/` and writes it to a gitignored directory instead, to be uploaded
 * to a release the browser fetches from.
 *
 * # Content-addressed, which is what makes the cache correct
 *
 * A remote asset's filename carries a hash of its own contents
 * (`bvh-stress-5164eca50f3f3003.bin`), so re-packing a scene produces a *new*
 * filename rather than new bytes behind an old one. That single property does
 * all the cache-invalidation work: a stale entry can never be served for fresh
 * bytes, because the key the loader asks for no longer matches the key the
 * stale entry is under. There is no expiry to tune and no version to bump.
 *
 * It also means the hash does not need recomputing here. Verifying it in
 * JavaScript would need a 64-bit FNV-1a over 27 MB via 32-bit limbs — BigInt is
 * far too slow at that size — and it would catch nothing the two cheaper checks
 * miss: TLS covers authenticity, and the length check below covers the failure
 * that actually happens, which is a truncated transfer.
 */

const DB_NAME = 'pt-assets';
const DB_VERSION = 1;
const STORE = 'scenes';

export interface Progress {
  received: number;
  /** Zero when the server sends no `Content-Length`. */
  total: number;
}

/**
 * Open the asset database, or resolve to null if IndexedDB is unusable.
 *
 * Null rather than throwing: a private window, a storage-blocking setting or a
 * browser in a strange state should cost the user a re-download, not the scene.
 */
function openDb(): Promise<IDBDatabase | null> {
  return new Promise((resolve) => {
    let req: IDBOpenDBRequest;
    try {
      req = indexedDB.open(DB_NAME, DB_VERSION);
    } catch {
      resolve(null);
      return;
    }
    req.onupgradeneeded = () => {
      if (!req.result.objectStoreNames.contains(STORE)) {
        req.result.createObjectStore(STORE);
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => resolve(null);
    // A blocked open means another tab holds an older version. Do not hang the
    // scene load waiting for it to close.
    req.onblocked = () => resolve(null);
  });
}

function idb<T>(req: IDBRequest<T>): Promise<T | null> {
  return new Promise((resolve) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => resolve(null);
  });
}

async function readCached(key: string): Promise<ArrayBuffer | null> {
  const db = await openDb();
  if (!db) return null;
  try {
    const v = await idb<unknown>(db.transaction(STORE, 'readonly').objectStore(STORE).get(key));
    return v instanceof ArrayBuffer ? v : null;
  } finally {
    db.close();
  }
}

async function writeCached(key: string, buf: ArrayBuffer): Promise<boolean> {
  const db = await openDb();
  if (!db) return false;
  try {
    // Storing the buffer can fail on quota, and that is a normal outcome rather
    // than an error: the caller already has the bytes it needs.
    const ok = await new Promise<boolean>((resolve) => {
      const tx = db.transaction(STORE, 'readwrite');
      tx.objectStore(STORE).put(buf, key);
      tx.oncomplete = () => resolve(true);
      tx.onerror = () => resolve(false);
      tx.onabort = () => resolve(false);
    });
    return ok;
  } finally {
    db.close();
  }
}

/** Everything currently cached, as `[key, bytes]`. */
export async function cacheContents(): Promise<[string, number][]> {
  const db = await openDb();
  if (!db) return [];
  try {
    const store = db.transaction(STORE, 'readonly').objectStore(STORE);
    const keys = (await idb<IDBValidKey[]>(store.getAllKeys())) ?? [];
    const values = (await idb<unknown[]>(store.getAll())) ?? [];
    return keys.map((k, i) => [
      String(k),
      values[i] instanceof ArrayBuffer ? (values[i] as ArrayBuffer).byteLength : 0,
    ]);
  } finally {
    db.close();
  }
}

export async function clearCache(): Promise<void> {
  const db = await openDb();
  if (!db) return;
  try {
    await new Promise<void>((resolve) => {
      const tx = db.transaction(STORE, 'readwrite');
      tx.objectStore(STORE).clear();
      tx.oncomplete = () => resolve();
      tx.onerror = () => resolve();
      tx.onabort = () => resolve();
    });
  } finally {
    db.close();
  }
}

/**
 * Read the body with progress, because 27 MB on a slow link is long enough that
 * a UI saying nothing reads as a UI that has crashed.
 */
async function readWithProgress(
  res: Response,
  expectedBytes: number,
  onProgress?: (p: Progress) => void,
): Promise<ArrayBuffer> {
  // `Content-Length` is absent under some proxies and is the *compressed* length
  // when the response is encoded, so the manifest's figure is the honest total
  // and the header is only a fallback.
  const total = expectedBytes || Number(res.headers.get('content-length') ?? 0);
  if (!res.body) return res.arrayBuffer();

  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;
    onProgress?.({ received, total });
  }

  const out = new Uint8Array(received);
  let at = 0;
  for (const c of chunks) {
    out.set(c, at);
    at += c.byteLength;
  }
  return out.buffer;
}

/**
 * Fetch `url`, or return the cached copy stored under `key`.
 *
 * `expectedBytes` comes from the manifest and is checked on both paths — a
 * truncated download is the realistic failure here, and the geometry it would
 * produce renders as noise rather than as an error, so it has to be caught
 * before the bytes reach the renderer.
 */
export async function loadRemoteAsset(
  url: string,
  key: string,
  expectedBytes: number,
  onProgress?: (p: Progress & { cached: boolean }) => void,
): Promise<ArrayBuffer> {
  const hit = await readCached(key);
  if (hit && hit.byteLength === expectedBytes) {
    onProgress?.({ received: expectedBytes, total: expectedBytes, cached: true });
    return hit;
  }

  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(
      `fetching ${url}: ${res.status} ${res.statusText}. This scene is too large ` +
        `to ship with the site, so it is downloaded on demand — check that the ` +
        `release asset exists.`,
    );
  }
  const buf = await readWithProgress(res, expectedBytes, (p) =>
    onProgress?.({ ...p, cached: false }),
  );
  if (buf.byteLength !== expectedBytes) {
    throw new Error(
      `${url} returned ${buf.byteLength} bytes, manifest expects ${expectedBytes}. ` +
        `The download was truncated, or the published asset does not match this build.`,
    );
  }
  // Best effort. A failed write costs a re-download next time and nothing else.
  void writeCached(key, buf);
  return buf;
}
