import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import type { Plugin } from 'vite';

/**
 * Resolves `//!include "path"` directives in WGSL and exposes each shader as a
 * default-exported string.
 *
 * WGSL has no module system, so shared code has to be textually spliced. The
 * native harness implements the *identical* directive in
 * `crates/gpu/src/shaders.rs`, so the browser and the native renderer compile
 * byte-identical shader source — which means a difference between them is a
 * platform difference, never a source difference.
 *
 * Include-once is mandatory rather than an optimisation: WGSL rejects duplicate
 * function and struct definitions, and the include graph is a diamond
 * (megakernel -> {scene, camera} -> math).
 */
export function wgslPlugin(shaderRoot: string): Plugin {
  const root = path.resolve(shaderRoot);

  function resolve(absPath: string, seen: Set<string>, deps: Set<string>, depth = 0): string {
    if (depth > 16) throw new Error(`include depth exceeded at ${absPath} — is there a cycle?`);
    const key = path.normalize(absPath);
    if (seen.has(key)) return '';
    seen.add(key);
    if (!existsSync(key)) throw new Error(`no such shader: ${key}`);
    deps.add(key);

    const out: string[] = [];
    for (const line of readFileSync(key, 'utf8').split('\n')) {
      const m = /^\s*\/\/!include\s+"([^"]+)"\s*$/.exec(line);
      if (m) {
        const inc = path.join(root, m[1]);
        out.push(`// ---- begin ${m[1]} (included by ${path.relative(root, key)}) ----`);
        out.push(resolve(inc, seen, deps, depth + 1));
        out.push(`// ---- end ${m[1]} ----`);
      } else {
        out.push(line);
      }
    }
    return out.join('\n');
  }

  return {
    name: 'wgsl-include',
    load(id) {
      const [file] = id.split('?');
      if (!file.endsWith('.wgsl')) return null;
      const deps = new Set<string>();
      const source = resolve(file, new Set(), deps);
      // Watch every included file, not just the entry, so editing math.wgsl
      // reloads the shaders that include it.
      for (const d of deps) this.addWatchFile(d);
      return `export default ${JSON.stringify(source)};`;
    },
  };
}
