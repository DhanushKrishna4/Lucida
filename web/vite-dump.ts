import { mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import type { Plugin } from 'vite';

/**
 * Dev-only endpoint that writes a POSTed body to a file under `out/`.
 *
 * This closes the loop between the browser and the native test tooling. Without
 * it, comparing the browser's render against the CPU reference means triggering
 * a download, finding it, and moving it — every single time. With it, the
 * browser can hand its HDR framebuffer straight to:
 *
 *   cargo run --release -p pt-cli --bin compare -- out/cornell-cpu.pfm out/browser.pfm
 *
 * Only mounted by the dev server; `vite build` never sees it, so nothing is
 * exposed by the deployed static site.
 */
export function dumpPlugin(outDir: string): Plugin {
  const root = path.resolve(outDir);
  return {
    name: 'dev-dump',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use('/__dump', (req, res) => {
        if (req.method !== 'POST') {
          res.statusCode = 405;
          res.end('POST only');
          return;
        }
        // Take only the basename, so a crafted path cannot escape `out/`.
        const name = path.basename(decodeURIComponent((req.url ?? '/').slice(1)) || 'dump.bin');
        const chunks: Buffer[] = [];
        req.on('data', (c: Buffer) => chunks.push(c));
        req.on('end', () => {
          try {
            mkdirSync(root, { recursive: true });
            const file = path.join(root, name);
            writeFileSync(file, Buffer.concat(chunks));
            server.config.logger.info(`[dev-dump] wrote ${file} (${Buffer.concat(chunks).length} bytes)`);
            res.statusCode = 200;
            res.end(file);
          } catch (e) {
            res.statusCode = 500;
            res.end(String(e));
          }
        });
      });
    },
  };
}
