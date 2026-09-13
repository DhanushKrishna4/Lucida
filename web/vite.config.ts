import { defineConfig } from 'vite';
import path from 'node:path';
import { wgslPlugin } from './vite-wgsl';
import { dumpPlugin } from './vite-dump';

const shaderRoot = path.resolve(__dirname, '../shaders');

export default defineConfig({
  // GitHub Pages serves the project from a repository subpath. Overridable so a
  // fork or a custom domain does not need a code change.
  base: process.env.PT_BASE ?? '/',
  define: {
    // Optional third-party host for the scenes too large to commit.
    //
    // Empty by default, which means "serve them from the site like everything
    // else" — GitHub Pages is itself CDN-backed, the asset is same-origin so
    // there is no CORS question, and the blob still never enters git: codegen
    // writes it under a gitignored content-addressed name and the Pages
    // workflow pulls it from the release into the built site.
    //
    // GitHub *release* assets are deliberately not the default, despite being
    // the obvious choice: they serve no `access-control-allow-origin`, so a
    // browser `fetch` of one fails with an opaque `TypeError`. Verified rather
    // than assumed, after building against them first.
    //
    // Set PT_ASSET_BASE to point at a real CDN (it must send CORS headers); the
    // loader appends `remote.file` to it verbatim.
    __ASSET_BASE__: JSON.stringify(process.env.PT_ASSET_BASE ?? ''),
  },
  plugins: [wgslPlugin(shaderRoot), dumpPlugin(path.resolve(__dirname, '../out'))],
  server: { fs: { allow: ['..'] } },
  build: { target: 'es2022', sourcemap: true },
});
