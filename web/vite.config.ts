import { defineConfig } from 'vite';
import path from 'node:path';
import { wgslPlugin } from './vite-wgsl';
import { dumpPlugin } from './vite-dump';

const shaderRoot = path.resolve(__dirname, '../shaders');

export default defineConfig({
  // GitHub Pages serves the project from a repository subpath. Overridable so a
  // fork or a custom domain does not need a code change.
  base: process.env.PT_BASE ?? '/',
  plugins: [wgslPlugin(shaderRoot), dumpPlugin(path.resolve(__dirname, '../out'))],
  server: { fs: { allow: ['..'] } },
  build: { target: 'es2022', sourcemap: true },
});
