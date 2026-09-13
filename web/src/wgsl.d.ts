declare module '*.wgsl' {
  /** WGSL source with `//!include` directives already resolved. */
  const source: string;
  export default source;
}

/** Injected by `vite.config.ts`: base URL for scenes fetched from the CDN. */
declare const __ASSET_BASE__: string;
