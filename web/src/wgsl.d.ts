declare module '*.wgsl' {
  /** WGSL source with `//!include` directives already resolved. */
  const source: string;
  export default source;
}
