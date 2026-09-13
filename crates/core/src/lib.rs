//! `pt-core` — scene representation, BSDFs, and the CPU reference path tracer.
//!
//! This crate has **no GPU dependencies**. It is the correctness oracle for
//! everything that runs on the GPU: the same scene, the same RNG algorithm, the
//! same sampling order, evaluated on the CPU in plain `f32`.
//!
//! It is also the single source of truth for GPU buffer *layout* (see
//! [`gpu_layout`]). The WGSL struct declarations and the TypeScript-side scene
//! blobs are generated from the types in that module, so a layout change can
//! only ever happen in one place.

pub mod bsdf;
pub mod bvh;
pub mod bvh4;
pub mod camera;
pub mod chi2;
pub mod denoise;
pub mod diagnostic;
pub mod envmap;
pub mod gpu_layout;
pub mod image;
pub mod instance;
pub mod integrator;
pub mod lbvh;
pub mod light;
pub mod math;
pub mod mesh;
pub mod rng;
pub mod scene;
pub mod scenes;
pub mod sobol;
pub mod tonemap;

pub use glam::{Mat4, Vec2, Vec3, Vec4};
