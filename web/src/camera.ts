/**
 * Camera resolution — the one piece of renderer math that genuinely exists in
 * two languages.
 *
 * Everything else the host touches is either pre-packed by codegen (scene
 * buffers) or trivially copied (scalar uniforms). The camera is different: orbit
 * controls need to recompute the basis live, on the host, every frame. So this
 * file mirrors `Camera::write_uniforms` in `crates/core/src/camera.rs`.
 *
 * Because it is duplicated, it is checked: `verifyCameraFixtures()` compares
 * this implementation against Rust's answers for several configurations
 * (emitted by codegen) at startup. A drift in handedness, a flipped vertical
 * axis, or a mis-derived FOV surfaces as a console error immediately rather than
 * as a render that is subtly mis-framed against the reference images.
 */
import { CAMERA_FIXTURES } from './generated/scenes';
import type { CameraDef } from './generated/scenes';
import { add, cross, dot, normalize, scale, sub, type Vec3 } from './vec3';

export interface ResolvedCamera {
  camOrigin: Vec3;
  /** Image-plane corner for pixel (0, 0) — the TOP-left, since vertical points down. */
  camUpperLeft: Vec3;
  camHorizontal: Vec3;
  /** Spans the image height pointing DOWN, so image row 0 is the top row. */
  camVertical: Vec3;
  lensRadius: number;
  focusDistance: number;
}

/** Right-handed view basis: `[right, down, forward]`. */
export function cameraBasis(cam: CameraDef): [Vec3, Vec3, Vec3] {
  const forward = normalize(sub(cam.lookAt, cam.eye));
  const right = normalize(cross(forward, cam.up));
  // Re-derive the vertical axis from the orthogonalised right vector, so the
  // basis stays orthonormal even when `up` is not perpendicular to `forward`.
  const up = cross(right, forward);
  return [right, scale(up, -1), forward];
}

export function resolveCamera(cam: CameraDef, aspect: number): ResolvedCamera {
  const [right, down, forward] = cameraBasis(cam);

  // The image plane sits at the focus distance. That is what makes the thin-lens
  // model work: displacing the origin on the lens and re-aiming at the same
  // image-plane point leaves points at that distance perfectly sharp.
  const halfHeight = Math.tan((cam.vfovDeg * Math.PI) / 360) * cam.focusDistance;
  const halfWidth = halfHeight * aspect;

  const camHorizontal = scale(right, 2 * halfWidth);
  const camVertical = scale(down, 2 * halfHeight);
  const camUpperLeft = sub(
    sub(add(cam.eye, scale(forward, cam.focusDistance)), scale(right, halfWidth)),
    scale(down, halfHeight),
  );

  return {
    camOrigin: [...cam.eye] as Vec3,
    camUpperLeft,
    camHorizontal,
    camVertical,
    lensRadius: cam.aperture * 0.5,
    focusDistance: cam.focusDistance,
  };
}

/** Orbit the camera around its look-at point. Angles in radians. */
export function orbit(cam: CameraDef, dYaw: number, dPitch: number): CameraDef {
  const offset = sub(cam.eye, cam.lookAt);
  const radius = Math.sqrt(dot(offset, offset));
  let pitch = Math.asin(Math.max(-1, Math.min(1, offset[1] / radius)));
  let yaw = Math.atan2(offset[0], offset[2]);
  yaw += dYaw;
  // Clamp short of the poles: at exactly +/-90 degrees the view direction is
  // parallel to `up` and the basis degenerates.
  const limit = Math.PI / 2 - 1e-3;
  pitch = Math.max(-limit, Math.min(limit, pitch + dPitch));
  const c = Math.cos(pitch);
  const eye: Vec3 = [
    cam.lookAt[0] + radius * c * Math.sin(yaw),
    cam.lookAt[1] + radius * Math.sin(pitch),
    cam.lookAt[2] + radius * c * Math.cos(yaw),
  ];
  return { ...cam, eye, focusDistance: radius };
}

/**
 * Check this implementation against Rust's, using fixtures emitted by codegen.
 *
 * Returns a list of human-readable problems; empty means agreement. The
 * tolerance is loose enough for f32-vs-f64 rounding (the fixtures are printed
 * from `f32`) and far tighter than any real bug would be.
 */
/**
 * Check this implementation against Rust's, using fixtures emitted by codegen.
 *
 * Returns a list of human-readable problems; empty means agreement. Each fixture
 * carries its own camera, so this tests the *math* and is independent of which
 * scenes happen to ship to the browser.
 *
 * The tolerance is relative, because these are world-space values in the
 * hundreds — a fixed epsilon would be either meaningless or impossibly strict —
 * and loose enough for the f32-versus-f64 rounding introduced by printing the
 * fixtures from `f32`.
 */
export function verifyCameraFixtures(): string[] {
  const problems: string[] = [];
  for (const f of CAMERA_FIXTURES) {
    const got = resolveCamera(f.camera, f.width / f.height);
    const check = (name: string, a: Vec3, b: readonly number[]) => {
      for (let i = 0; i < 3; i++) {
        const tol = 1e-4 * Math.max(1, Math.abs(b[i]));
        if (Math.abs(a[i] - b[i]) > tol) {
          problems.push(
            `${f.label} ${f.width}x${f.height}: ${name}[${i}] = ${a[i]}, Rust says ${b[i]}`,
          );
        }
      }
    };
    check('camOrigin', got.camOrigin, f.camOrigin);
    check('camUpperLeft', got.camUpperLeft, f.camUpperLeft);
    check('camHorizontal', got.camHorizontal, f.camHorizontal);
    check('camVertical', got.camVertical, f.camVertical);
  }
  if (CAMERA_FIXTURES.length === 0) {
    problems.push('no camera fixtures were generated — run: cargo run -p pt-cli --bin codegen');
  }
  return problems;
}
