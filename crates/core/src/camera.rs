//! Perspective camera with a thin-lens model.
//!
//! The camera is resolved once on the CPU into the "corner + spanning vectors"
//! form stored in [`GpuUniforms`], and every consumer (CPU tracer, WGSL kernel,
//! TypeScript host) then generates rays with the *same* two multiply-adds. No
//! projection matrix, no inverse, no handedness debate.

use crate::gpu_layout::GpuUniforms;
use crate::math::concentric_sample_disk;
use crate::scene::Ray;
use glam::{Vec2, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub eye: Vec3,
    pub look_at: Vec3,
    pub up: Vec3,
    /// Vertical field of view, in degrees.
    pub vfov_deg: f32,
    /// Lens radius. Zero gives a pinhole camera (everything in focus).
    pub aperture: f32,
    /// Distance to the plane of perfect focus. Also the distance at which the
    /// image plane is placed, which is what makes the thin-lens jitter below
    /// leave the focal plane stationary.
    pub focus_distance: f32,
}

impl Camera {
    pub fn look_at(eye: Vec3, look_at: Vec3, vfov_deg: f32) -> Self {
        Self {
            eye,
            look_at,
            up: Vec3::Y,
            vfov_deg,
            aperture: 0.0,
            focus_distance: (look_at - eye).length(),
        }
    }

    /// Right-handed view basis: `(right, down, forward)`.
    ///
    /// `down` rather than `up` because image row 0 is the top row; see
    /// [`GpuUniforms::cam_upper_left`].
    pub fn basis(&self) -> (Vec3, Vec3, Vec3) {
        let forward = (self.look_at - self.eye).normalize();
        let right = forward.cross(self.up).normalize();
        // Re-derive the vertical axis from the orthogonalised right vector so
        // the basis stays orthonormal even when `up` is not perpendicular to
        // `forward` (which it generally will not be once orbit controls exist).
        let up = right.cross(forward);
        (right, -up, forward)
    }

    /// Bake into the GPU uniform form. `aspect` is width / height.
    pub fn write_uniforms(&self, u: &mut GpuUniforms, aspect: f32) {
        let (right, down, forward) = self.basis();

        // Place the image plane at the focus distance. With the plane there,
        // displacing the ray origin on the lens and re-aiming at the same image
        // -plane point keeps points at that distance perfectly sharp — that is
        // the entire thin-lens model.
        let half_height = (self.vfov_deg.to_radians() * 0.5).tan() * self.focus_distance;
        let half_width = half_height * aspect;

        let horizontal = right * (2.0 * half_width);
        let vertical = down * (2.0 * half_height);
        let upper_left =
            self.eye + forward * self.focus_distance - right * half_width - down * half_height;

        u.cam_origin = self.eye.to_array();
        u.cam_upper_left = upper_left.to_array();
        u.cam_horizontal = horizontal.to_array();
        u.cam_vertical = vertical.to_array();
        u.lens_radius = self.aperture * 0.5;
        u.focus_distance = self.focus_distance;
    }
}

/// Generate the primary ray for pixel `(x, y)` with sub-pixel jitter `pixel_uv`
/// in [0,1)^2 and lens sample `lens_uv` in [0,1)^2.
///
/// This function is mirrored exactly in `shaders/trace/megakernel.wgsl`.
#[inline]
pub fn generate_ray(u: &GpuUniforms, x: u32, y: u32, pixel_uv: Vec2, lens_uv: Vec2) -> Ray {
    let s = (x as f32 + pixel_uv.x) / u.width as f32;
    let t = (y as f32 + pixel_uv.y) / u.height as f32;

    let origin = Vec3::from_array(u.cam_origin);
    let target = Vec3::from_array(u.cam_upper_left)
        + s * Vec3::from_array(u.cam_horizontal)
        + t * Vec3::from_array(u.cam_vertical);

    if u.lens_radius <= 0.0 {
        return Ray {
            origin,
            dir: (target - origin).normalize(),
        };
    }

    // Thin lens: jitter the origin over the aperture disk, but keep aiming at
    // the same point on the focal plane. Points at `focus_distance` are hit by
    // every lens sample and stay sharp; everything else smears by an amount
    // proportional to its defocus.
    let d = concentric_sample_disk(lens_uv) * u.lens_radius;
    let (right, down, _) = (
        Vec3::from_array(u.cam_horizontal).normalize(),
        Vec3::from_array(u.cam_vertical).normalize(),
        Vec3::ZERO,
    );
    let offset = right * d.x + down * d.y;
    let origin = origin + offset;
    Ray {
        origin,
        dir: (target - origin).normalize(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cornell_cam() -> (Camera, GpuUniforms) {
        let cam = Camera::look_at(
            Vec3::new(278.0, 278.0, -800.0),
            Vec3::new(278.0, 278.0, 0.0),
            37.0,
        );
        let mut u = GpuUniforms {
            width: 512,
            height: 512,
            ..Default::default()
        };
        cam.write_uniforms(&mut u, 1.0);
        (cam, u)
    }

    #[test]
    fn basis_is_orthonormal_and_right_handed() {
        let (cam, _) = cornell_cam();
        let (r, d, f) = cam.basis();
        for v in [r, d, f] {
            assert!((v.length() - 1.0).abs() < 1e-6);
        }
        assert!(r.dot(d).abs() < 1e-6);
        assert!(r.dot(f).abs() < 1e-6);
        assert!(d.dot(f).abs() < 1e-6);
        // Looking down +z with world up +y, screen-right is world -x. This is
        // the standard Cornell box orientation: the x = 555 wall (red) appears
        // on the left of the image.
        assert!(
            (r - Vec3::new(-1.0, 0.0, 0.0)).length() < 1e-6,
            "right = {r}"
        );
        assert!(
            (d - Vec3::new(0.0, -1.0, 0.0)).length() < 1e-6,
            "down = {d}"
        );
    }

    #[test]
    fn centre_pixel_looks_at_the_target() {
        let (cam, u) = cornell_cam();
        let ray = generate_ray(&u, 256, 256, Vec2::new(0.0, 0.0), Vec2::ZERO);
        assert!((ray.origin - cam.eye).length() < 1e-4);
        assert!((ray.dir - Vec3::Z).length() < 1e-3, "dir = {}", ray.dir);
    }

    /// Row 0 must be the top of the image, i.e. its rays point upward in world
    /// space. Getting this backwards produces a vertically flipped render that
    /// is easy to miss in a symmetric scene like the Cornell box.
    #[test]
    fn row_zero_is_the_top() {
        let (_, u) = cornell_cam();
        let top = generate_ray(&u, 256, 0, Vec2::splat(0.5), Vec2::ZERO);
        let bottom = generate_ray(&u, 256, 511, Vec2::splat(0.5), Vec2::ZERO);
        assert!(top.dir.y > 0.0, "top row should look up, got {}", top.dir);
        assert!(
            bottom.dir.y < 0.0,
            "bottom row should look down, got {}",
            bottom.dir
        );
    }

    /// Column 0 is the left of the image, which for this camera is world +x.
    #[test]
    fn column_zero_is_world_plus_x() {
        let (_, u) = cornell_cam();
        let left = generate_ray(&u, 0, 256, Vec2::splat(0.5), Vec2::ZERO);
        assert!(
            left.dir.x > 0.0,
            "left column should look toward +x, got {}",
            left.dir
        );
    }

    /// The configured vertical FOV must be exactly the angle subtended between
    /// the top-centre and bottom-centre rays.
    #[test]
    fn vertical_fov_is_respected() {
        let (_, u) = cornell_cam();
        let top = generate_ray(&u, 256, 0, Vec2::new(0.5, 0.0), Vec2::ZERO);
        let bottom = generate_ray(&u, 256, 511, Vec2::new(0.5, 1.0), Vec2::ZERO);
        let angle = top.dir.dot(bottom.dir).clamp(-1.0, 1.0).acos().to_degrees();
        assert!((angle - 37.0).abs() < 0.1, "fov = {angle}");
    }

    /// Every lens sample must pass through the same point on the focal plane.
    #[test]
    fn thin_lens_keeps_the_focal_plane_sharp() {
        let mut cam = Camera::look_at(Vec3::new(0.0, 0.0, -10.0), Vec3::ZERO, 45.0);
        cam.aperture = 2.0;
        let mut u = GpuUniforms {
            width: 64,
            height: 64,
            ..Default::default()
        };
        cam.write_uniforms(&mut u, 1.0);

        let reference = generate_ray(&u, 20, 40, Vec2::splat(0.5), Vec2::splat(0.5));
        let focal_point = reference.origin
            + reference.dir * (cam.focus_distance / reference.dir.dot(cam.basis().2));

        for (lx, ly) in [(0.1, 0.9), (0.7, 0.2), (0.33, 0.66), (0.99, 0.01)] {
            let r = generate_ray(&u, 20, 40, Vec2::splat(0.5), Vec2::new(lx, ly));
            let t = (focal_point - r.origin).dot(cam.basis().2) / r.dir.dot(cam.basis().2);
            let p = r.origin + r.dir * t;
            assert!(
                (p - focal_point).length() < 1e-3,
                "lens sample missed focus by {}",
                (p - focal_point).length()
            );
        }
    }
}
