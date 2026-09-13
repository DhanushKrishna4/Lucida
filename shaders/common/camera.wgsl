// Primary ray generation. Mirrors `generate_ray` in crates/core/src/camera.rs.
//
// The camera arrives already resolved into corner + spanning vectors, so this is
// two multiply-adds. `cam_vertical` points *down*, which is why there is no
// vertical flip here: image row 0 is the top row by construction.

fn generate_ray(x: u32, y: u32, pixel_uv: vec2<f32>, lens_uv: vec2<f32>) -> Ray {
  let s = (f32(x) + pixel_uv.x) / f32(U.width);
  let t = (f32(y) + pixel_uv.y) / f32(U.height);

  // 'target' is a WGSL reserved keyword.
  let image_point = U.cam_upper_left + s * U.cam_horizontal + t * U.cam_vertical;

  var r: Ray;
  if (U.lens_radius <= 0.0) {
    r.origin = U.cam_origin;
    r.dir = normalize(image_point - U.cam_origin);
    return r;
  }

  // Thin lens: jitter the origin across the aperture but keep aiming at the same
  // point on the focal plane, so that plane stays sharp and everything else
  // blurs in proportion to its defocus.
  let d = concentric_sample_disk(lens_uv) * U.lens_radius;
  let right = normalize(U.cam_horizontal);
  let down = normalize(U.cam_vertical);
  r.origin = U.cam_origin + right * d.x + down * d.y;
  r.dir = normalize(image_point - r.origin);
  return r;
}
