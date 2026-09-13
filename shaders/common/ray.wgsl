// The ray type, shared by every stage.
//
// Its own file because the wavefront kernels include different subsets of the
// shared code, and more than one of those subsets needs `Ray`.

struct Ray {
  origin: vec3<f32>,
  dir: vec3<f32>,
};
