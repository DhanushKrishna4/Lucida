// RESET: roll the queues forward between bounces.
//
// A single invocation, and the only kernel that writes the indirect dispatch
// arguments. It can, because it is dispatched *directly* — WebGPU forbids a
// buffer being both a read-write binding and the indirect source within one
// dispatch, so no stage that dispatches indirectly may also size the next one.
//
// It runs between SHADE and CONNECT rather than at the end of the bounce, so
// CONNECT can be sized from the shadow rays SHADE just appended.

//!include "common/generated.wgsl"

@group(0) @binding(0) var<uniform> U: Uniforms;

@group(1) @binding(0) var<storage, read_write> counters: WavefrontCounters;
@group(1) @binding(1) var<storage, read_write> args: DispatchArgs;
@group(1) @binding(2) var<storage, read_write> queue_out: array<u32>;

const INVALID_PATH: u32 = 0xFFFFFFFFu;
const WG: u32 = 64u;

@compute @workgroup_size(1, 1, 1)
fn main() {
  let live = atomicLoad(&counters.next_queue_len);
  let groups = atomicLoad(&counters.next_trace_x);
  let shadows = atomicLoad(&counters.shadow_len);

  // Sentinel-terminate the tail of the last partial workgroup. EXTEND reads this
  // instead of a queue length, which saves that kernel one of the eight storage
  // buffers it has no room for; the cost is at most 63 writes from one thread.
  let end = groups * WG;
  for (var i = live; i < end; i = i + 1u) {
    if (i < arrayLength(&queue_out)) {
      queue_out[i] = INVALID_PATH;
    }
  }

  // Size the next bounce, and this bounce's shadow pass.
  args.trace_x = groups;
  args.trace_y = 1u;
  args.trace_z = 1u;
  args.shadow_x = (shadows + WG - 1u) / WG;
  args.shadow_y = 1u;
  args.shadow_z = 1u;
  counters.shadow_count = shadows;

  atomicStore(&counters.next_queue_len, 0u);
  atomicStore(&counters.next_trace_x, 0u);
  atomicStore(&counters.shadow_len, 0u);
}
