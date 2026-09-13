/**
 * WebGPU device acquisition, with a real diagnosis when it is unavailable.
 *
 * A blank canvas is the worst possible failure mode: the user cannot tell
 * whether the renderer crashed, the GPU is unsupported, or the page is still
 * loading. Each failure below carries a specific reason and a specific remedy.
 */

export type GpuInitResult =
  | { ok: true; adapter: GPUAdapter; device: GPUDevice; info: AdapterSummary }
  | { ok: false; reason: string; detail: string };

export interface AdapterSummary {
  vendor: string;
  architecture: string;
  description: string;
  maxStorageBufferBindingSize: number;
  maxBufferSize: number;
  maxComputeWorkgroupsPerDimension: number;
  hasTimestampQuery: boolean;
}

export async function initWebGPU(): Promise<GpuInitResult> {
  if (!('gpu' in navigator)) {
    return {
      ok: false,
      reason: 'This browser does not support WebGPU.',
      detail:
        'WebGPU ships in Chrome and Edge 113+, and in Safari 18+ on macOS 15 / iOS 18. ' +
        'In Firefox it is available from version 141 on Windows, and behind ' +
        'dom.webgpu.enabled elsewhere. A secure context (https, or localhost) is required.',
    };
  }

  let adapter: GPUAdapter | null = null;
  try {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  } catch (e) {
    return { ok: false, reason: 'Requesting a GPU adapter failed.', detail: String(e) };
  }
  if (!adapter) {
    return {
      ok: false,
      reason: 'No suitable GPU adapter was found.',
      detail:
        'The browser supports WebGPU but could not find a usable GPU. This usually means ' +
        'hardware acceleration is disabled, a driver is blocklisted, or the page is running ' +
        'in a virtual machine or remote desktop session without GPU passthrough.',
    };
  }

  // Timestamp queries are optional and often gated behind a browser flag, so
  // they are requested opportunistically and the profiler falls back to
  // CPU-side timing when they are absent.
  const wanted: GPUFeatureName[] = [];
  if (adapter.features.has('timestamp-query')) wanted.push('timestamp-query');

  let device: GPUDevice;
  try {
    device = await adapter.requestDevice({
      requiredFeatures: wanted,
      requiredLimits: {
        // Ask for the scene buffers we actually need rather than accepting the
        // default 128 MiB, so an over-large scene fails at init with a clear
        // message instead of at buffer creation.
        maxStorageBufferBindingSize: Math.min(
          adapter.limits.maxStorageBufferBindingSize,
          512 * 1024 * 1024,
        ),
        maxBufferSize: Math.min(adapter.limits.maxBufferSize, 512 * 1024 * 1024),
      },
    });
  } catch (e) {
    return { ok: false, reason: 'Could not create a GPU device.', detail: String(e) };
  }

  const raw = (adapter as GPUAdapter & { info?: GPUAdapterInfo }).info;
  return {
    ok: true,
    adapter,
    device,
    info: {
      vendor: raw?.vendor || 'unknown',
      architecture: raw?.architecture || 'unknown',
      description: raw?.description || raw?.device || '',
      maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
      maxBufferSize: device.limits.maxBufferSize,
      maxComputeWorkgroupsPerDimension: device.limits.maxComputeWorkgroupsPerDimension,
      hasTimestampQuery: device.features.has('timestamp-query'),
    },
  };
}

/**
 * Run `fn` inside a validation error scope and throw with the shader/pipeline
 * error if one occurs.
 *
 * Without this, a WGSL compile failure produces an invalid module, then an
 * invalid pipeline, and finally an error at the draw call — three layers from
 * the line of WGSL that is actually wrong.
 */
export async function captureErrors<T>(
  device: GPUDevice,
  label: string,
  fn: () => T,
): Promise<T> {
  device.pushErrorScope('validation');
  const result = fn();
  const err = await device.popErrorScope();
  if (err) throw new Error(`${label}: ${err.message}`);
  return result;
}

/** Compile a shader module and surface its compilation messages. */
export async function createShaderModule(
  device: GPUDevice,
  label: string,
  code: string,
): Promise<GPUShaderModule> {
  const module = device.createShaderModule({ label, code });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length > 0) {
    const lines = code.split('\n');
    const detail = errors
      .map((m) => {
        const src = lines[m.lineNum - 1] ?? '';
        return `  ${label}:${m.lineNum}:${m.linePos}  ${m.message}\n    ${src.trim()}`;
      })
      .join('\n');
    throw new Error(`WGSL compilation failed:\n${detail}`);
  }
  for (const m of info.messages.filter((m) => m.type === 'warning')) {
    console.warn(`${label}:${m.lineNum}:${m.linePos} ${m.message}`);
  }
  return module;
}
