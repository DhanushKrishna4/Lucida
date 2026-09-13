/**
 * Entry point: acquire a device, build the renderer, drive the progressive loop.
 *
 * Everything in the control panel is a knob on something the renderer actually
 * does, and every claim in a panel note is a number this project measured rather
 * than a figure from a paper. Where a control exposes a tradeoff rather than a
 * setting — MIS against BSDF sampling, megakernel against wavefront, denoising
 * against bias — the note says what it costs as well as what it buys, because a
 * renderer that only advertises its wins is not much use for learning from.
 */
import './style.css';
import { initWebGPU } from './webgpu';
import {
  Renderer,
  type Architecture,
  type RenderMode,
  type RenderSettings,
  type SamplingMode,
  type SamplerKind,
  type DiagnosticMode,
  type Tonemap,
} from './renderer';
import { REMOTE_SCENES, SCENES } from './generated/scenes';
import { loadScene, type SceneData, type SceneProgress } from './sceneLoader';
import { verifyCameraFixtures } from './camera';
import { samplesToReach } from './stats';
import { cacheContents, clearCache } from './assetCache';
import { length, sub } from './vec3';
import type { CameraDef } from './generated/scenes';
import { downloadBlob, encodePFM } from './pfm';
import { button, checkbox, group, note, select, slider, type SliderHandle } from './ui';
import { Controls } from './controls';

const RESOLUTIONS = [128, 192, 256, 384, 512, 768, 1024];

async function main(): Promise<void> {
  const canvas = document.getElementById('canvas') as HTMLCanvasElement;
  const panel = document.getElementById('panel') as HTMLElement;
  const overlay = document.getElementById('overlay') as HTMLElement;

  const init = await initWebGPU();
  if (!init.ok) {
    showUnsupported(init.reason, init.detail);
    return;
  }
  const { device, info } = init;

  // The camera basis is the only renderer math that exists in both Rust and
  // TypeScript. Check it against Rust's answers before rendering anything, so a
  // divergence is a named error rather than a mysteriously mis-framed image.
  const cameraProblems = verifyCameraFixtures();
  if (cameraProblems.length > 0) {
    console.error('Camera implementation disagrees with the Rust reference:\n' + cameraProblems.join('\n'));
  }

  let renderer: Renderer;
  try {
    // Scene geometry is a binary asset fetched at runtime, not baked into the
    // bundle: a mesh scene's packed buffers are most of a megabyte, and the BVH
    // stress scene is ten.
    const first = await loadScene(SCENES[0], import.meta.env.BASE_URL);
    renderer = await Renderer.create(device, canvas, first);
  } catch (e) {
    showUnsupported('The renderer failed to start.', String(e instanceof Error ? e.stack ?? e.message : e));
    return;
  }

  // Validation errors outside an explicit error scope are reported and then
  // *ignored* by WebGPU — the offending command buffer is dropped and the page
  // carries on looking almost right. Surface them, because "almost right" is the
  // hardest kind of rendering bug to notice.
  let reportedErrors = 0;
  device.addEventListener('uncapturederror', (e) => {
    const err = (e as GPUUncapturedErrorEvent).error;
    console.error('WebGPU validation error:', err.message);
    if (reportedErrors++ === 0) {
      const p = document.createElement('p');
      p.className = 'problem';
      p.textContent = `WebGPU validation error (see console):\n${err.message.split('\n')[0]}`;
      panel.prepend(p);
    }
  });

  // A lost device kills the context silently otherwise — the canvas simply stops
  // updating. Long compute dispatches are exactly what triggers it.
  device.lost.then((detail) => {
    if (detail.reason === 'destroyed') return;
    showUnsupported(
      'The GPU device was lost.',
      `${detail.message}\n\nThis is usually a driver timeout from a dispatch that ran too long. ` +
        `Lower the resolution or samples/frame and reload.`,
    );
  });

  // Camera controls. `setInteracting` is what lets the renderer drop internal
  // resolution while the camera is moving and restore it once the user settles.
  const controls = new Controls(canvas, SCENES[0].camera, {
    onChange: (cam) => renderer.setCamera(cam),
    onInteracting: (active) => renderer.setInteracting(active),
  });

  revealOnScroll();

  buildPanel(
    panel,
    canvas,
    renderer,
    controls,
    info.description || `${info.vendor} ${info.architecture}`,
    cameraProblems,
  );

  makeNotesFoldable(panel);

  // Expose the renderer for console poking during development.
  if (import.meta.env.DEV) {
    (globalThis as unknown as { pt: unknown }).pt = { renderer, device, info };
  }

  // The loop waits for each frame to complete on the GPU before submitting the
  // next. That bounds the queue to one frame in flight — which keeps the page
  // responsive, keeps dispatches short enough to stay clear of the driver
  // watchdog, and is the only way to measure GPU time without timestamp queries.
  try {
    for (;;) {
      await new Promise((r) => requestAnimationFrame(r));
      const added = renderer.renderFrame();
      await renderer.waitForGpu();
      // The registration marks warm while the estimate is still moving, so the
      // frame itself reports the state without another label.
      document.body.classList.toggle('is-converging', added > 0);
      overlay.innerHTML = formatStats(renderer, info);
    }
  } catch (e) {
    console.error(e);
    showUnsupported('The render loop stopped.', String(e instanceof Error ? e.stack ?? e.message : e));
  }
}

/** Relative standard error the "samples to reach" readout aims at. */
const QUALITY_TARGET = 0.01;

function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)} s`;
  const m = Math.floor(ms / 60_000);
  return `${m}m ${Math.round((ms % 60_000) / 1000)}s`;
}

function formatCount(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e4) return `${(n / 1e3).toFixed(1)}k`;
  return String(Math.round(n));
}

function formatStats(renderer: Renderer, info: { vendor: string; hasTimestampQuery: boolean }): string {
  const s = renderer.getStats();
  const set = renderer.getSettings();
  const geo = renderer.getScene();

  // `live` marks a measurement that is still moving. Amber is reserved for those
  // two, so a glance at the strip says what is being computed right now without
  // reading a word of it.
  const cells: string[] = [];
  const cell = (k: string, v: string, cls = 'dim') =>
    cells.push(`<div class="cell ${cls}"><span class="k">${k}</span><b class="v">${v}</b></div>`);

  cell('samples', String(s.samples) + (set.targetSamples > 0 ? ` / ${set.targetSamples}` : ''), 'live');

  if (s.convergence > 0) {
    const pct = s.convergence * 100;
    cell('noise', `${pct < 1 ? pct.toFixed(2) : pct.toFixed(1)}%`, 'live');
    const more = samplesToReach(s.convergence, s.samples, QUALITY_TARGET);
    if (more > 0) cell('to 1%', `+${formatCount(more)}`);
  } else {
    // A dash, not a zero: nothing measurable yet is not the same as no error.
    cell('noise', '\u2014', 'live');
  }

  if (s.accumMs > 0) {
    cell('elapsed', formatDuration(s.accumMs));
    if (set.targetSamples > 0 && s.samples < set.targetSamples) {
      cell('left', '~' + formatDuration((s.accumMs / s.samples) * (set.targetSamples - s.samples)));
    }
  }

  cell('frame', `${s.avgFrameMs.toFixed(1)} ms`);
  const mrays = s.maxRaysPerSecond / 1e6;
  // Prefixed with the bound symbol because it assumes every path runs to the
  // full bounce limit; see Stats.maxRaysPerSecond.
  cell('rays/s', '\u2264' + (mrays >= 1 ? `${mrays.toFixed(0)}M` : `${(s.maxRaysPerSecond / 1e3).toFixed(0)}K`));

  if (set.targetSamples > 0) {
    const pct = Math.min(100, (s.samples / set.targetSamples) * 100);
    cells.push(`<div class="bar"><span style="width:${pct.toFixed(1)}%"></span></div>`);
  }

  // The standing facts sit under the rule, away from the moving ones.
  const facts = [
    s.scale === 1 ? `${s.renderWidth}\u00d7${s.renderHeight}` : `${s.renderWidth}\u00d7${s.renderHeight} (1/${s.scale})`,
    geo.numTriangles > 0 ? `${geo.numTriangles.toLocaleString()} tris` : 'analytic',
    ...(geo.numTriangles > 0
      ? [renderer.isBvhEnabled() ? `${geo.numBvhNodes.toLocaleString()} bvh nodes` : 'bvh off']
      : []),
    `${geo.numLights} light${geo.numLights === 1 ? '' : 's'}`,
    `${(s.gpuBytes / (1024 * 1024)).toFixed(1)} MiB`,
    info.vendor,
    // Say which kind of timing this is rather than letting the two be confused.
    `CPU-side timing${info.hasTimestampQuery ? '' : ', no timestamp-query'}`,
  ];
  cells.push(`<div class="sub">${facts.join('&nbsp;&nbsp;\u00b7&nbsp;&nbsp;')}</div>`);

  return cells.join('');
}

/**
 * Fold the panel's explanatory notes until asked for.
 *
 * There is a great deal of genuine explanation in the panel and none of it
 * should compete with the control it annotates — but deleting it would lose the
 * part that makes the controls worth having.
 */
function makeNotesFoldable(panel: HTMLElement): void {
  for (const n of panel.querySelectorAll<HTMLElement>('.note')) {
    // The scene description is what the reader is looking at, not a note about
    // it, so it always stays open. Short notes are already glanceable and
    // folding them would add a click for nothing.
    if (n.id === 'scene-description') continue;
    if (n.textContent && n.textContent.length < 210) continue;
    n.classList.add('foldable', 'folded');
    n.title = 'Click to expand';
    n.addEventListener('click', () => {
      n.classList.toggle('folded');
      n.title = n.classList.contains('folded') ? 'Click to expand' : 'Click to collapse';
    });
  }
}

/** Reveal each essay band as it arrives, which suits a page about convergence. */
function revealOnScroll(): void {
  const bands = document.querySelectorAll('.band');
  if (!('IntersectionObserver' in window)) {
    bands.forEach((b) => b.classList.add('seen'));
    return;
  }
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          e.target.classList.add('seen');
          io.unobserve(e.target);
        }
      }
    },
    { rootMargin: '0px 0px -12% 0px' },
  );
  bands.forEach((b) => io.observe(b));
}


function buildPanel(
  panel: HTMLElement,
  canvas: HTMLCanvasElement,
  renderer: Renderer,
  controls: Controls,
  deviceName: string,
  cameraProblems: string[],
): void {
  const s = renderer.getSettings();
  const set = (patch: Partial<RenderSettings>) => renderer.setSettings(patch);

  // --- depth of field ---------------------------------------------------
  //
  // The lens controls are scene-relative. A focus distance is in scene units,
  // and this project's scenes disagree about those by three orders of magnitude
  // — the Cornell box is 555 units across because that is what the original
  // measurements were in millimetres, while the glass-ball scenes are about 5 —
  // so a fixed slider range would be unusable on all but one of them. Both
  // ranges are derived from the camera's distance to its target instead, and
  // re-derived whenever a scene loads.
  let apertureSlider: SliderHandle | undefined;
  let focusSlider: SliderHandle | undefined;
  let setAutofocusBox: ((v: boolean) => void) | undefined;

  const syncLens = (cam: CameraDef): void => {
    const d = Math.max(length(sub(cam.eye, cam.lookAt)), 1e-3);
    apertureSlider?.setRange(0, d / 6, d / 600);
    apertureSlider?.set(cam.aperture);
    focusSlider?.setRange(d / 20, d * 3, d / 500);
    focusSlider?.set(cam.focusDistance);
    setAutofocusBox?.(controls.autofocus);
  };

  // --- cached downloads -------------------------------------------------
  const cacheButton = button('Cached scenes: \u2014', () => {
    void clearCache().then(refreshCacheButton);
  });
  const refreshCacheButton = async (): Promise<void> => {
    const entries = await cacheContents();
    const bytes = entries.reduce((a, [, n]) => a + n, 0);
    cacheButton.textContent =
      entries.length === 0
        ? 'Cached scenes: none'
        : `Clear ${entries.length} cached scene${entries.length === 1 ? '' : 's'} (${(bytes / (1024 * 1024)).toFixed(1)} MB)`;
    cacheButton.disabled = entries.length === 0;
  };
  void refreshCacheButton();

  let armed = false;
  const focusLabel = 'Focus on a click';
  const focusButton = button(focusLabel, () => {
    armed = !armed;
    focusButton.textContent = armed ? 'Click the image\u2026 (Esc to cancel)' : focusLabel;
    canvas.style.cursor = armed ? 'crosshair' : 'grab';
  });
  const disarm = (): void => {
    armed = false;
    focusButton.textContent = focusLabel;
    canvas.style.cursor = 'grab';
  };

  // Capture phase on the document, not on the canvas: listeners on the *target*
  // element fire in registration order regardless of the capture flag, and the
  // orbit controls registered theirs first — so a capturing listener on the
  // canvas itself would still run second and the click would start a drag.
  document.addEventListener(
    'pointerdown',
    (e) => {
      if (!armed || e.target !== canvas) return;
      e.stopPropagation();
      e.preventDefault();
      disarm();
      const rect = canvas.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
      const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
      void renderer.probeDepth(x, y).then((depth) => {
        if (depth <= 0) {
          // A background ray has no depth to focus on, and silently doing
          // nothing would read as a broken button.
          focusButton.textContent = 'Nothing there \u2014 try again';
          window.setTimeout(() => {
            if (!armed) focusButton.textContent = focusLabel;
          }, 1600);
          return;
        }
        controls.setFocusDistance(depth);
        syncLens(controls.getCamera());
      });
    },
    { capture: true },
  );
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && armed) disarm();
  });

  panel.append(
    Object.assign(document.createElement('h1'), { textContent: 'Lucida' }),
    Object.assign(document.createElement('p'), {
      className: 'subtitle',
      textContent: deviceName,
    }),
  );

  if (cameraProblems.length > 0) {
    const p = document.createElement('p');
    p.className = 'problem';
    p.textContent =
      'Camera math disagrees with the Rust reference:\n' + cameraProblems.slice(0, 4).join('\n');
    panel.append(p);
  }

  panel.append(
    group(
      'render',
      select<RenderMode>(
        'Mode',
        [
          { value: 'pathtrace', label: 'Path trace' },
          { value: 'gradient', label: 'Gradient (pipeline smoke test)' },
        ],
        s.mode,
        (v) => set({ mode: v }),
      ),
      select(
        'Scene',
        // Remote scenes carry their size in the label. A 27 MB download should
        // announce itself before the click, not after it.
        SCENES.map((sc) => ({
          value: sc.name,
          label: sc.remote ? `${sc.name} (${sc.remote.megabytes} MB download)` : sc.name,
        })),
        SCENES[0].name,
        (v) => {
          const manifest = SCENES.find((x) => x.name === v);
          if (!manifest) return;
          const status = document.getElementById('scene-status');
          const show = (text: string) => {
            if (status) {
              status.textContent = text;
              status.hidden = text === '';
            }
          };
          const onProgress = (p: SceneProgress) => {
            if (p.cached) {
              show('loaded from cache');
              return;
            }
            const mb = (n: number) => (n / (1024 * 1024)).toFixed(1);
            show(
              p.total > 0
                ? `downloading ${mb(p.received)} / ${mb(p.total)} MB (${Math.round((p.received / p.total) * 100)}%)`
                : `downloading ${mb(p.received)} MB`,
            );
          };
          show(manifest.remote ? 'starting download\u2026' : '');
          void loadScene(manifest, import.meta.env.BASE_URL, onProgress)
            .then((scene: SceneData) => {
              renderer.setScene(scene);
              controls.setCamera(scene.camera);
              // A new scene brings its own units, so the lens sliders need new
              // ranges as well as new values.
              controls.autofocus = true;
              syncLens(scene.camera);
              const desc = document.getElementById('scene-description');
              if (desc) desc.textContent = scene.description;
              show('');
              void refreshCacheButton();
            })
            .catch((e: unknown) => {
              // A failed 27 MB download is the one scene load a user can
              // plausibly hit, so it says so in the panel rather than only in
              // the console.
              const msg = e instanceof Error ? e.message : String(e);
              console.error(`loading scene "${v}":`, e);
              show(`failed: ${msg}`);
            });
        },
      ),
      select(
        'Resolution',
        RESOLUTIONS.map((r) => ({ value: String(r), label: `${r} x ${r}` })),
        String(s.width),
        (v) => set({ width: Number(v), height: Number(v) }),
      ),
      Object.assign(note(SCENES[0].description), { id: 'scene-description' }),
      Object.assign(note(''), { id: 'scene-status', hidden: true }),
      ...(REMOTE_SCENES.length > 0
        ? [
            note(
              REMOTE_SCENES.map((r) => `<code>${r.name}</code> (${r.megabytes} MB)`).join(', ') +
                (REMOTE_SCENES.length === 1 ? ' is' : ' are') +
                ' too large to commit, so ' +
                (REMOTE_SCENES.length === 1 ? 'it is' : 'they are') +
                ' downloaded on demand and kept in <strong>IndexedDB</strong>. ' +
                'The filename carries a hash of the contents, which is what makes ' +
                'the cache correct without an expiry to tune: re-packing a scene ' +
                'produces a new filename rather than new bytes behind an old one, ' +
                'so a stale entry can never be served for fresh geometry.',
            ),
            cacheButton,
          ]
        : []),
    ),
    group(
      'sampling',
      select<SamplingMode>(
        'Light sampling',
        [
          { value: 'mis', label: 'MIS (power heuristic)' },
          { value: 'nee', label: 'Next event estimation only' },
          { value: 'bsdf', label: 'BSDF sampling only' },
        ],
        s.sampling,
        (v) => set({ sampling: v }),
      ),
      note(
        'All three estimate the same integral and converge to the same image — ' +
          'they differ only in variance. On the Cornell box, NEE is about 8x less ' +
          'noisy than BSDF sampling because a bounce wanders into that small ' +
          'ceiling light only a few percent of the time. On <code>mis-scene</code>, ' +
          'where neither strategy dominates, MIS beats BSDF sampling by 2.4x and ' +
          'NEE by 1.9x — it beats <em>both</em>, rather than landing between them. ' +
          'Switch between them at a low sample count to see it.',
      ),
      select<DiagnosticMode>(
        'Diagnostic',
        [
          { value: 'beauty', label: 'Beauty (the render)' },
          { value: 'normal', label: 'Normals' },
          { value: 'albedo', label: 'Albedo' },
          { value: 'depth', label: 'Depth' },
          { value: 'heat', label: 'BVH traversal heat' },
        ],
        s.diagnostic,
        (v) => set({ diagnostic: v }),
      ),
      note(
        'What the renderer is doing, rather than what the scene looks like. ' +
          'Almost every bug in this project produced an image that still looked ' +
          'like the scene — a transposed BVH count, a normal transformed by the ' +
          'matrix instead of its inverse-transpose — and a diagnostic is a ' +
          'reference that needs no second implementation. Normals, albedo and ' +
          'depth are free: they are the guide channels the denoiser already ' +
          'records. <strong>BVH traversal heat</strong> shows node visits per ' +
          'ray on a fixed scale, so two heatmaps are comparable; you can see the ' +
          'tree\u2019s own box structure in it. These bypass tone mapping, since ' +
          'a filmic curve on a normal map would make a diagnostic misreport its ' +
          'own values.',
      ),
      select<SamplerKind>(
        'Point set',
        [
          { value: 'sobol', label: 'Sobol (Owen-scrambled)' },
          { value: 'independent', label: 'Independent (random)' },
        ],
        s.sampler,
        (v) => set({ sampler: v }),
      ),
      note(
        'Where the samples land, as opposed to how they are weighted. Independent ' +
          'points clump by chance, and that clumping <em>is</em> the noise; a ' +
          'low-discrepancy sequence cannot clump, and Owen scrambling randomises ' +
          'it per pixel without destroying that. Measured here: about ' +
          '<strong>1.2-1.4x</strong> less noise at equal samples on the diffuse ' +
          'scenes. Not more, because the gain depends on the integrand being ' +
          'smooth — on <code>sunset</code>, whose variance is caustics, it is a ' +
          'wash. The sequence itself does far better than that on a smooth ' +
          'function; a path tracer just is not one.',
      ),
      slider({
        label: 'Denoise passes',
        min: 0,
        max: 6,
        value: s.denoisePasses,
        onInput: (v) => set({ denoisePasses: v }),
      }),
      note(
        'Zero is off, which is the default. An edge-avoiding à-trous filter, ' +
          'guided by the noise-free albedo, normal and depth of the first hit, ' +
          'run on the accumulation at display time — the render itself is never ' +
          'touched, so one more sample still converges to the unbiased answer. ' +
          'Measured against a converged reference on the Cornell box, it moves ' +
          'the <em>typical</em> pixel <strong>2.4x</strong> closer at 4 spp and ' +
          '1.4x at 16, then stops helping around 32 and actively hurts above it. ' +
          'Fireflies survive it: the worst 1% of pixels hold 98% of the squared ' +
          'error, and an edge-stopping filter sees an outlier as a different ' +
          'surface and refuses to blend it away. That needs a different ' +
          'technique, not a better filter.',
      ),
      slider({
        label: 'Max bounces',
        min: 1,
        max: 32,
        value: s.maxDepth,
        onInput: (v) => set({ maxDepth: v }),
      }),
      slider({
        label: 'Samples / frame',
        min: 1,
        max: 32,
        value: s.samplesPerFrame,
        onInput: (v) => set({ samplesPerFrame: v }),
      }),
      slider({
        label: 'Stop at',
        min: 0,
        max: 4096,
        step: 64,
        value: s.targetSamples,
        format: (v) => (v === 0 ? 'never' : `${v} spp`),
        onInput: (v) => set({ targetSamples: v }),
      }),
      note(
        'The <strong>noise</strong> figure in the overlay is measured, not ' +
          'modelled: every pixel accumulates the sum and the sum of squares of ' +
          'its per-sample estimates, and a reduction turns those into the mean ' +
          'relative standard error over the image. A <code>1/sqrt(N)</code> curve ' +
          'would have been free, and would be wrong exactly when anyone asks \u2014 ' +
          'it assumes a variance that a caustic or a firefly does not have. ' +
          'Checked against sixteen independent renders of the same scene, the ' +
          'reduction lands within 4%. With Sobol it reads about 1.3x ' +
          '<em>high</em>, because stratification beats the independent-sample ' +
          'formula it uses; of the two ways to be wrong, over-reporting noise is ' +
          'the one that does not tell you an image is finished when it is not.',
      ),
      note(
        '<strong>Stop at</strong> freezes the estimate once it reaches that many ' +
          'samples; the display pass keeps running, so exposure and tone map stay ' +
          'live on a finished image. Leave it at <em>never</em> to let it refine ' +
          'indefinitely.',
      ),
    ),
    group(
      'display',
      slider({
        label: 'Exposure',
        min: -4,
        max: 4,
        step: 0.1,
        value: 0,
        format: (v) => `${v > 0 ? '+' : ''}${v.toFixed(1)} EV`,
        // Exposure is a display-only change, so it must not reset accumulation.
        onInput: (v) => set({ exposure: Math.pow(2, v) }),
      }),
      select<Tonemap>(
        'Tone map',
        [
          { value: 'clamp', label: 'Clamp (no curve)' },
          { value: 'reinhard', label: 'Reinhard' },
          { value: 'aces', label: 'ACES (Hill fit)' },
          { value: 'agx', label: 'AgX' },
        ],
        s.tonemap,
        (v) => set({ tonemap: v }),
      ),
      note(
        'All four are display transforms only — they re-run a fragment shader, not ' +
          'the integrator, so a converged image stays converged while you scrub. ' +
          'They disagree about exposure by design: middle grey lands at sRGB 0.46 ' +
          '(clamp), 0.43 (Reinhard), 0.36 (ACES) and 0.50 (AgX).',
      ),
    ),
    group(
      'camera',
      slider({
        label: 'Frame budget',
        min: 8,
        max: 66,
        step: 1,
        value: s.targetFrameMs,
        format: (v) => `${v} ms`,
        onInput: (v) => set({ targetFrameMs: v }),
      }),
      note(
        'While the camera moves, internal resolution drops to whatever keeps the ' +
          'frame inside this budget, then returns to full once you let go. ' +
          'Drag to orbit, shift-drag or middle-drag to pan, scroll to dolly.',
      ),
      button('Reset camera', () => {
        const cam = renderer.getScene().camera;
        controls.setCamera(cam);
        controls.autofocus = true;
        renderer.setCamera(cam);
        syncLens(cam);
      }),
    ),
    group(
      'depth of field',
      slider({
        label: 'Aperture',
        min: 0,
        max: 1,
        step: 0.001,
        value: 0,
        format: (v) => (v <= 0 ? 'pinhole' : v.toFixed(2)),
        onInput: (v) => controls.setAperture(v),
        bind: (h) => {
          apertureSlider = h;
        },
      }),
      slider({
        label: 'Focus distance',
        min: 0,
        max: 1,
        step: 0.001,
        value: 0,
        format: (v) => v.toFixed(v < 10 ? 2 : 1),
        onInput: (v) => {
          controls.setFocusDistance(v);
          setAutofocusBox?.(false);
        },
        bind: (h) => {
          focusSlider = h;
        },
      }),
      focusButton,
      checkbox(
        'Auto-focus on the orbit target',
        true,
        (v) => {
          if (v) {
            controls.refocusOnTarget();
            syncLens(controls.getCamera());
          } else {
            controls.autofocus = false;
          }
        },
        (setter) => {
          setAutofocusBox = setter;
        },
      ),
      note(
        'A thin lens, not a pinhole: the ray origin is jittered across the ' +
          'aperture while it keeps aiming at the same point on the image plane, ' +
          'so everything at the focus distance stays sharp and everything else ' +
          'spreads into a circle of confusion. It costs nothing per sample \u2014 ' +
          'two extra random numbers \u2014 and it converges like any other ' +
          'integral, so a wide aperture is noisier at low sample counts and ' +
          'identical at high ones.',
      ),
      note(
        'Aperture is a diameter in <em>scene units</em> rather than an f-number, ' +
          'because an f-number is a ratio to a focal length and this renderer has ' +
          'no physical sensor size to derive one from \u2014 the Cornell box is ' +
          '555 units across because the original measurements were in ' +
          'millimetres, and calling that 555 mm would be a fiction. The slider ' +
          'range is scaled to each scene instead. <strong>Focus on a click</strong> ' +
          'reads the first-hit distance straight out of the accumulation buffer: ' +
          'the denoiser already stores it as a guide channel, so focusing costs ' +
          'one 32-byte readback of a number that was there anyway.',
      ),
    ),
    group(
      'architecture',
      select<Architecture>(
        'GPU architecture',
        [
          { value: 'megakernel', label: 'Megakernel (one kernel per path)' },
          { value: 'wavefront', label: 'Wavefront (one kernel per stage)' },
        ],
        s.architecture,
        (v) => set({ architecture: v }),
      ),
      note(
        'Same shading code either way — the two agree to 8e-9 mean relative ' +
          'error, which is float reassociation and nothing else. They differ in ' +
          'how the work is scheduled. The megakernel runs a whole path in one ' +
          'shader, so a warp keeps looping until its longest-lived path finishes ' +
          'and cost tracks the <em>bounce limit</em>: raising it from 1 to 32 ' +
          'costs 14x on <code>cornell-mesh</code>. The wavefront splits the path ' +
          'into six kernels and compacts dead paths out between them, so cost ' +
          'tracks how long paths actually live — 3.5x for the same sweep. ' +
          'Neither wins everywhere: the wavefront pays 80 bytes of path state per ' +
          'stage per bounce, so on a trivial scene it is several times ' +
          '<em>slower</em>. Raise max bounces and watch them cross over.',
      ),
    ),
    group(
      'acceleration',
      (() => {
        const b = button('BVH: on', () => {
          renderer.setBvhEnabled(!renderer.isBvhEnabled());
          b.textContent = renderer.isBvhEnabled() ? 'BVH: on' : 'BVH: OFF (brute force)';
        });
        return b;
      })(),
      note(
        'Turning the BVH off makes the shader test every triangle for every ray — ' +
          'the same reference path the correctness tests compare against. On a mesh ' +
          'scene the frame time difference is the whole argument for the data structure.',
      ),
    ),
    group(
      'export',
      button('Reset accumulation', () => renderer.reset()),
      button('Export HDR (.pfm)', async () => {
        const { width, height, data } = await renderer.readbackHDR();
        const bytes = encodePFM(width, height, data);
        downloadBlob(
          `browser-${width}x${height}-${renderer.getStats().samples}spp.pfm`,
          new Blob([bytes as BlobPart], { type: 'application/octet-stream' }),
        );
      }),
    // Dev-only: hand the HDR buffer straight to the native comparison tool.
    // In a production build `import.meta.env.DEV` is statically false, so this
    // branch and the endpoint it talks to are both absent.
    ...(import.meta.env.DEV
      ? [
          button('Send HDR to ./out (dev)', async () => {
            const { width, height, data } = await renderer.readbackHDR();
            const name = `browser-${width}x${height}-${renderer.getStats().samples}spp.pfm`;
            const res = await fetch(`/__dump/${name}`, {
              method: 'POST',
              body: new Blob([encodePFM(width, height, data) as BlobPart]),
            });
            console.log(res.ok ? `wrote ${await res.text()}` : `dump failed: ${res.status}`);
          }),
        ]
      : []),
      button('Export PNG', () => {
        const canvas = document.getElementById('canvas') as HTMLCanvasElement;
        canvas.toBlob((b) => {
          if (b) downloadBlob(`render-${renderer.getStats().samples}spp.png`, b);
        }, 'image/png');
      }),
      note(
        'The HDR export is linear float radiance, which is what the native tooling ' +
          'compares against:<br><code>cargo run --release -p pt-cli --bin compare -- ' +
          'out/cornell-cpu.pfm browser.pfm</code>',
      ),
    ),
  );

  // The lens sliders exist only once the panel above is built, and their ranges
  // depend on the scene that is already loaded.
  syncLens(controls.getCamera());
}

function showUnsupported(reason: string, detail: string): void {
  const app = document.getElementById('app');
  const box = document.getElementById('unsupported');
  if (!app || !box) return;
  app.hidden = true;
  box.hidden = false;
  box.innerHTML = '';

  const h = document.createElement('h1');
  h.textContent = reason;
  const p = document.createElement('p');
  p.textContent =
    'This project renders entirely in WebGPU compute shaders, so there is no ' +
    'fallback renderer — but here is what it looks like when it runs.';
  const d = document.createElement('p');
  d.className = 'detail';
  d.textContent = detail;

  const img = document.createElement('img');
  img.src = `${import.meta.env.BASE_URL}preview.png`;
  img.alt = 'A Cornell box rendered by this path tracer: red wall on the left, green on the right, two spheres lit by a ceiling area light.';
  // A broken image icon would be worse than no image at all.
  img.addEventListener('error', () => img.remove());

  box.append(h, p, d, img);
}

main().catch((e) => {
  console.error(e);
  showUnsupported('Failed to start.', String(e instanceof Error ? e.stack ?? e.message : e));
});
