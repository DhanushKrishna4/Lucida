/**
 * Orbit / pan / dolly camera controls.
 *
 * Deliberately not a generic input library: the renderer needs to know not just
 * that the camera moved but *that it is still moving*, so it can hold a lower
 * internal resolution during interaction and restore full quality once the user
 * settles. That "interaction ended" signal is the reason this owns its own
 * event handling rather than just emitting camera values.
 */
import { cameraBasis, orbit } from './camera';
import type { CameraDef } from './generated/scenes';
import { add, length, scale, sub, type Vec3 } from './vec3';

export interface ControlsOptions {
  /** Called whenever the camera changes. */
  onChange: (camera: CameraDef) => void;
  /** Called when interaction starts, and again when it ends (after a delay). */
  onInteracting: (interacting: boolean) => void;
  /** How long after the last input to consider the interaction finished, in ms. */
  settleMs?: number;
}

export class Controls {
  private camera: CameraDef;
  private readonly canvas: HTMLCanvasElement;
  private readonly opts: Required<ControlsOptions>;

  private dragging: 'orbit' | 'pan' | null = null;
  private lastX = 0;
  private lastY = 0;
  private settleTimer: number | undefined;
  private interacting = false;

  constructor(canvas: HTMLCanvasElement, camera: CameraDef, opts: ControlsOptions) {
    this.canvas = canvas;
    this.camera = { ...camera };
    this.opts = { settleMs: 250, ...opts };

    canvas.addEventListener('pointerdown', this.onPointerDown);
    canvas.addEventListener('pointermove', this.onPointerMove);
    canvas.addEventListener('pointerup', this.onPointerUp);
    canvas.addEventListener('pointercancel', this.onPointerUp);
    canvas.addEventListener('wheel', this.onWheel, { passive: false });
    // Middle-drag to pan would otherwise trigger the browser's autoscroll.
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    canvas.style.touchAction = 'none';
    canvas.style.cursor = 'grab';
  }

  setCamera(camera: CameraDef): void {
    this.camera = { ...camera };
  }

  /**
   * Whether dollying re-focuses the lens on the orbit target.
   *
   * On by default, and off the moment the user sets a focus distance by hand.
   * The two behaviours are both right and cannot coexist: a pinhole camera wants
   * focus pinned to the subject so scrolling never defocuses it, and a
   * photographer placing the plane of focus in front of or behind the subject
   * wants exactly that placement kept when they move.
   */
  autofocus = true;

  /** Set the focus distance directly. Turns autofocus off, since it just lost. */
  setFocusDistance(focusDistance: number): void {
    this.autofocus = false;
    this.commit({ ...this.camera, focusDistance });
  }

  /** Set the lens aperture (the full diameter, in scene units). */
  setAperture(aperture: number): void {
    this.commit({ ...this.camera, aperture });
  }

  /** Re-focus on the orbit target and resume doing so on every dolly. */
  refocusOnTarget(): void {
    this.autofocus = true;
    this.commit({
      ...this.camera,
      focusDistance: length(sub(this.camera.eye, this.camera.lookAt)),
    });
  }

  getCamera(): CameraDef {
    return { ...this.camera };
  }

  destroy(): void {
    this.canvas.removeEventListener('pointerdown', this.onPointerDown);
    this.canvas.removeEventListener('pointermove', this.onPointerMove);
    this.canvas.removeEventListener('pointerup', this.onPointerUp);
    this.canvas.removeEventListener('pointercancel', this.onPointerUp);
    this.canvas.removeEventListener('wheel', this.onWheel);
    window.clearTimeout(this.settleTimer);
  }

  /**
   * Mark that input happened. Fires `onInteracting(true)` on the leading edge and
   * `onInteracting(false)` once input has stopped for `settleMs`.
   *
   * The trailing edge is debounced rather than fired on pointerup because a
   * wheel gesture has no "up" event, and because restoring full resolution in
   * the middle of a flick of scroll wheel clicks would stutter.
   */
  private touch(): void {
    if (!this.interacting) {
      this.interacting = true;
      this.opts.onInteracting(true);
    }
    window.clearTimeout(this.settleTimer);
    this.settleTimer = window.setTimeout(() => {
      this.interacting = false;
      this.opts.onInteracting(false);
    }, this.opts.settleMs);
  }

  private commit(camera: CameraDef): void {
    this.camera = camera;
    this.touch();
    this.opts.onChange(camera);
  }

  private onPointerDown = (e: PointerEvent): void => {
    // Middle button, or shift-drag, pans; otherwise orbit.
    this.dragging = e.button === 1 || e.shiftKey ? 'pan' : 'orbit';
    this.lastX = e.clientX;
    this.lastY = e.clientY;
    this.canvas.setPointerCapture(e.pointerId);
    this.canvas.style.cursor = this.dragging === 'pan' ? 'move' : 'grabbing';
    e.preventDefault();
  };

  private onPointerUp = (e: PointerEvent): void => {
    this.dragging = null;
    if (this.canvas.hasPointerCapture(e.pointerId)) {
      this.canvas.releasePointerCapture(e.pointerId);
    }
    this.canvas.style.cursor = 'grab';
  };

  private onPointerMove = (e: PointerEvent): void => {
    if (!this.dragging) return;
    const dx = e.clientX - this.lastX;
    const dy = e.clientY - this.lastY;
    this.lastX = e.clientX;
    this.lastY = e.clientY;
    if (dx === 0 && dy === 0) return;

    // Normalise by the canvas's *displayed* size, not its internal resolution:
    // dragging half way across the viewport should rotate the same amount
    // whether the render is running at 128px or 1024px.
    const rect = this.canvas.getBoundingClientRect();

    if (this.dragging === 'orbit') {
      // A full drag across the viewport is a half turn.
      this.commit(orbit(this.camera, (-dx / rect.width) * Math.PI, (-dy / rect.height) * Math.PI));
      return;
    }

    // Pan moves eye and target together, in the camera's own plane, scaled by
    // distance so the scene appears to track the cursor at any zoom level.
    const [right, down] = cameraBasis(this.camera);
    const dist = length(sub(this.camera.eye, this.camera.lookAt));
    const worldPerPixel = (2 * Math.tan((this.camera.vfovDeg * Math.PI) / 360) * dist) / rect.height;
    const offset: Vec3 = add(scale(right, -dx * worldPerPixel), scale(down, -dy * worldPerPixel));
    this.commit({
      ...this.camera,
      eye: add(this.camera.eye, offset),
      lookAt: add(this.camera.lookAt, offset),
    });
  };

  private onWheel = (e: WheelEvent): void => {
    // The canvas fills the hero, and the page has a great deal to read below
    // it. Swallowing every wheel event would trap a reader whose cursor happens
    // to be over the render — which is most of the screen — so a plain wheel
    // scrolls the page and only a modifier dollies.
    //
    // Alt rather than Ctrl or Shift: Ctrl-wheel is the browser's own zoom and
    // Shift-wheel is horizontal scroll, so both are already spoken for.
    if (!e.altKey) return;
    e.preventDefault();
    // Exponential dolly: each notch is a fixed *ratio*, so zooming feels the
    // same whether you are 1 unit or 1000 units from the subject. A linear step
    // would crawl when far away and overshoot through the subject when close.
    //
    // deltaMode 1 is lines, 2 is pages; normalise both to something pixel-ish.
    const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 100 : 1;
    const factor = Math.exp((e.deltaY * unit) / 500);

    const offset = sub(this.camera.eye, this.camera.lookAt);
    const dist = length(offset);
    // Clamp so the camera cannot pass through the target or fly off to infinity.
    const next = Math.min(Math.max(dist * factor, 1e-3), 1e7);
    const eye = add(this.camera.lookAt, scale(offset, next / dist));
    this.commit({
      ...this.camera,
      eye,
      // Only when autofocus is on. Overwriting a hand-placed focus distance on
      // every scroll notch would make the depth-of-field controls unusable:
      // the plane of focus would snap back to the subject on any camera move.
      focusDistance: this.autofocus ? next : this.camera.focusDistance,
    });
  };
}
