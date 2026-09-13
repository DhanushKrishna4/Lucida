/** Small typed helpers for building the control panel without a UI framework. */

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string> = {},
  ...children: (Node | string)[]
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  node.append(...children);
  return node;
}

export function group(legend: string, ...children: Node[]): HTMLFieldSetElement {
  return el('fieldset', {}, el('legend', {}, legend), ...children);
}

/**
 * Moves a slider from outside the panel.
 *
 * Needed because two controls here are driven by something other than a drag:
 * click-to-focus sets the focus distance from the image, and loading a scene
 * changes what range a distance in scene units should even span — a Cornell box
 * is 555 units across and a glass-ball scene is about 5.
 */
export interface SliderHandle {
  set(value: number): void;
  setRange(min: number, max: number, step: number): void;
}

export interface SliderOptions {
  label: string;
  min: number;
  max: number;
  step?: number;
  value: number;
  /** Render the numeric readout; defaults to the raw value. */
  format?: (v: number) => string;
  onInput: (v: number) => void;
  /** Receives a handle for driving this slider from elsewhere. */
  bind?: (handle: SliderHandle) => void;
}

export function slider(o: SliderOptions): HTMLLabelElement {
  const fmt = o.format ?? String;
  const readout = el('span', {}, fmt(o.value));
  const input = el('input', {
    type: 'range',
    min: String(o.min),
    max: String(o.max),
    step: String(o.step ?? 1),
    value: String(o.value),
  });
  input.addEventListener('input', () => {
    const v = Number(input.value);
    readout.textContent = fmt(v);
    o.onInput(v);
  });
  o.bind?.({
    // Deliberately does not call `onInput`: the caller already knows the value
    // it just set, and echoing it back invites a loop between a control and the
    // thing it controls.
    set(value) {
      input.value = String(value);
      readout.textContent = fmt(Number(input.value));
    },
    setRange(min, max, step) {
      input.min = String(min);
      input.max = String(max);
      input.step = String(step);
      readout.textContent = fmt(Number(input.value));
    },
  });
  return el(
    'label',
    {},
    el('span', { class: 'label-row' }, el('span', {}, o.label), readout),
    input,
  );
}

export function checkbox(
  label: string,
  value: boolean,
  onChange: (v: boolean) => void,
  bind?: (set: (v: boolean) => void) => void,
): HTMLLabelElement {
  const input = el('input', { type: 'checkbox' });
  input.checked = value;
  input.addEventListener('change', () => onChange(input.checked));
  bind?.((v) => {
    input.checked = v;
  });
  return el('label', { class: 'check' }, input, el('span', {}, label));
}

export function select<T extends string>(
  label: string,
  options: { value: T; label: string }[],
  value: T,
  onChange: (v: T) => void,
): HTMLLabelElement {
  const sel = el('select', {});
  for (const o of options) {
    const opt = el('option', { value: o.value }, o.label);
    if (o.value === value) opt.selected = true;
    sel.append(opt);
  }
  sel.addEventListener('change', () => onChange(sel.value as T));
  return el('label', {}, el('span', { class: 'label-row' }, el('span', {}, label)), sel);
}

export function button(label: string, onClick: () => void): HTMLButtonElement {
  const b = el('button', {}, label);
  b.addEventListener('click', onClick);
  return b;
}

export function note(html: string): HTMLParagraphElement {
  const p = el('p', { class: 'note' });
  p.innerHTML = html;
  return p;
}
