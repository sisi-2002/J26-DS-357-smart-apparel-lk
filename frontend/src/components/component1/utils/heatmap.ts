import type { Bbox } from '../types/component1';

export interface Heatmap { width: number; height: number; values: Float64Array; min: number; max: number }

/** Decode only genuine 2D NumPy float arrays. Never infer saliency from a composite. */
export function decodeHeatmap(buffer: ArrayBuffer, width: number, height: number): Heatmap {
  const bytes = new Uint8Array(buffer);
  if (bytes.length < 10 || bytes[0] !== 0x93 || String.fromCharCode(...bytes.slice(1, 6)) !== 'NUMPY') throw new Error('Invalid NumPy heatmap.');
  const view = new DataView(buffer);
  const major = bytes[6];
  if (![1, 2, 3].includes(major)) throw new Error('Unsupported NumPy format.');
  const start = major === 1 ? 10 : 12;
  const length = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  if (start + length > bytes.length) throw new Error('Truncated heatmap header.');
  const header = new TextDecoder().decode(bytes.slice(start, start + length));
  const dtype = /['"]descr['"]\s*:\s*['"]([<>|])f([48])['"]/.exec(header);
  const shape = /['"]shape['"]\s*:\s*\(\s*(\d+)\s*,\s*(\d+)\s*,?\s*\)/.exec(header);
  if (!dtype || !shape || !/['"]fortran_order['"]\s*:\s*False/.test(header)) throw new Error('Unsupported heatmap layout; expected a C-order 2D float array.');
  if (+shape[1] !== height || +shape[2] !== width) throw new Error('Heatmap dimensions do not match the original image.');
  const size = +dtype[2];
  const count = width * height;
  const offset = start + length;
  if (!Number.isSafeInteger(count) || count <= 0 || offset + count * size !== bytes.length) throw new Error('Invalid heatmap payload size.');
  const values = new Float64Array(count);
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < count; i++) {
    const value = size === 4 ? view.getFloat32(offset + i * size, dtype[1] !== '>') : view.getFloat64(offset + i * size, dtype[1] !== '>');
    if (!Number.isFinite(value)) throw new Error('Heatmap contains non-finite values.');
    values[i] = value;
    min = Math.min(min, value); max = Math.max(max, value);
  }
  if (max - min < 1e-8) throw new Error('Heatmap has no meaningful range to display.');
  return { width, height, values, min, max };
}

/** OpenCV JET's 256-entry display scale. The values themselves remain untouched. */
export function heatmapPixels(map: Heatmap): Uint8ClampedArray {
  const pixels = new Uint8ClampedArray(map.values.length * 4);
  const channel = (x: number) => Math.round(255 * Math.max(0, Math.min(1, x)));
  map.values.forEach((value, i) => {
    // Same full-image min/max normalization and uint8 truncation as backend _overlay.
    const t = Math.floor(255 * (value - map.min) / (map.max - map.min)) / 255;
    pixels[i * 4] = channel(1.5 - Math.abs(4 * t - 3));
    pixels[i * 4 + 1] = channel(1.5 - Math.abs(4 * t - 2));
    pixels[i * 4 + 2] = channel(1.5 - Math.abs(4 * t - 1));
    pixels[i * 4 + 3] = 255;
  });
  return pixels;
}

export function heatmapImage(map: Heatmap): string {
  const canvas = document.createElement('canvas');
  canvas.width = map.width; canvas.height = map.height;
  const context = canvas.getContext('2d');
  if (!context) throw new Error('Heatmap rendering is unavailable in this browser.');
  const pixels = context.createImageData(map.width, map.height);
  pixels.data.set(heatmapPixels(map));
  context.putImageData(pixels, 0, 0);
  return canvas.toDataURL('image/png');
}

/** Geometric crop only. Both images use these identical original-image coordinates. */
export function defectCrop(box: Bbox, width: number, height: number): Bbox {
  const padX = Math.max(1, (box.x_max - box.x_min) * 0.15);
  const padY = Math.max(1, (box.y_max - box.y_min) * 0.15);
  return { x_min: Math.max(0, box.x_min - padX), y_min: Math.max(0, box.y_min - padY), x_max: Math.min(width, box.x_max + padX), y_max: Math.min(height, box.y_max + padY) };
}
