// Frame -> model input. Matches eval_transform in ml/src/rupiahvision/training/data.py:
// the centre square of the frame, resized to size x size, RGB scaled to 0..1 and then
// normalised with the ImageNet mean/std, laid out NCHW.

export interface Square {
  sx: number;
  sy: number;
  side: number;
}

export function centerSquare(width: number, height: number): Square {
  const side = Math.min(width, height);
  return { sx: Math.floor((width - side) / 2), sy: Math.floor((height - side) / 2), side };
}

/** RGBA pixels (from canvas getImageData) -> Float32 NCHW tensor data. */
export function toTensorData(
  rgba: Uint8ClampedArray,
  size: number,
  mean: readonly number[],
  std: readonly number[],
): Float32Array {
  const plane = size * size;
  const out = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) {
      out[c * plane + i] = (rgba[i * 4 + c]! / 255 - mean[c]!) / std[c]!;
    }
  }
  return out;
}

/** Mean luma (0..255) of RGBA pixels, used to tell the user it's too dark to see. */
export function brightness(rgba: Uint8ClampedArray): number {
  let sum = 0;
  const pixels = rgba.length / 4;
  for (let i = 0; i < pixels; i++) {
    sum += 0.299 * rgba[i * 4]! + 0.587 * rgba[i * 4 + 1]! + 0.114 * rgba[i * 4 + 2]!;
  }
  return pixels ? sum / pixels : 0;
}

// Below this mean brightness the camera image is mostly noise; the app skips the frame and
// says "terlalu gelap" (too dark). Chosen by eye, not tuned on data.
export const TOO_DARK = 35;
