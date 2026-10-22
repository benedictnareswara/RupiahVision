import { describe, expect, it } from "vitest";
import { brightness, centerSquare, toTensorData } from "./preprocess";

describe("centerSquare", () => {
  it("crops the middle of landscape and portrait frames", () => {
    expect(centerSquare(1280, 720)).toEqual({ sx: 280, sy: 0, side: 720 });
    expect(centerSquare(720, 1280)).toEqual({ sx: 0, sy: 280, side: 720 });
  });
});

describe("toTensorData", () => {
  it("normalises RGB into NCHW planes", () => {
    // a single red pixel: R plane = (1 - 0.5) / 0.5 = 1, G and B planes = -1
    const rgba = new Uint8ClampedArray([255, 0, 0, 255]);
    const out = toTensorData(rgba, 1, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]);
    expect(Array.from(out)).toEqual([1, -1, -1]);
  });

  it("uses the ImageNet statistics the model was trained with", () => {
    const gray = new Uint8ClampedArray(4 * 4).fill(128);
    const out = toTensorData(gray, 2, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
    expect(out).toHaveLength(12);
    expect(out[0]).toBeCloseTo((128 / 255 - 0.485) / 0.229, 5); // R plane
    expect(out[4]).toBeCloseTo((128 / 255 - 0.456) / 0.224, 5); // G plane
    expect(out[8]).toBeCloseTo((128 / 255 - 0.406) / 0.225, 5); // B plane
  });
});

describe("brightness", () => {
  it("is mean luma", () => {
    expect(brightness(new Uint8ClampedArray([0, 0, 0, 255]))).toBe(0);
    expect(brightness(new Uint8ClampedArray([255, 255, 255, 255]))).toBeCloseTo(255);
  });
});
