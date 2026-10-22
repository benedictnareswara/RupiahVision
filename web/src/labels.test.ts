import { describe, expect, it } from "vitest";
import { aggregate, CLASSES, OUTPUTS, speechFor, UNKNOWN } from "./labels";

describe("labels", () => {
  it("has unknown plus 7 denominations x 2 series x 2 sides", () => {
    expect(CLASSES).toHaveLength(29);
    expect(CLASSES[0]).toBe(UNKNOWN);
    expect(OUTPUTS).toEqual(["unknown", "1000", "2000", "5000", "10000", "20000", "50000", "100000"]);
  });

  it("sums class probabilities per denomination", () => {
    const probs = new Float32Array(CLASSES.length);
    probs[CLASSES.indexOf("50000_TE2016_front")] = 0.5;
    probs[CLASSES.indexOf("50000_TE2022_back")] = 0.3;
    probs[CLASSES.indexOf("unknown")] = 0.2;
    const out = aggregate(probs);
    expect(out[OUTPUTS.indexOf("50000")]).toBeCloseTo(0.8);
    expect(out[0]).toBeCloseTo(0.2);
    expect(out.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
  });

  it("rejects the wrong number of probabilities", () => {
    expect(() => aggregate([0.5, 0.5])).toThrow();
  });

  it("has an Indonesian and English phrase for every output", () => {
    for (const output of OUTPUTS) {
      expect(speechFor(output, "id-ID")).toBeTruthy();
      expect(speechFor(output, "en-US")).toBeTruthy();
    }
    expect(speechFor("100000")).toBe("Seratus ribu rupiah");
  });
});
