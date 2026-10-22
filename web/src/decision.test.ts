import { describe, expect, it } from "vitest";
import { Announcer, classify, type FrameResult, type ModelMeta, softmax } from "./decision";

const META: ModelMeta = {
  run: "test",
  outputs: ["unknown", "1000", "2000", "5000", "10000", "20000", "50000", "100000"],
  input: { name: "input", size: 224, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] },
  temperature: 2,
  threshold: 0.6,
};

function frame(label: string, confident = true): FrameResult {
  return { label, confidence: confident ? 0.9 : 0.3, noteLevel: 0.9, confident };
}

describe("softmax", () => {
  it("sums to 1 and is softened by temperature", () => {
    const sharp = softmax([2, 0, -1]);
    const soft = softmax([2, 0, -1], 3);
    expect(sharp.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
    expect(soft[0]!).toBeLessThan(sharp[0]!);
  });

  it("matches the Python reference", () => {
    // numpy: softmax(np.array([[1.0, 2.0, 3.0]]) / 1.5)
    const p = softmax([1, 2, 3], 1.5);
    expect(p[0]).toBeCloseTo(0.1483, 3);
    expect(p[2]).toBeCloseTo(0.5627, 3);
  });
});

describe("classify", () => {
  it("is confident only for a note above the threshold", () => {
    const logits = [0, 0, 0, 0, 0, 0, 8, 0]; // strongly 50000
    const result = classify(logits, META);
    expect(result.label).toBe("50000");
    expect(result.confident).toBe(true);
    expect(result.noteLevel).toBe(result.confidence);
  });

  it("never treats unknown as an announcement", () => {
    const result = classify([9, 0, 0, 0, 0, 0, 0, 0], META);
    expect(result.label).toBe("unknown");
    expect(result.confident).toBe(false);
  });

  it("reports no note level when unknown is the top guess, even a weak one", () => {
    // P(unknown) is only ~0.2 here, but it is still the most likely output.
    const result = classify([1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], { ...META, temperature: 1 });
    expect(result.label).toBe("unknown");
    expect(result.noteLevel).toBe(0);
  });

  it("applies the temperature before the threshold", () => {
    // softmax([0, 2] + six zeros) has P(1000) = 0.51 at T=1 but only 0.24 at T=2.
    const logits = [0, 2, 0, 0, 0, 0, 0, 0];
    expect(classify(logits, { ...META, temperature: 1, threshold: 0.5 }).confident).toBe(true);
    expect(classify(logits, { ...META, temperature: 2, threshold: 0.5 }).confident).toBe(false);
  });
});

describe("Announcer", () => {
  it("needs 3 agreeing confident frames out of 5", () => {
    const a = new Announcer();
    expect(a.push(frame("50000"))).toBeNull();
    expect(a.push(frame("2000"))).toBeNull();
    expect(a.push(frame("50000"))).toBeNull();
    expect(a.push(frame("50000"))).toBe("50000");
  });

  it("ignores frames below the threshold", () => {
    const a = new Announcer();
    for (let i = 0; i < 10; i++) expect(a.push(frame("50000", false))).toBeNull();
  });

  it("says a note once while it stays in view", () => {
    const a = new Announcer();
    const said = Array.from({ length: 10 }, () => a.push(frame("100000"))).filter(Boolean);
    expect(said).toEqual(["100000"]);
  });

  it("speaks again for a different note", () => {
    const a = new Announcer();
    for (let i = 0; i < 3; i++) a.push(frame("1000"));
    const said = Array.from({ length: 5 }, () => a.push(frame("5000"))).filter(Boolean);
    expect(said).toEqual(["5000"]);
  });

  it("repeats the same note after it has left the frame", () => {
    const a = new Announcer();
    for (let i = 0; i < 3; i++) a.push(frame("20000"));
    for (let i = 0; i < 4; i++) a.push(null); // dark or empty frames
    const said = Array.from({ length: 3 }, () => a.push(frame("20000"))).filter(Boolean);
    expect(said).toEqual(["20000"]);
  });

  it("does not repeat after a brief flicker", () => {
    const a = new Announcer();
    for (let i = 0; i < 3; i++) a.push(frame("20000"));
    a.push(frame("20000", false));
    const said = Array.from({ length: 3 }, () => a.push(frame("20000"))).filter(Boolean);
    expect(said).toEqual([]);
  });
});
