// When to speak. Mirrors ml/src/rupiahvision/training/metrics.py:
//   1. softmax(logits / temperature): calibrated probabilities (temperature fitted on val);
//   2. a frame is "confident" if its top output isn't "unknown" and its probability is at
//      least the threshold (chosen on val so at most 1% of announcements were wrong);
//   3. the app speaks only when `agree` of the last `window` frames are confident about the
//      same denomination. One lucky frame never triggers an announcement.

export interface ModelMeta {
  run: string;
  outputs: string[];
  input: { name: string; size: number; mean: number[]; std: number[] };
  temperature: number;
  threshold: number;
}

export interface FrameResult {
  label: string; // top output: "unknown" or a denomination like "50000"
  confidence: number; // calibrated probability of `label`
  noteLevel: number; // confidence of the top guess if it's a note, else 0; drives aiming cues
  confident: boolean;
}

export function softmax(logits: ArrayLike<number>, temperature = 1): Float32Array {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i]! / temperature);
  const out = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    out[i] = Math.exp(logits[i]! / temperature - max);
    sum += out[i]!;
  }
  for (let i = 0; i < out.length; i++) out[i]! /= sum;
  return out;
}

export function classify(logits: ArrayLike<number>, meta: ModelMeta): FrameResult {
  const probs = softmax(logits, meta.temperature);
  let best = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i]! > probs[best]!) best = i;
  const label = meta.outputs[best]!;
  const confidence = probs[best]!;
  return {
    label,
    confidence,
    // Not 1 - P(unknown): when "unknown" is the top guess but spread thin (say 0.3), that
    // would still report a strong note and tick over an empty table.
    noteLevel: label === "unknown" ? 0 : confidence,
    confident: label !== "unknown" && confidence >= meta.threshold,
  };
}

export interface AnnouncerOptions {
  window: number; // frames remembered
  agree: number; // confident frames that must agree to speak
  clearAfter: number; // frames without a confident note before the same note is spoken again
}

export const DEFAULT_ANNOUNCER: AnnouncerOptions = { window: 5, agree: 3, clearAfter: 4 };

/** Turns a stream of per-frame results into announcements, each spoken once. */
export class Announcer {
  private recent: (string | null)[] = [];
  private lastSpoken: string | null = null;
  private quietFrames = 0;
  private readonly options: AnnouncerOptions;

  constructor(options: AnnouncerOptions = DEFAULT_ANNOUNCER) {
    this.options = options;
  }

  /** Add one frame (null for a frame that was skipped, e.g. too dark). Returns what to say. */
  push(frame: FrameResult | null): string | null {
    const label = frame?.confident ? frame.label : null;
    this.recent.push(label);
    if (this.recent.length > this.options.window) this.recent.shift();

    if (label === null) {
      this.quietFrames++;
      // The note has left the frame: the same note may be announced again next time.
      if (this.quietFrames >= this.options.clearAfter) this.lastSpoken = null;
      return null;
    }
    this.quietFrames = 0;

    const votes = this.recent.filter((l) => l === label).length;
    if (votes >= this.options.agree && label !== this.lastSpoken) {
      this.lastSpoken = label;
      return label;
    }
    return null;
  }

  get last(): string | null {
    return this.lastSpoken;
  }

  reset(): void {
    this.recent = [];
    this.lastSpoken = null;
    this.quietFrames = 0;
  }
}
