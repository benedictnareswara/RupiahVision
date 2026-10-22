// Non-speech feedback while aiming: soft ticks that speed up as a note comes into view, and a
// short rising "earcon" right before a denomination is spoken. Vibration is used as well
// where the browser supports it (Android Chrome; iOS Safari has no Vibration API).

const MIN_INTERVAL_MS = 150; // note clearly in view
const MAX_INTERVAL_MS = 900; // a hint of a note
const SILENT_BELOW = 0.2; // noteLevel under this: no ticks at all

/** Milliseconds between ticks for a given note level (0..1), or null for silence. */
export function tickInterval(noteLevel: number): number | null {
  if (noteLevel < SILENT_BELOW) return null;
  const t = Math.min(1, (noteLevel - SILENT_BELOW) / (1 - SILENT_BELOW));
  return Math.round(MAX_INTERVAL_MS - t * (MAX_INTERVAL_MS - MIN_INTERVAL_MS));
}

export class Cues {
  private context: AudioContext | null = null;
  private level = 0;
  private timer: ReturnType<typeof setTimeout> | null = null;
  enabled = true;

  /** Call from a tap handler: browsers start audio contexts suspended until a gesture. */
  unlock(): void {
    this.context ??= new AudioContext();
    void this.context.resume();
  }

  private beep(frequency: number, durationMs: number, gain = 0.08, startOffset = 0): void {
    const ctx = this.context;
    if (!ctx || !this.enabled) return;
    const start = ctx.currentTime + startOffset;
    const osc = ctx.createOscillator();
    const amp = ctx.createGain();
    osc.frequency.value = frequency;
    amp.gain.setValueAtTime(gain, start);
    amp.gain.exponentialRampToValueAtTime(0.0001, start + durationMs / 1000);
    osc.connect(amp).connect(ctx.destination);
    osc.start(start);
    osc.stop(start + durationMs / 1000);
  }

  /** Update how strongly a note is in view; drives the tick rate. */
  setLevel(noteLevel: number): void {
    this.level = noteLevel;
    if (this.timer === null) this.scheduleTick();
  }

  private scheduleTick(): void {
    const interval = tickInterval(this.level);
    if (interval === null) {
      this.timer = null;
      return;
    }
    this.timer = setTimeout(() => {
      this.beep(1400, 12, 0.05);
      this.scheduleTick();
    }, interval);
  }

  /** Rising two-note sound, then a short buzz: "a denomination is coming". */
  earcon(): void {
    this.beep(660, 90);
    this.beep(990, 120, 0.08, 0.1);
    navigator.vibrate?.([60, 40, 60]);
  }

  stop(): void {
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = null;
    this.level = 0;
  }
}
