// One recognition session: camera (or a test image) -> model -> voting -> speech and cues.
// Owns the frame loop so React only renders the state it reports.

import { Cues } from "./cues";
import { Announcer } from "./decision";
import { speechFor } from "./labels";
import { message, type MessageKey } from "./messages";
import { type FrameSource, Recognizer } from "./recognizer";
import { type Lang, Speaker } from "./speech";

const FRAME_INTERVAL_MS = 150; // ~6 frames per second when inference is fast enough
const HINT_COOLDOWN_MS = 6000; // don't repeat "too dark" more often than this

export type Phase = "idle" | "loading" | "running" | "error";
export type Status = "searching" | "too_dark" | "seen";

export interface SessionState {
  phase: Phase;
  status: Status;
  lastLabel: string | null; // last announced denomination
  announcement: string; // text for the aria-live region (screen reader mode)
  error: MessageKey | null;
  inferenceMs: number | null;
}

export const INITIAL_STATE: SessionState = {
  phase: "idle",
  status: "searching",
  lastLabel: null,
  announcement: "",
  error: null,
  inferenceMs: null,
};

export async function openCamera(video: HTMLVideoElement): Promise<void> {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;
  video.muted = true;
  video.playsInline = true;
  await video.play();
}

export class RecognitionSession {
  private state: SessionState = INITIAL_STATE;
  private readonly speaker = new Speaker("id-ID");
  private readonly cues = new Cues();
  private readonly announcer = new Announcer();
  private recognizer: Recognizer | null = null;
  private running = false;
  private lastHintAt = -Infinity;
  private readonly onChange: (state: SessionState) => void;
  screenReaderMode = false;

  constructor(onChange: (state: SessionState) => void) {
    this.onChange = onChange;
  }

  get lang(): Lang {
    return this.speaker.lang;
  }

  private update(patch: Partial<SessionState>): void {
    this.state = { ...this.state, ...patch };
    this.onChange(this.state);
  }

  /** Speak through the speech engine, or hand the text to the screen reader. */
  private say(text: string): void {
    if (this.screenReaderMode) {
      // Changing the text (even to the same words) makes the live region announce again.
      this.update({ announcement: this.state.announcement === text ? `${text} ` : text });
    } else {
      this.speaker.say(text);
    }
  }

  /** Must be called from the first tap: unlocks speech and audio on iOS. */
  async start(getSource: () => Promise<FrameSource>): Promise<void> {
    this.speaker.unlock();
    this.cues.unlock();
    this.update({ phase: "loading" });
    this.say(message(this.lang, "loading"));

    let source: FrameSource;
    try {
      [source, this.recognizer] = await Promise.all([
        getSource().catch((e: unknown) => Promise.reject({ key: "cameraError", e })),
        Recognizer.load().catch((e: unknown) => Promise.reject({ key: "modelError", e })),
      ]);
    } catch (failure) {
      const key = (failure as { key: MessageKey }).key ?? "modelError";
      console.error(failure);
      this.update({ phase: "error", error: key });
      this.say(message(this.lang, key));
      return;
    }

    this.update({ phase: "running" });
    this.say(message(this.lang, "ready"));
    this.running = true;
    void this.loop(source);
  }

  private async loop(source: FrameSource): Promise<void> {
    while (this.running && this.recognizer) {
      const started = performance.now();
      const outcome = await this.recognizer.recognize(source);
      if (outcome.kind === "too_dark") {
        this.cues.setLevel(0);
        this.announcer.push(null);
        this.update({ status: "too_dark" });
        if (started - this.lastHintAt > HINT_COOLDOWN_MS) {
          this.lastHintAt = started;
          this.say(message(this.lang, "tooDark"));
        }
      } else if (outcome.kind === "result") {
        const { result } = outcome;
        this.cues.setLevel(result.noteLevel);
        const spoken = this.announcer.push(result);
        this.update({
          status: result.noteLevel >= 0.5 ? "seen" : "searching",
          inferenceMs: outcome.ms.inference,
        });
        if (spoken) {
          this.cues.earcon();
          this.update({ lastLabel: spoken });
          this.say(speechFor(spoken, this.lang));
        }
      }
      const elapsed = performance.now() - started;
      await new Promise((r) => setTimeout(r, Math.max(0, FRAME_INTERVAL_MS - elapsed)));
    }
  }

  /** Tap anywhere while running: say the last denomination again. */
  repeat(): void {
    const last = this.state.lastLabel;
    this.say(last ? speechFor(last, this.lang) : message(this.lang, "nothingYet"));
  }

  setLang(lang: Lang): void {
    this.speaker.setLang(lang);
    this.onChange({ ...this.state });
  }

  setSounds(enabled: boolean): void {
    this.cues.enabled = enabled;
  }

  stop(): void {
    this.running = false;
    this.cues.stop();
  }
}
