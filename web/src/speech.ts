// Spoken output through the Web Speech API, in Indonesian when the device has a voice for it.
//
// iOS only lets a page speak after a user gesture, so `unlock()` must be called from the
// first tap; after that the app can speak on its own (hands-free).

export type Lang = "id-ID" | "en-US";

export class Speaker {
  private voice: SpeechSynthesisVoice | null = null;
  private readonly synth: SpeechSynthesis | undefined =
    typeof window !== "undefined" ? window.speechSynthesis : undefined;
  lang: Lang;

  constructor(lang: Lang = "id-ID") {
    this.lang = lang;
    this.pickVoice();
    this.synth?.addEventListener?.("voiceschanged", () => this.pickVoice());
  }

  get supported(): boolean {
    return this.synth !== undefined;
  }

  /** True if a voice for the current language is installed (otherwise the default voice reads it). */
  get hasNativeVoice(): boolean {
    return this.voice !== null;
  }

  setLang(lang: Lang): void {
    this.lang = lang;
    this.pickVoice();
  }

  private pickVoice(): void {
    const prefix = this.lang.slice(0, 2);
    const voices = this.synth?.getVoices() ?? [];
    this.voice =
      voices.find((v) => v.lang.replace("_", "-") === this.lang) ??
      voices.find((v) => v.lang.toLowerCase().startsWith(prefix)) ??
      null;
  }

  /** Call from a tap handler: iOS ignores speech that wasn't first started by a gesture. */
  unlock(): void {
    this.say("");
  }

  say(text: string, { interrupt = true } = {}): void {
    if (!this.synth) return;
    if (interrupt) this.synth.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = this.lang;
    if (this.voice) utterance.voice = this.voice;
    utterance.rate = 0.95;
    this.synth.speak(utterance);
  }
}
