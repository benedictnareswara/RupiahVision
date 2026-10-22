// The app is one full-screen button. First tap: start camera, model and speech (iOS needs a
// gesture for audio). After that it runs hands-free; tapping anywhere repeats the last
// denomination. The camera picture is shown for sighted helpers and hidden from screen readers.

import { useEffect, useRef, useState } from "react";
import { speechFor } from "./labels";
import { message } from "./messages";
import { INITIAL_STATE, openCamera, RecognitionSession, type SessionState } from "./session";

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`cannot load ${url}`));
    image.src = url;
  });
}

function formatRupiah(label: string): string {
  return `Rp${Number(label).toLocaleString("id-ID")}`;
}

export default function App() {
  const [state, setState] = useState<SessionState>(INITIAL_STATE);
  const [screenReader, setScreenReader] = useState(false);
  const [sounds, setSounds] = useState(true);
  const videoRef = useRef<HTMLVideoElement>(null);
  const sessionRef = useRef<RecognitionSession | null>(null);

  useEffect(() => () => sessionRef.current?.stop(), []);

  const session = (sessionRef.current ??= new RecognitionSession(setState));
  const lang = session.lang;
  const t = (key: Parameters<typeof message>[1]) => message(lang, key);

  // ?image=<url> replaces the camera with a still image (for testing without a camera).
  const testImage = new URLSearchParams(window.location.search).get("image");

  function onTap() {
    if (state.phase === "idle") {
      void session.start(async () => {
        if (testImage) return loadImage(testImage);
        const video = videoRef.current!;
        await openCamera(video);
        return video;
      });
    } else if (state.phase === "running") {
      session.repeat();
    }
  }

  const statusText =
    state.phase === "running"
      ? t(state.status === "too_dark" ? "tooDark" : state.status === "seen" ? "seen" : "searching")
      : state.phase === "loading"
        ? t("loading")
        : state.phase === "error" && state.error
          ? t(state.error)
          : t("startHint");

  return (
    <main className="app" lang={lang.slice(0, 2)}>
      <video ref={videoRef} className="camera" aria-hidden="true" muted playsInline />
      <button type="button" className="tap-area" onClick={onTap}>
        {state.phase === "idle" ? (
          <span className="headline">{t("start")}</span>
        ) : (
          <span className="headline" aria-hidden={screenReader}>
            {state.lastLabel ? formatRupiah(state.lastLabel) : "RupiahVision"}
          </span>
        )}
        <span className="status">{statusText}</span>
        {state.phase === "running" && <span className="hint">{t("repeatHint")}</span>}
      </button>

      <div className="settings">
        <button type="button" onClick={() => session.setLang(lang === "id-ID" ? "en-US" : "id-ID")}>
          {t("language")}
        </button>
        <label>
          <input
            type="checkbox"
            checked={screenReader}
            onChange={(e) => {
              setScreenReader(e.target.checked);
              session.screenReaderMode = e.target.checked;
            }}
          />
          {t("screenReader")}
        </label>
        <label>
          <input
            type="checkbox"
            checked={sounds}
            onChange={(e) => {
              setSounds(e.target.checked);
              session.setSounds(e.target.checked);
            }}
          />
          {t("sounds")}
        </label>
      </div>

      {/* Screen reader mode: announcements go here instead of the speech engine. */}
      <p className="visually-hidden" role="status" aria-live="assertive">
        {screenReader ? state.announcement : ""}
      </p>
      {state.lastLabel && (
        <p className="visually-hidden">{speechFor(state.lastLabel, lang)}</p>
      )}
    </main>
  );
}
