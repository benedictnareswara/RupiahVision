// Everything the app says or shows besides denominations (those live in data/labels.json).
import type { Lang } from "./speech";

export const MESSAGES = {
  "id-ID": {
    start: "Ketuk di mana saja untuk mulai",
    startHint: "Kamera akan menyala dan aplikasi akan menyebutkan nilai uang kertas.",
    loading: "Memuat",
    ready: "Siap. Arahkan kamera ke uang kertas.",
    searching: "Mencari uang kertas…",
    tooDark: "Terlalu gelap",
    seen: "Uang terlihat, tahan sebentar",
    repeatHint: "Ketuk layar untuk mengulang",
    nothingYet: "Belum ada uang yang dikenali",
    cameraError: "Kamera tidak dapat dibuka. Izinkan akses kamera, lalu muat ulang halaman.",
    modelError: "Model tidak dapat dimuat. Periksa koneksi, lalu muat ulang halaman.",
    language: "English",
    screenReader: "Mode pembaca layar",
    sounds: "Bunyi bantuan",
  },
  "en-US": {
    start: "Tap anywhere to start",
    startHint: "The camera will turn on and the app will say the value of banknotes.",
    loading: "Loading",
    ready: "Ready. Point the camera at a banknote.",
    searching: "Looking for a banknote…",
    tooDark: "Too dark",
    seen: "Note in view, hold still",
    repeatHint: "Tap the screen to repeat",
    nothingYet: "No banknote recognized yet",
    cameraError: "The camera could not be opened. Allow camera access, then reload the page.",
    modelError: "The model could not be loaded. Check the connection, then reload the page.",
    language: "Bahasa Indonesia",
    screenReader: "Screen reader mode",
    sounds: "Aiming sounds",
  },
} satisfies Record<Lang, Record<string, string>>;

export type MessageKey = keyof (typeof MESSAGES)["id-ID"];

export function message(lang: Lang, key: MessageKey): string {
  return MESSAGES[lang][key];
}
