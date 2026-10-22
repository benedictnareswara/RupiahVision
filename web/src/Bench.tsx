// /?bench: measures in-browser latency on the device it's opened on (PLAN.md section 5).
// Runs the same path as the app (canvas crop -> tensor -> ONNX Runtime Web) on a synthetic
// frame, then offers the timings as JSON to save into results/bench/.

import { useState } from "react";
import { Recognizer } from "./recognizer";

const WARMUP = 10;
const RUNS = 50;

interface BenchResult {
  model: string;
  backend: string;
  cross_origin_isolated: boolean;
  user_agent: string;
  load_ms: number;
  runs: number;
  preprocess_ms: { median: number; p95: number };
  inference_ms: { median: number; p95: number };
  measured_at: string;
}

function percentile(values: number[], p: number): number {
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.ceil((p / 100) * sorted.length) - 1);
  return Math.round(sorted[Math.max(0, index)]! * 10) / 10;
}

function syntheticFrame(): HTMLCanvasElement {
  // A bright 1280x720 gradient: passes the brightness check like a real frame would.
  const canvas = document.createElement("canvas");
  canvas.width = 1280;
  canvas.height = 720;
  const ctx = canvas.getContext("2d")!;
  const gradient = ctx.createLinearGradient(0, 0, 1280, 720);
  gradient.addColorStop(0, "#d8c9a0");
  gradient.addColorStop(1, "#6a8fb5");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 1280, 720);
  return canvas;
}

async function runBench(onProgress: (text: string) => void): Promise<BenchResult> {
  onProgress("Loading model…");
  const t0 = performance.now();
  const recognizer = await Recognizer.load();
  const loadMs = performance.now() - t0;
  const frame = syntheticFrame();
  const preprocess: number[] = [];
  const inference: number[] = [];
  for (let i = 0; i < WARMUP + RUNS; i++) {
    const outcome = await recognizer.recognize(frame);
    if (outcome.kind !== "result") throw new Error(`unexpected outcome ${outcome.kind}`);
    if (i >= WARMUP) {
      preprocess.push(outcome.ms.preprocess);
      inference.push(outcome.ms.inference);
    }
    onProgress(`Run ${i + 1} / ${WARMUP + RUNS}`);
  }
  return {
    model: recognizer.meta.run,
    backend: "onnxruntime-web wasm",
    cross_origin_isolated: window.crossOriginIsolated,
    user_agent: navigator.userAgent,
    load_ms: Math.round(loadMs),
    runs: RUNS,
    preprocess_ms: { median: percentile(preprocess, 50), p95: percentile(preprocess, 95) },
    inference_ms: { median: percentile(inference, 50), p95: percentile(inference, 95) },
    measured_at: new Date().toISOString(),
  };
}

export default function Bench() {
  const [progress, setProgress] = useState("");
  const [result, setResult] = useState<BenchResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const json = result ? JSON.stringify(result, null, 2) : "";

  function download() {
    const blob = new Blob([json + "\n"], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `bench-${result!.model}.json`;
    link.click();
  }

  return (
    <main className="bench">
      <h1>Latency benchmark</h1>
      <p>
        Runs the model {RUNS} times (after {WARMUP} warm-up runs) on this device, the same way the
        app does. Save the JSON into <code>results/bench/</code>.
      </p>
      <button
        type="button"
        onClick={() => {
          setError(null);
          runBench(setProgress).then(setResult, (e: unknown) => setError(String(e)));
        }}
      >
        Run benchmark
      </button>
      <p role="status">{error ?? (result ? "Done." : progress)}</p>
      {result && (
        <>
          <pre>{json}</pre>
          <button type="button" onClick={download}>
            Download JSON
          </button>
        </>
      )}
    </main>
  );
}
