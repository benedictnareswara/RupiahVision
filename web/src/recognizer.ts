// Runs the exported model in the browser with ONNX Runtime Web (WebAssembly, CPU).
//
// The model (public/model/model.onnx) and its metadata (public/model/model.json) come from
// `python -m rupiahvision.training.export --run <model> --publish`.

import * as ort from "onnxruntime-web/wasm";
import { classify, type FrameResult, type ModelMeta } from "./decision";
import { brightness, centerSquare, toTensorData, TOO_DARK } from "./preprocess";

const MODEL_DIR = `${import.meta.env.BASE_URL}model/`;

export type FrameSource = HTMLVideoElement | HTMLImageElement | HTMLCanvasElement;

export type FrameOutcome =
  | { kind: "result"; result: FrameResult; ms: { preprocess: number; inference: number } }
  | { kind: "too_dark"; brightness: number }
  | { kind: "not_ready" };

function sourceSize(source: FrameSource): [number, number] {
  if (source instanceof HTMLVideoElement) return [source.videoWidth, source.videoHeight];
  if (source instanceof HTMLImageElement) return [source.naturalWidth, source.naturalHeight];
  return [source.width, source.height];
}

export class Recognizer {
  readonly meta: ModelMeta;
  private readonly session: ort.InferenceSession;
  private readonly canvas: HTMLCanvasElement;
  private readonly context: CanvasRenderingContext2D;

  private constructor(meta: ModelMeta, session: ort.InferenceSession) {
    this.meta = meta;
    this.session = session;
    this.canvas = document.createElement("canvas");
    this.canvas.width = this.canvas.height = meta.input.size;
    this.context = this.canvas.getContext("2d", { willReadFrequently: true })!;
    this.context.imageSmoothingQuality = "high";
  }

  static async load(): Promise<Recognizer> {
    const meta = (await (await fetch(`${MODEL_DIR}model.json`)).json()) as ModelMeta;
    const session = await ort.InferenceSession.create(`${MODEL_DIR}model.onnx`, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    return new Recognizer(meta, session);
  }

  /** The centre square of the source, resized to the model's input size, as RGBA pixels. */
  pixels(source: FrameSource): Uint8ClampedArray | null {
    const [width, height] = sourceSize(source);
    if (!width || !height) return null;
    const { sx, sy, side } = centerSquare(width, height);
    const size = this.meta.input.size;
    this.context.drawImage(source, sx, sy, side, side, 0, 0, size, size);
    return this.context.getImageData(0, 0, size, size).data;
  }

  async logits(rgba: Uint8ClampedArray): Promise<Float32Array> {
    const { size, mean, std, name } = this.meta.input;
    const input = new ort.Tensor("float32", toTensorData(rgba, size, mean, std), [1, 3, size, size]);
    const output = await this.session.run({ [name]: input });
    return output[this.session.outputNames[0]!]!.data as Float32Array;
  }

  async recognize(source: FrameSource): Promise<FrameOutcome> {
    const t0 = performance.now();
    const rgba = this.pixels(source);
    if (!rgba) return { kind: "not_ready" };
    const light = brightness(rgba);
    if (light < TOO_DARK) return { kind: "too_dark", brightness: light };
    const t1 = performance.now();
    const logits = await this.logits(rgba);
    const t2 = performance.now();
    return {
      kind: "result",
      result: classify(logits, this.meta),
      ms: { preprocess: t1 - t0, inference: t2 - t1 },
    };
  }
}
