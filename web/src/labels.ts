// Label schema shared with the ML package (data/labels.json). The model outputs one
// probability per fine-grained class; the app only announces a denomination, so
// `aggregate` sums class probabilities per denomination (mirrors LabelSchema.aggregate
// in ml/src/rupiahvision/labels.py).
import schema from "../../data/labels.json";

export type Lang = keyof typeof schema.speech;

export const UNKNOWN = schema.unknown_class;
export const CLASSES: readonly string[] = schema.classes;

/** What the app can announce: "unknown" followed by each denomination. */
export const OUTPUTS: readonly string[] = [UNKNOWN, ...schema.denominations.map(String)];

/** For each class index, the index into OUTPUTS it contributes to. */
const CLASS_TO_OUTPUT: readonly number[] = CLASSES.map((name) => {
  if (name === UNKNOWN) return 0;
  const denom = name.split("_", 1)[0] ?? "";
  const index = OUTPUTS.indexOf(denom);
  if (index < 1) throw new Error(`class ${name} has no matching denomination`);
  return index;
});

export function aggregate(probs: ArrayLike<number>): Float32Array {
  if (probs.length !== CLASSES.length) {
    throw new Error(`expected ${CLASSES.length} probabilities, got ${probs.length}`);
  }
  const out = new Float32Array(OUTPUTS.length);
  for (let i = 0; i < probs.length; i++) {
    out[CLASS_TO_OUTPUT[i]!]! += probs[i]!;
  }
  return out;
}

export function speechFor(output: string, lang: Lang = "id-ID"): string {
  const phrase = (schema.speech[lang] as Record<string, string>)[output];
  if (phrase === undefined) throw new Error(`no ${lang} phrase for ${output}`);
  return phrase;
}
