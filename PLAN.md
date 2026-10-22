# RupiahVision: Plan

## Context

This rebuilds a lost project: an app that recognizes Indonesian Rupiah banknotes and announces the denomination aloud for visually impaired users. It is a portfolio piece for software/ML internships, so a reviewer should find a working demo, honest and reproducible evaluation, and clean engineering.

What we agreed:
- **Time:** about 2–3 months part-time (~8–12 h/week, ~100 h). Core work is sized to roughly 75 h, which leaves slack and room for some stretch goals.
- **Hardware:** an iPhone and a Mac laptop with a webcam. No Android phone and no Orange Pi or Raspberry Pi.
- **Notes:** you have physical notes for only some denominations and series. Collecting the rest is the biggest project risk, so it starts in week 1.
- **Users:** you have no access to visually impaired testers. We test with blindfolded sighted people and VoiceOver, and the README says exactly that.

Prior art to stand out from: there is already a public Android + MobileNetV3 + TTS project (`aracel1234/rupiah-banknote-recognition-mobilenetv3`). This project sets itself apart with a try-in-browser demo, a leak-free test split, a well-tested "unknown" output, stress-condition results, and latency measured on real devices.

Rules for every milestone:
- Every number in the README comes from a script in the repo that writes a JSON file under `results/`.
- README tables are generated from those JSON files, and CI fails if the README and the JSON disagree.
- We never write estimated or placeholder numbers. A table cell with no measurement yet says "not measured".

---

## 1. Target platform: recommend **(b) PWA**

| | (a) Android app | (b) PWA, in-browser model | (c) Orange Pi / RPi device |
|---|---|---|---|
| Accessibility for blind users | Best: TalkBack, full haptics, native TTS | Good: VoiceOver/TalkBack, Web Speech, audio cues. **No haptics on iOS Safari**¹ | Excellent physical UX (one button, speaker, always-on), but needs a dedicated device |
| Latency | Lowest (TFLite + GPU/NNAPI) | Good: ONNX Runtime Web on WASM SIMD; WebGPU on iOS 26+ / Chrome | Moderate: CPU/NPU, ~depends on board |
| Effort for you | High: Kotlin/Android is new, **and you have no Android device** | Lowest: you already know TS/React | Medium: you know the hardware, but you'd need to buy a board + camera + enclosure |
| Recruiter can try it | Must sideload an APK: few will | **Open a URL**, works on laptop and phone | Video only |

¹ To my knowledge Safari has not implemented `navigator.vibrate`. We confirm this in M4 and fall back to audio cues on iOS. Android Chrome supports vibration, so it gets haptics.

**Recommendation: PWA.** It is the only option you can build and test on the hardware you have, it uses a stack you know, and a recruiter can try it in 10 seconds. A native iOS app (Swift + Core ML) would give better haptics, but it is new tech for you and recruiters would need TestFlight to try it. That makes it a stretch goal, along with an Orange Pi port, which can reuse the same ONNX model.

*New concepts, explained as we reach them:* a **PWA** is a website with a manifest and a service worker, so it can be installed to the home screen and work offline. **ONNX Runtime Web** runs an exported `.onnx` model in the browser through WebAssembly or WebGPU. The camera needs HTTPS, so we deploy to GitHub Pages or Cloudflare Pages.

---

## 2. Scope

- **Denominations:** Rp1.000, 2.000, 5.000, 10.000, 20.000, 50.000 and 100.000, in both the **TE 2016** and **TE 2022** series. In M1 we check this list against Bank Indonesia's current legal-tender list. The Rp75.000 commemorative note (2020) is a stretch goal. Until then the app answers "tidak dikenali" (not recognized) for it, which is a safe failure.
- **Training labels are finer than what the app says.** We train on `denomination × series × side`, which is 28 classes, plus `unknown`. At inference the app sums probabilities per denomination and announces only the denomination. Keeping series and side as separate labels makes errors easier to analyze, for example "the model confuses the backs of the 2022 Rp2.000 and Rp20.000", and gives each class a cleaner visual identity.
- **"Unknown / no note" uses three defenses:**
  1. an explicit `unknown` class trained on negative images: empty tables, hands, receipts, cards, foreign notes and old rupiah if you have any;
  2. a confidence threshold, calibrated with temperature scaling (one learned scalar that makes softmax probabilities honest), chosen on the validation set to keep wrong announcements under a target rate;
  3. temporal voting: the app speaks only after at least K of the last N frames agree above the threshold.
- **Out of scope:** coins, counterfeit detection, counting several notes at once.

---

## 3. Data

**Public sources (training only, audited in M1):**
- Kaggle "Uang emisi 2022 baru"
- Roboflow Universe: `orbitaibanknotes/indonesian-banknotes`, `yolo-lex9h/indonesia-bank-notes-dataset`, `rupiahcurrency/mata-uang-rupiah`
- the 2022-series dataset described in the Jurnal UNIM paper

These are mostly detection datasets. We crop them to their bounding boxes and relabel by series and side. The license and image count of each source are recorded in `data/SOURCES.md`.

We don't know which physical note each public image shows, so public images **never go in the test set**. At most they form a separate "external" test set, reported apart and with that caveat.

**Your own photos (the core of the dataset):**
- **Inventory first (M1, week 1):** `data/notes.csv` gets one row per physical note: `note_id, denomination, series, wear (new/used/worn), source`. We record a note ID, never the serial number. Then fill the gaps: ask family and friends, and swap at a bank or money changer for 2022-series notes.
- **Minimum target:** 3 physical notes per class (1 each for train, val and test). Where that isn't possible, the class uses a session-only split, and the results table marks it as such.
- **Capture protocol:** use the iPhone as the main camera, since it is the target device, plus the laptop webcam. Record short videos and extract frames at about 1–2 fps, which is faster than taking stills. Deduplicate near-identical frames with perceptual hashing.
- **Session:** one sitting at one location and lighting, on one day. Every image carries `note_id`, `session_id` and condition tags.
- **Conditions grid**, tagged per clip:
  - side: front / back
  - state: flat, folded in half, folded in quarters, crumpled, partly covered by a hand (~30–50%)
  - wear: from the inventory
  - lighting: daylight, warm indoor, dim, backlit, glare
  - background: plain, cluttered desk, patterned cloth, held in hand
- **Negatives:** photograph the same scenes without a note, plus look-alike objects. For an open-set test, we hold out some object *types* entirely; they never appear in training.

**Splits:** `scripts/make_splits.py` is deterministic and seeded.
- The **test** set is whole held-out physical notes, with all their sessions.
- The **val** set is held-out notes where we have enough, otherwise held-out sessions.
- A pytest check asserts that no `note_id` or `session_id` appears in more than one split, and that no near-duplicate perceptual hash crosses splits. The hash check also catches public images that duplicate your own.

**Storage and reproducibility:**
- Images stay out of git. Git holds only the manifests.
- The data gets published so others can rerun the scripts: core = at least the val and test splits on Hugging Face Datasets; stretch = the full dataset. EXIF is stripped before publishing.
- Before publishing, check Indonesia's rules on reproducing images of rupiah (UU 7/2011 on currency). Low-resolution research photos are common, but confirm first.

---

## 4. Model

- **Baseline:** a small custom CNN (4–5 conv blocks, BN, GAP), close to what you built before.
- **Transfer learning:** MobileNetV3-Small and MobileNetV3-Large (torchvision), plus EfficientNet-Lite0 (timm). All start from ImageNet weights, train the classifier head first, then fine-tune the whole network.
- **Augmentation** fills gaps in the photo set: random resized crop, rotation, perspective, color jitter, blur, JPEG compression, and random occlusion (cutout) to mimic fingers.
- **Training** runs on the Mac with PyTorch's MPS backend, with Kaggle or Colab GPUs as a fallback. A YAML config plus a fixed seed define each run. Run logs and metrics go to JSON. Weights & Biases is optional.
- **Choosing the model:** pick on validation macro accuracy at the chosen coverage, then on in-browser latency and model size. The test set is used **once**, for the final models.
- **Export:**
  - PyTorch → ONNX (opset 17), with a parity test: ONNX Runtime output must match PyTorch within a set tolerance on fixed inputs.
  - Stretch: int8 quantization, with its accuracy and latency change reported.
  - Stretch: TFLite, for an Orange Pi or Android port.

---

## 5. Evaluation (every row produced by a script → `results/*.json`)

- `ml/eval/evaluate.py` produces, on the test set:
  - **Per-class accuracy** and macro accuracy, with **95% bootstrap confidence intervals.** The test sets are small, so the intervals show how much a number could move with a different sample. They are computed by resampling the test images about 1,000 times.
  - **Confusion matrices**, at both 28+1 and 7+1 classes, saved as PNG.
- **Per stress condition:** accuracy for each tag (folded, covered, dim, glare, cluttered, worn, and so on), with confidence intervals and n.
- **"Unknown" threshold:**
  - a risk–coverage curve, where coverage is the share of notes the app answers and risk is the share of those answers that are wrong;
  - the chosen threshold on val, then on test: coverage, error rate among answers, and false-announcement rate on negatives and on open-set negatives;
  - the same metrics after temporal voting, simulated on frame sequences from the test videos.
- **Latency:**
  - a `/bench` page in the PWA runs warmup plus N inferences. It reports median and p95 for model time and for frame-to-decision time, along with device, browser and backend (WASM or WebGPU), and exports JSON.
  - On the laptop, Playwright runs it automatically.
  - On the iPhone, you open the page and it saves the JSON, which we commit. The README says which numbers came from this manual run.
- **Model comparison table:** baseline vs. transfer models on val accuracy, test accuracy, parameter count, ONNX size, and iPhone and laptop latency.

---

## 6. Accessibility

- **Hands-free:** the camera starts on open, and capture and inference loop at about 5–10 fps.
  - A quality gate (blur via Laplacian variance, brightness check) skips bad frames and gives spoken hints such as "terlalu gelap" (too dark).
  - Temporal voting decides when to speak. No button press is needed.
  - Each denomination is announced once, then the app waits until the note changes or leaves the frame before speaking again.
- **Speech in Indonesian, e.g. "Seratus ribu rupiah":**
  - The app uses the Web Speech API with an `id-ID` voice when the device has one.
  - Otherwise it plays bundled audio clips. A script generates them, or you record them yourself.
  - Optional English. Repeat-last on a tap anywhere on screen.
- **Aiming cues:**
  - Web Audio ticks whose rate rises as confidence rises, like a Geiger counter.
  - Vibration where supported (Android Chrome).
  - A distinct "earcon" (a short signature sound) right before an announcement.
  - Stretch: torch toggle where the browser supports it.
- **Screen readers:** the app works with VoiceOver on and off. A full-screen layout, an `aria-live` region and large targets keep it usable for low-vision users. We audit against WCAG 2.1 AA.
- **Testing without blind participants (stated honestly in the README):**
  - First, VoiceOver walkthroughs.
  - Then a pilot with 3–5 **blindfolded sighted** participants. Each gets a shuffled set of notes and conditions. We measure task success, time-to-announcement, wrong announcements, and the SUS questionnaire, logged anonymously to JSON.
  - The protocol and consent form live in `docs/user-testing.md`, ready for a real study.
  - Stretch: contact Pertuni (the Indonesian Blind Union) or an SLB-A school (school for blind students) about a real study.

---

## 7. Engineering

```
rupiahvision/
  ml/                     # Python package (pyproject, src layout)
    src/rupiahvision/
      data/   (manifest, splits, dataset, transforms)
      models/ (custom_cnn, transfer)
      train.py  eval/  export/
    tests/                # pytest: splits/leakage, transforms, model shapes, ONNX parity, metrics
    configs/              # YAML per experiment
  web/                    # Vite + React + TS PWA
    src/ (camera, inference worker, voting, speech, audio cues, bench page)
    tests/                # vitest unit tests; Playwright smoke test
  data/                   # notes.csv, sessions.csv, SOURCES.md, manifests (no images)
  scripts/                # make_splits, extract_frames, dedupe, make_readme_tables, gen_audio
  results/                # committed JSON + plots produced by scripts
  docs/                   # architecture, data protocol, user-testing protocol
  .github/workflows/      # ci.yml, deploy.yml
```

- **Python tooling:** `uv` for environments, `ruff` for lint and formatting, `pytest`.
  - Tests run on a tiny synthetic fixture dataset, so CI never needs the real images.
- **Web tooling:**
  - ESLint, `tsc --noEmit`, `vitest`.
  - Inference runs in a Web Worker so the UI stays responsive.
- **CI (GitHub Actions):**
  - Python lint and tests, and an ONNX export plus parity test on a tiny model.
  - Web typecheck, tests and build.
  - `make_readme_tables.py --check`.
- **Deploy:** a workflow publishes the PWA to GitHub Pages. The model `.onnx` ships as a release asset or a static file.
- **README:**
  - pitch and demo GIF (screen recording from the iPhone);
  - "try it" link and architecture diagram;
  - setup, data and reproduce commands;
  - generated results tables, an honest limitations section, and how the app was tested.

---

## Milestones (~10 weeks at ~10 h/week)

| # | Milestone | Weeks | Core deliverables | Done when |
|---|---|---|---|---|
| M0 | Skeleton | 1 | Repo layout, pyproject, Vite app shell, CI green, label schema | CI passes on an empty pipeline |
| M1 | Data v1 | 1–3 | Inventory + gap-filling, capture protocol, ~2–3 sessions per note, frame extraction + dedupe, public data audit, `make_splits.py` + leakage tests | Manifests committed; leakage tests pass; per-class counts table generated |
| M2 | Baseline | 3–4 | Dataset/transforms, training loop, custom CNN, `evaluate.py` with per-class accuracy, confusion matrix, confidence intervals | Baseline val results in `results/`, generated into README |
| M3 | Transfer + unknown | 4–6 | MobileNetV3 S/L + EfficientNet-Lite0, temperature scaling, threshold selection, stress-condition eval, voting simulation, final test run | Model comparison + stress + risk–coverage tables generated |
| M4 | PWA | 6–8 | ONNX export + parity test, worker inference, camera loop, quality gate, voting, speech + audio cues, offline service worker, `/bench`, deploy | Works hands-free on iPhone Safari; latency JSON from iPhone and laptop committed |
| M5 | Accessibility testing | 8–9 | VoiceOver pass, WCAG audit, blindfolded pilot (3–5 people), fixes | Pilot results JSON + summary in README |
| M6 | Polish | 9–10 | README, demo GIF, architecture diagram, limitations, publish val/test split to Hugging Face | A stranger can clone, download data, rerun eval and get the README numbers |

**Stretch goals, roughly by value:**
1. int8 quantization study
2. full public dataset on Hugging Face
3. real study via Pertuni or an SLB school
4. Rp75.000 commemorative note
5. Grad-CAM view of what the model looks at
6. a note-detection or cropping stage
7. native iOS (Core ML) app for comparison
8. Orange Pi port with a physical button and speaker
9. torch control

**Main risk:** missing physical notes for some classes. We handle it by starting collection in week 1, using public data for training only, and marking any session-only split class in the results table, so no result is overstated.

---

## Verification (how we check the plan is working)

- `uv run pytest` and `npm test` pass locally and in CI at every milestone.
- `python scripts/make_readme_tables.py --check` passes, which means the README matches `results/`.
- From a clean clone, the documented reproduce commands regenerate `results/*.json`, and the numbers match the README.
- The deployed PWA opens from its URL on the iPhone, announces a held-out test note hands-free in Indonesian, and stays silent on an empty table.

## First step after approval

Write this plan to `PLAN.md` in the repo, `git init`, and start M0.
