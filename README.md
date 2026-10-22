# RupiahVision

Hands-free Indonesian Rupiah banknote recognition for blind and low-vision users. Point a phone
camera at a note and the app speaks its denomination in Indonesian. If it isn't confident, it
stays silent or says "Tidak dikenali" (not recognized), never a guess. The app runs fully in the
browser as an installable web app (PWA), so trying it needs only a link.

> **Status:** early development (milestone M0 of [PLAN.md](PLAN.md)). There is no trained model
> and there are no results yet.

## What it recognizes

All banknotes from the 2016 (TE2016) and 2022 (TE2022) series. The model is trained on 28
classes (denomination × series × side) plus `unknown`; the app announces only the denomination.
This table is generated from [`data/labels.json`](data/labels.json):

<!-- BEGIN GENERATED: classes -->
| Denomination | Series | Spoken (id-ID) | Spoken (en-US) |
|---|---|---|---|
| Rp1.000 | TE2016, TE2022 | Seribu rupiah | One thousand rupiah |
| Rp2.000 | TE2016, TE2022 | Dua ribu rupiah | Two thousand rupiah |
| Rp5.000 | TE2016, TE2022 | Lima ribu rupiah | Five thousand rupiah |
| Rp10.000 | TE2016, TE2022 | Sepuluh ribu rupiah | Ten thousand rupiah |
| Rp20.000 | TE2016, TE2022 | Dua puluh ribu rupiah | Twenty thousand rupiah |
| Rp50.000 | TE2016, TE2022 | Lima puluh ribu rupiah | Fifty thousand rupiah |
| Rp100.000 | TE2016, TE2022 | Seratus ribu rupiah | One hundred thousand rupiah |
| (no note / not confident) | - | Tidak dikenali | Not recognized |
<!-- END GENERATED: classes -->

## Dataset

Photos of real notes, taken following the [capture protocol](docs/data-collection.md). Splits
are made per physical note and per capture session, never per random image, so the test set
shows notes, places and lighting the model never trained on. Public-dataset images are used
for training only, because there's no way to tell which physical note each one shows.

**Test separation**, measured from the images rather than the plan:

- **note-disjoint**: every test note is a physical note that never appears in train or val.
- **test note-disjoint, val by session**: the same, except val reuses the training note in
  other sessions (classes with only two notes).
- **session only**: the class has one note, so train, val and test differ only by session.
  Accuracy for these classes is likely optimistic.

<!-- BEGIN GENERATED: dataset -->
| Denomination | Series | Notes | Test separation | Train | Val | Test | Public (train only) |
|---|---|---|---|---|---|---|---|
| Rp1.000 | TE2016 | 0 | no notes yet | 0 | 0 | 0 | 0 |
| Rp1.000 | TE2022 | 0 | no notes yet | 0 | 0 | 0 | 0 |
| Rp2.000 | TE2016 | 0 | no notes yet | 0 | 0 | 0 | 0 |
| Rp2.000 | TE2022 | 0 | no notes yet | 0 | 0 | 0 | 0 |
| Rp5.000 | TE2016 | 0 | no notes yet | 0 | 0 | 0 | 0 |
| Rp5.000 | TE2022 | 0 | no notes yet | 0 | 0 | 0 | 0 |
| Rp10.000 | TE2016 | 0 | no notes yet | 0 | 0 | 0 | 0 |
| Rp10.000 | TE2022 | 0 | no notes yet | 0 | 0 | 0 | 0 |
| Rp20.000 | TE2016 | 0 | no notes yet | 0 | 0 | 0 | 0 |
| Rp20.000 | TE2022 | 0 | no notes yet | 0 | 0 | 0 | 0 |
| Rp50.000 | TE2016 | 0 | no notes yet | 0 | 0 | 0 | 0 |
| Rp50.000 | TE2022 | 0 | no notes yet | 0 | 0 | 0 | 0 |
| Rp100.000 | TE2016 | 0 | no notes yet | 0 | 0 | 0 | 0 |
| Rp100.000 | TE2022 | 0 | no notes yet | 0 | 0 | 0 | 0 |
| unknown (no note) | - | 0 objects | - | 0 | 0 | 0 | - |

Total: 0 train / 0 val / 0 test images from 0 / 0 / 0 capture sessions.
<!-- END GENERATED: dataset -->

## Results

No results yet. Every number that appears here will be generated from a JSON file in
[`results/`](results/) by a script in this repo. See [Reproducing](#reproducing).

## Repository layout

```
ml/        Python package: data, training, evaluation, export (pytest tests in ml/tests)
web/       Vite + React + TypeScript PWA (vitest tests in web/src)
data/      Label schema and dataset manifests (images are not stored in git)
results/   Metrics and plots written by scripts; the README tables are generated from these
docs/      Architecture, data collection and user-testing protocols
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Node.js 24+.

```bash
cd ml && uv sync          # Python 3.12 env with dev tools
uv run pytest             # ML tests
cd ../web && npm ci       # web dependencies
npm test                  # web tests
npm run dev               # local dev server
```

## Reproducing

All commands run from `ml/`.

Rebuild the dataset splits and check them for leaks (writes `data/splits.csv` and
`results/data/dataset_summary.json`):

```bash
uv run python -m rupiahvision.data.splits make
```

Regenerate the README tables from the files in the repo:

```bash
uv run python -m rupiahvision.reporting.readme_tables
```

CI runs `splits check` and `readme_tables --check`, and fails if either the splits or the
README are out of date.
