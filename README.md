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

Regenerate the README tables from the files in the repo:

```bash
cd ml && uv run python -m rupiahvision.reporting.readme_tables
```

CI runs the same command with `--check` and fails if the README is out of date.
