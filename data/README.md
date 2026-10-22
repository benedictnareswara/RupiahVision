# data

Manifests and the label schema. Images and videos are never committed; they live in
`data/raw/` (git-ignored). See [docs/data-collection.md](../docs/data-collection.md) for how
to fill these in.

| File | Written by | Contents |
|---|---|---|
| `labels.json` | hand | class list, spoken phrases, condition tags |
| `split_config.json` | hand | split seed, val/test fractions, open-set object types, dHash threshold |
| `notes.csv` | hand | one row per physical banknote |
| `negatives.csv` | hand | one row per non-banknote object |
| `sessions.csv` | hand | one row per capture session, including its split |
| `clips.csv` | hand | one row per recorded video or photo |
| `subject_roles.csv` | `splits plan` | which splits each note/object may appear in |
| `manifest.csv` | `rupiahvision.data.frames` | one row per image, with its dHash |
| `public_<dataset>.csv` | public import (planned) | public images in manifest format, train only |
| `SOURCES.md` | hand | audit of public datasets: licenses, counts, what's usable |
| `splits.csv` | `splits make` | the split of every image |
