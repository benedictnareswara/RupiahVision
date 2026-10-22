# Public data sources

Audit of public Rupiah banknote datasets, checked on 2026-10-22. Public images are used for
**training only** (see [docs/data-collection.md](../docs/data-collection.md#how-splits-work)):
we can't tell which physical note each one shows, so none can go in val or test.

Counts are what each dataset page listed on that date. Roboflow totals can include
augmented copies (flips, crops, colour shifts) generated per version. The import step
removes near-duplicates by dHash, so expect fewer usable images than listed.

## Usable (license allows reuse)

| Dataset | Images listed | Classes | Series | Format | License |
|---|---|---|---|---|---|
| [Uang Rupiah Emisi 2022](https://universe.roboflow.com/bangkit-poc9z/uang-rupiah-emisi-2022-bw0in) (bangkit) | 1,350 | 7 denominations + one generic class | TE2022 (from the title) | detection | CC BY 4.0 |
| [Uang rupiah](https://universe.roboflow.com/mas-peri/uang-rupiah-fukxr) (mas peri) | 422 | 7 denominations | not stated | detection | CC BY 4.0 |
| [Indonesia Bank Notes Dataset](https://universe.roboflow.com/yolo-lex9h/indonesia-bank-notes-dataset) (yolo-lex9h) | 2,522 across 3 versions | 7 denominations | not stated, likely mixed | detection | CC BY 4.0 |

All three label only the denomination. To fit our 28 classes, each crop needs **series and
side labelled by hand**. Never let a model guess these labels, because its own mistakes
would then become training data. The mas peri set has the same image count (422) as the
TE2022 dataset in the [Jurnal UNIM paper](https://ejurnal.unim.ac.id/index.php/submit/article/download/4524/2226/16286),
which suggests they're the same images, but we haven't confirmed it.

CC BY 4.0 requires attribution: the README will credit each dataset we use, with a link.

## Low value (usable license, little to add)

| Dataset | Images listed | Why not |
|---|---|---|
| [Indonesian Banknotes](https://universe.roboflow.com/orbitaibanknotes/indonesian-banknotes) (orbitaibanknotes) | 335 | One class (`rupiah`), no denomination; published July 2022, before TE2022 existed |
| [Mata Uang Rupiah](https://universe.roboflow.com/rupiahcurrency/mata-uang-rupiah) (RupiahCurrency) | 33 | Only Rp1.000 and Rp10.000 |

## Not used: license unknown

Kaggle shows these as "Unknown" or "Other" with no terms given, so we have no permission
to redistribute or build on them. We'd use one only with the author's written OK, recorded
here.

| Dataset | Files listed | Notes |
|---|---|---|
| [rupiah banknotes 2022](https://www.kaggle.com/datasets/fzmhww8782/rupiah-22) (fzmhww8782) | 985 (2.24 GB) | Folders per denomination plus `background` and `multiclass`; full-size iPhone photos. The most useful one if the author agrees |
| [Rupiah Banknotes Dataset](https://www.kaggle.com/datasets/anidwiastuti/rupiah-banknotes-dataset) (Ani Dwi Astuti) | 2,398 (2.53 GB) | 7 folders; about 5 years old, so TE2016 or older |
| [Rupiah Banknotes](https://www.kaggle.com/datasets/nurulalfiyyah/rupiah-banknotes) (Nurul Alfiyyah) | 30 folders × 600 (182 MB) | Folders named by series, denomination and side (e.g. `2016-001B`); 600 small files per folder suggests augmentation |

## Import plan

1. Download the three usable Roboflow sets yourself (this needs a free Roboflow account).
   Pick the version **without augmentation** where one exists, in COCO or YOLO format.
   Put each in `data/public/<dataset>/`, which is git-ignored.
2. An import script (to be written) crops each box, drops crops too small to be useful,
   removes near-duplicates by dHash, and writes `data/public_<dataset>.csv` in manifest
   format with `source` set to `public:<dataset>`.
3. Label series and side for each crop by hand, with a small review tool. Crops you can't
   label confidently are dropped.
4. `uv run python -m rupiahvision.data.frames` merges every `data/public_*.csv` into
   `manifest.csv`, and `splits make` sends them all to train. The leakage check then also
   catches any public image that duplicates one of your own.
