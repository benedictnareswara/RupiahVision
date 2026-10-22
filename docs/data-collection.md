# Data collection protocol

How to photograph banknotes so the test set measures what really matters: does the model
recognize **a note it has never seen, in a place and light it has never seen**?

Everything you record goes into four small CSV files in [`data/`](../data). Scripts check
them, so a typo shows up as an error message instead of a silent leak.

| File | One row per | You write it |
|---|---|---|
| `notes.csv` | physical banknote | once per note, before any photos |
| `negatives.csv` | non-banknote object (receipt, book, …) | once per object |
| `sessions.csv` | capture session | before each session |
| `clips.csv` | video or photo | right after recording it |

## 1. Take inventory of your notes

Give every physical note an ID and add a row to `notes.csv`:

```csv
note_id,denomination,series,wear,owner,comment
N001,50000,TE2022,used,me,
N002,50000,TE2022,worn,mum,small tear at left edge
N003,2000,TE2016,new,bank,
```

- `note_id`: `N001`, `N002`, … Write the ID on a sticky note kept with the banknote.
  **Never record the serial number.**
- `denomination`: 1000, 2000, 5000, 10000, 20000, 50000 or 100000.
- `series`: `TE2016` or `TE2022` (tahun emisi, the year of issue printed on the note).
- `wear`: `new` (crisp, no folds), `used` (folds, soft paper), `worn` (stains, fading,
  tears, tape).
- `owner`: who it belongs to (`me`, `mum`, `bank`…), so borrowed notes get returned.

**Target: 3 notes for each of the 14 classes** (7 denominations × 2 series), so 42 notes.
With 3, a class gets separate train, val and test notes. Fewer still works, but the README
marks that class as less independent (see [splits](#how-splits-work)). Ways to fill gaps:

- check your own wallet and your family's;
- ask a bank teller for new notes (*uang baru*), which are mostly TE2022;
- keep change from shops, which brings in older, worn TE2016 notes;
- borrow from friends. Photographing takes about an hour per session, and the note
  goes straight back.

Then run, from `ml/`:

```bash
uv run python -m rupiahvision.data.splits plan
```

This gives every note a **role** (`train`, `val`, `test`, `trainval` or `any`), saved in
`data/subject_roles.csv`, and prints how many more notes each class needs. Roles never
change once given, so photos you've already taken stay valid when you add notes later.
Add notes **before** you start photographing a class if you can: a note that started as
the only one in its class (`any`) can't become a clean training note later without
removing some of its photos.

## 2. Add negatives

The model needs to see what "no banknote" looks like, so it can say *tidak dikenali*
(not recognized) instead of guessing. Add objects to `negatives.csv`:

```csv
object_id,object_type,comment
O001,receipt,supermarket receipt with the name cut off
O002,foreign_note,Malaysian 10 ringgit
O003,old_rupiah,1998 Rp10.000
```

Useful object types: `receipt`, `foreign_note`, `old_rupiah` (withdrawn series, which must
come out as unknown), `book_cover`, `magazine`, `leaflet`, `phone`, `wallet`, `paper`. A scene
with no object at all needs no row; in `clips.csv` it uses the subject `EMPTY`.

**Never photograph** bank cards, ID cards (KTP), or anything showing names, card numbers
or addresses.

To test truly unseen objects, list one or two object types under `open_set_object_types` in
`data/split_config.json` *before* photographing them. Those types then appear only in test
sessions.

## 3. Plan sessions

A **session** is one sitting at one location with one main light source, on one day.
Every session belongs to exactly one split, and all of its images go to that split.
Add a row to `sessions.csv` before you start:

```csv
session_id,split,date,location,device,comment
S001,train,2026-10-25,home_desk,iphone,evening under the ceiling light
S002,train,2026-10-26,living_room_floor,iphone,daylight from window
S003,test,2026-11-01,cafe_table,iphone,
```

- `split`: `train`, `val` or `test`.
- `location`: a short lowercase name. Reuse it whenever you're in the same spot.
- `device`: `iphone`, `webcam` or `other`.

**Targets:** at least 4 train, 2 val and 2 test sessions, each in a different place or
light. Hold test sessions in places you never use for training, such as a friend's house
or a café. The checker warns if a test location also appears in a train session.

**Which notes to bring.** The role printed by `plan` decides it:

| Session split | Notes with role | Negatives with role |
|---|---|---|
| train | `train`, `trainval`, `any` | `train`, `trainval`, `any` |
| val | `val`, `trainval`, `any` | `val`, `trainval`, `any` |
| test | `test`, `any` | `test`, `any`, `open_set` |

Photographing a note in the wrong kind of session is caught later by `splits make`.

## 4. Record clips

Use the iPhone (the app's target device) in **Video** mode at 1080p, 30 fps. Short videos
are much faster than taking stills: frames get extracted at 1–2 per second, and
near-identical frames are removed.

**Each clip: 8–10 seconds of one note in one state.** While recording, move slowly:
- tilt the phone;
- go from about 15 to 40 cm away;
- rotate the note;
- let it drift partly out of frame.

Hold the phone the way a blind user would: roughly above the note, not perfectly framed.

**Tip:** say the clip details out loud at the start ("N003, back, folded half"). The audio
makes `clips.csv` easy to fill in afterwards.

**Per note, per session:**
- 2 clips flat: one front, one back.
- 2 clips `folded_half`: front and back.
- 2 clips in one other state, rotating across sessions so every state appears in every
  split:
  - `folded_quarter`;
  - `crumpled`: crumple lightly in your hand, then open it out;
  - `covered`: 30–50% hidden by your hand or thumb.

**Per session, negatives:** 2–3 `EMPTY` clips of the bare surface or background, and 2–4
clips of negative objects.

**Lighting and background** are tagged per clip. Within a session, try to switch at least
one: lamp on or off, or note on the table vs. held in your hand.

| Tag | Values |
|---|---|
| `state` | `flat`, `folded_half`, `folded_quarter`, `crumpled`, `covered` |
| `lighting` | `daylight`, `indoor_warm`, `dim`, `backlit`, `glare` |
| `background` | `plain`, `cluttered`, `patterned`, `handheld` |

**Care for the notes.** Fold and crumple gently. Never cut, tear, write on or tape a note.
Indonesian currency law (UU 7/2011) prohibits deliberately damaging Rupiah. For borrowed
notes, use `crumpled` only with the owner's OK.

## 5. Log clips and check

Copy each session's videos (AirDrop) into `data/raw/<session_id>/`, keeping the phone's
file names. This folder is never committed. Add one row per clip to `clips.csv`; `file` is
the path under `data/raw/`:

```csv
file,session_id,subject_id,side,state,lighting,background
S001/IMG_4101.MOV,S001,N001,front,flat,indoor_warm,plain
S001/IMG_4102.MOV,S001,N001,back,folded_half,indoor_warm,plain
S001/IMG_4110.MOV,S001,O001,-,-,indoor_warm,cluttered
S001/IMG_4111.MOV,S001,EMPTY,-,-,indoor_warm,cluttered
```

Use `-` for `side` and `state` on anything that isn't a banknote.

After each capture day, from `ml/`:

```bash
uv run python -m rupiahvision.data.splits plan    # only if you added notes or objects
uv run python -m rupiahvision.data.frames         # extract frames, rebuild manifest.csv
uv run python -m rupiahvision.data.splits make    # assign splits, check for leaks
uv run pytest                                     # also validates every CSV
```

Frame extraction samples 2 frames per second from each video and drops frames that look
the same as the one before (the phone held still). Frames are saved to `data/frames/`,
with the longer side scaled to 1024 px. Clips that were already extracted are skipped, so
re-running is quick. If any clip can't be read, nothing is written and the error names it.

Videos are the main format. Photos also work as JPEG or PNG. iPhone photos are HEIC by
default, which OpenCV can't read: set **Settings > Camera > Formats > Most Compatible**
before taking stills.

Commit the CSV files and `results/data/dataset_summary.json`. Never commit images.

## How splits work

`rupiahvision.data.splits` applies two rules:

1. **Sessions never span splits.** A session's `split` decides where all its images go.
2. **Notes are planned per class**, from how many physical notes the class has:

   | Notes in class | Roles | Test separation in the README |
   |---|---|---|
   | 3 or more | separate test, val and train notes | note-disjoint |
   | 2 | one test note; the other for train and val | test note-disjoint, val by session |
   | 1 | the only note, in every split | session only |

The separation level in the README comes from **the images actually captured**, not the
plan. `splits make` also compares a 64-bit image hash (dHash) across splits and refuses to
continue if two images in different splits are near-duplicates. CI runs
`splits check` on the committed files.

Public-dataset images always go to train: there's no way to know which physical note they
show, so they could be the same note as one of your test images.
