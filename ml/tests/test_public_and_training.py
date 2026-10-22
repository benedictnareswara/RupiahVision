"""Import -> split -> train -> evaluate -> export, on tiny fake Kaggle and Roboflow exports."""

import json
import zipfile

import cv2
import numpy as np
import onnxruntime as ort
import pytest
import torch

from rupiahvision import public
from rupiahvision.data.inventory import read_csv
from rupiahvision.reporting import readme_tables
from rupiahvision.training.evaluate import evaluate
from rupiahvision.training.export import export, publish
from rupiahvision.training.train import TrainConfig, train

DENOMINATIONS = ("1000", "50000", "100000")


def scene(seed: int, size=(96, 128)) -> np.ndarray:
    small = np.random.default_rng(seed).integers(0, 256, (6, 8, 3), dtype=np.uint8)
    return cv2.resize(small, size[::-1], interpolation=cv2.INTER_CUBIC)


def fake_kaggle(root):
    """jpg<denomination>/IMG_<n>.jpg in runs of numbers, plus background and multiclass."""
    seed = 0
    for folder in [*(f"jpg{d}" for d in DENOMINATIONS), "jpgbackground", "jpgmulticlass"]:
        (root / folder).mkdir(parents=True)
        for n in range(40):
            seed += 1
            cv2.imwrite(str(root / folder / f"IMG_{6000 + 100 * seed // 40 + n}.jpg"), scene(seed))
    return root


def fake_roboflow_zip(path):
    """COCO export with a parent category and an augmented copy of every image."""
    categories = [{"id": 0, "name": "Uang-Rupiah-Emisi-2022"}] + [
        {"id": i + 1, "name": f"{int(d) // 1000}k"} for i, d in enumerate(DENOMINATIONS)
    ]
    images, annotations = [], []
    with zipfile.ZipFile(path, "w") as archive:
        for n in range(30):
            canvas = scene(5000 + n, size=(480, 640))
            ok, encoded = cv2.imencode(".jpg", canvas)
            for copy in ("aaa", "bbb"):  # Roboflow-style augmented copies of one original
                name = f"IMG_{n:04d}_jpg.rf.{copy}{n}.jpg"
                archive.writestr(f"train/{name}", encoded.tobytes())
                images.append({"id": len(images), "file_name": name, "width": 640, "height": 480})
                annotations.append(
                    {
                        "id": len(annotations),
                        "image_id": images[-1]["id"],
                        "category_id": 1 + n % 3,
                        "bbox": [400, 250, 200, 180],
                    }
                )
        coco = {"categories": categories, "images": images, "annotations": annotations}
        archive.writestr("train/_annotations.coco.json", json.dumps(coco))
    return path


def test_group_key():
    assert public.group_key("IMG_6643") == "IMG_664"
    assert public.group_key("IMG_0012_jpg.rf.9f8e7d") == "IMG_1"
    assert public.group_key("photo") == "photo"
    assert public.group_key("UANG-RUPIAH-50K-123-_jpg.rf.abc") == "UANG-RUPIAH-50K-12"
    whatsapp = "WhatsApp-Image-2023-07-30-at-17-17-28-1-_jpeg.rf.4adab0"
    assert public.group_key(whatsapp) == "WhatsApp-Image-2023-07-30-at-17-17"


def test_roboflow_label():
    assert public.roboflow_label("50k") == "50000"
    assert public.roboflow_label("Uang-Rupiah-Emisi-2022") is None


@pytest.fixture(scope="module")
def imported(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("public")
    data_dir = tmp / "data"
    common = ["--data-dir", str(data_dir), "--summary", str(tmp / "summary.json")]
    assert public.main([*common, "import-kaggle", str(fake_kaggle(tmp / "kaggle"))]) == 0
    assert public.main([*common, "import-roboflow", str(fake_roboflow_zip(tmp / "rf.zip"))]) == 0
    assert public.main([*common, "split"]) == 0
    return tmp, data_dir


def test_import_and_split(imported):
    tmp, data_dir = imported
    rows = read_csv(public.index_path(data_dir), public.INDEX_COLUMNS)
    kaggle = [r for r in rows if r["dataset"] == public.KAGGLE]
    roboflow = [r for r in rows if r["dataset"] == public.ROBOFLOW]

    assert {r["label"] for r in kaggle} == {*DENOMINATIONS, "unknown"}  # multiclass skipped
    # one note crop + one background crop per original; augmented copies dropped
    assert sum(r["label"] != "unknown" for r in roboflow) == 30
    assert sum(r["label"] == "unknown" for r in roboflow) == 30

    split_of_group = {}
    for row in rows:
        assert split_of_group.setdefault(row["group"], row["split"]) == row["split"]
        assert (data_dir / row["path"]).exists()
    summary = json.loads((tmp / "summary.json").read_text())
    for label in (*DENOMINATIONS, "unknown"):
        assert all(summary["labels"][label][s] > 0 for s in ("train", "val", "test")), label


def test_groups_spanning_labels_still_give_every_label_val_and_test():
    # Sessions that photograph two denominations each, like the WhatsApp bursts.
    rows = []
    for session in range(12):
        for label in (("1000", "2000"), ("2000", "5000"), ("5000", "1000"))[session % 3]:
            rows += [
                {"path": f"{session}-{label}-{i}.jpg", "label": label, "dataset": "d",
                 "group": f"s{session}", "dhash": f"{session * 100 + i:016x}"}
                for i in range(10)
            ]  # fmt: skip
    split_of = public.assign_groups(rows)
    for label in ("1000", "2000", "5000"):
        splits = {split_of[r["group"]] for r in rows if r["label"] == label}
        assert splits == {"train", "val", "test"}, label


def test_one_huge_group_is_not_sent_to_test():
    rows = [{"path": f"big{i}", "label": "1000", "dataset": "d", "group": "big"} for i in range(80)]
    rows += [{"path": f"s{g}{i}", "label": "1000", "dataset": "d", "group": f"s{g}"}
             for g in range(8) for i in range(10)]  # fmt: skip
    assert public.assign_groups(rows)["big"] == "train"


def test_label_with_three_big_groups_gets_every_split():
    # Like Rp2.000 in the Roboflow set: three sessions of 61, 50 and 31 photos.
    rows = [{"path": f"{g}{i}", "label": "2000", "dataset": "d", "group": g}
            for g, n in (("a", 61), ("b", 50), ("c", 31)) for i in range(n)]  # fmt: skip
    rows += [{"path": f"o{g}{i}", "label": "1000", "dataset": "d", "group": f"o{g}"}
             for g in range(10) for i in range(10)]  # fmt: skip
    split_of = public.assign_groups(rows)
    assert {split_of[g] for g in "abc"} == {"train", "val", "test"}


def test_split_drops_held_out_near_duplicates():
    rows = [
        {"path": f"{g}.jpg", "label": "1000", "dataset": "d", "group": g, "dhash": "0" * 16}
        for g in "abcdefghij"
    ]
    kept, notes = public.split_rows(rows)
    assert {r["split"] for r in kept} == {"train"}
    assert notes and "dropped" in notes[0]


@pytest.mark.parametrize("model", ["custom_cnn", "mobilenet_v3_small"])
def test_train_evaluate_export(imported, model):
    tmp, data_dir = imported
    out = tmp / model
    config = TrainConfig(
        model=model, epochs=1, batch_size=16, image_size=64, workers=0, pretrained=False
    )
    record = train(config, data_dir, out / "models", out / "results" / "runs", torch.device("cpu"))
    assert record["history"][0]["epoch"] == 1
    assert (out / "models" / model / "checkpoint.pt").exists()

    result = evaluate(
        model, 0.05, data_dir, out / "models", out / "results" / "eval", torch.device("cpu")
    )
    assert 0.0 <= result["test"]["accuracy"] <= 1.0
    low, high = result["test"]["accuracy_ci95"]
    assert low <= result["test"]["accuracy"] <= high
    assert len(result["test"]["confusion_matrix"]) == 8
    assert (out / "results" / "eval" / f"{model}_confusion.png").exists()

    exported = export(model, out / "models", out / "results" / "eval", out / "results" / "export")
    # Weights are embedded: one file, at least 4 bytes per float32 parameter.
    assert not (out / "models" / model / "model.onnx.data").exists()
    assert exported["onnx_bytes"] >= 4 * record["parameters"]
    assert exported["max_abs_logit_difference"] < 1e-3
    metadata = json.loads((out / "models" / model / "model.json").read_text())
    assert metadata["outputs"][0] == "unknown" and len(metadata["outputs"]) == 8
    assert metadata["temperature"] == result["calibration"]["temperature"]
    session = ort.InferenceSession(str(out / "models" / model / "model.onnx"))
    logits = session.run(["logits"], {"input": np.zeros((1, 3, 64, 64), np.float32)})[0]
    assert logits.shape == (1, 8)

    publish(model, out / "models", out / "web")
    assert {p.name for p in (out / "web").iterdir()} == {"model.onnx", "model.json"}

    # The README tables render from these results (out/ has the repo's results/ layout).
    assert f"`{model}`" in readme_tables.models_table(out)
    per_class = readme_tables.per_class_table(out)
    assert f"Selected model (best validation accuracy): `{model}`" in per_class
    assert "| 100000 |" in per_class
