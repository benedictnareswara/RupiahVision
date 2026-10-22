import json

import numpy as np
import pytest

from rupiahvision.labels import DEFAULT_SCHEMA_PATH, load_schema


@pytest.fixture(scope="module")
def schema():
    return load_schema()


def test_class_count(schema):
    # 7 denominations x 2 series x 2 sides + unknown
    assert schema.num_classes == 7 * 2 * 2 + 1
    assert schema.classes[0] == schema.unknown_class


def test_outputs_are_unknown_then_denominations(schema):
    assert schema.outputs == (
        "unknown",
        "1000",
        "2000",
        "5000",
        "10000",
        "20000",
        "50000",
        "100000",
    )


def test_denomination_of(schema):
    assert schema.denomination_of("unknown") is None
    assert schema.denomination_of("100000_TE2022_back") == 100000
    with pytest.raises(KeyError):
        schema.denomination_of("75000_TE2020_front")


def test_aggregation_matrix_maps_each_class_to_one_output(schema):
    matrix = schema.aggregation_matrix()
    assert matrix.shape == (schema.num_classes, len(schema.outputs))
    np.testing.assert_array_equal(matrix.sum(axis=1), 1.0)
    # each denomination gathers series x sides = 4 classes; unknown gathers 1
    np.testing.assert_array_equal(matrix.sum(axis=0), [1] + [4] * len(schema.denominations))


def test_aggregate_sums_probabilities(schema):
    probs = np.zeros((2, schema.num_classes), dtype=np.float32)
    probs[0, schema.index_of("50000_TE2016_front")] = 0.5
    probs[0, schema.index_of("50000_TE2022_back")] = 0.3
    probs[0, schema.index_of("unknown")] = 0.2
    probs[1, schema.index_of("1000_TE2022_front")] = 1.0
    out = schema.aggregate(probs)
    np.testing.assert_allclose(out.sum(axis=1), 1.0)
    assert out[0, schema.outputs.index("50000")] == pytest.approx(0.8)
    assert out[0, 0] == pytest.approx(0.2)
    assert schema.outputs[int(out[1].argmax())] == "1000"


def test_aggregate_rejects_wrong_width(schema):
    with pytest.raises(ValueError):
        schema.aggregate(np.zeros(5))


def test_every_output_has_speech(schema):
    for lang in ("id-ID", "en-US"):
        assert set(schema.speech[lang]) == set(schema.outputs)


def test_reordered_classes_are_rejected(tmp_path):
    raw = json.loads(DEFAULT_SCHEMA_PATH.read_text())
    raw["classes"][1], raw["classes"][2] = raw["classes"][2], raw["classes"][1]
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="canonical class order"):
        load_schema(path)


def test_missing_speech_is_rejected(tmp_path):
    raw = json.loads(DEFAULT_SCHEMA_PATH.read_text())
    del raw["speech"]["id-ID"]["20000"]
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="20000"):
        load_schema(path)
