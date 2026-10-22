import cv2
import numpy as np

from rupiahvision.data.dhash import dhash, near_duplicates, to_hex


def scene(seed: int) -> np.ndarray:
    """A smooth random BGR image, so small edits keep its structure."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, size=(12, 16, 3), dtype=np.uint8)
    return cv2.resize(small, (640, 480), interpolation=cv2.INTER_CUBIC)


def distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def test_hash_is_64_bit_hex():
    assert len(to_hex(dhash(scene(0)))) == 16
    assert to_hex(0) == "0" * 16


def test_survives_resize_jpeg_and_brightness():
    original = scene(1)
    resized = cv2.resize(original, (320, 240), interpolation=cv2.INTER_AREA)
    _, encoded = cv2.imencode(".jpg", original, [cv2.IMWRITE_JPEG_QUALITY, 40])
    jpeg = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    brighter = cv2.convertScaleAbs(original, alpha=1.0, beta=20)
    h = dhash(original)
    for variant in (resized, jpeg, brighter):
        assert distance(h, dhash(variant)) <= 4


def test_different_images_are_far_apart():
    assert distance(dhash(scene(1)), dhash(scene(2))) > 10


def test_grayscale_input():
    assert dhash(cv2.cvtColor(scene(3), cv2.COLOR_BGR2GRAY)) == dhash(scene(3))


def test_near_duplicates_across_blocks():
    a = ["0" * 16, "f" * 16, "00000000000000ff"]
    b = ["0000000000000001", "ffffffffffffff00", "00000000ffffffff"]
    # block=1 forces the chunked path to stitch indices back together correctly
    assert near_duplicates(a, b, max_distance=8, block=1) == [(0, 0, 1), (1, 1, 8), (2, 0, 7)]
    assert near_duplicates(a, [], max_distance=8) == []
