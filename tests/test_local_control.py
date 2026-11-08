from hashlib import sha256
import json

from PIL import Image
import pytest
import torch

from infer import ROOT, UnderwaterEnhancer, file_sha256, rgb_image
from local_control import limit_local_enhancement


@pytest.mark.parametrize("smooth", [False, True])
@pytest.mark.parametrize("shape", [(1, 1), (17, 19)])
def test_control_stays_in_range_and_zero_strength_is_identity(smooth, shape):
    height, width = shape
    generator = torch.Generator().manual_seed(608)
    source = torch.rand((2, 3, height, width), generator=generator)
    source[:, 0, 0, 0] = 0
    source[:, 1, -1, -1] = 1
    prediction = 2 * torch.rand(source.shape, generator=generator) - 0.5
    unchanged, zero = limit_local_enhancement(source, prediction, 0, smooth=smooth)
    assert torch.equal(unchanged, source)
    assert torch.count_nonzero(zero) == 0

    for strength in (0.37, 1.0):
        result, alpha = limit_local_enhancement(source, prediction, strength, smooth=smooth)
        assert torch.isfinite(result).all()
        assert (result >= -2e-7).all() and (result <= 1 + 2e-7).all()
        assert alpha.shape == (2, 1, height, width)
        assert (alpha >= 0).all() and (alpha <= strength).all()
        torch.testing.assert_close(result, source + alpha * (prediction - source), atol=2e-7, rtol=0)


def test_full_strength_synthetic_rgb_replay():
    grid = torch.arange(3 * 17 * 19, dtype=torch.float32).reshape(1, 3, 17, 19)
    source = grid.remainder(32) / 31
    prediction = ((grid * 13 + 17).remainder(71) - 15) / 40
    result, _ = limit_local_enhancement(source, prediction, 1.0)
    assert sha256(rgb_image(result).tobytes()).hexdigest() == (
        "5bcedbcbad3c1a656a31c27981d493fc56c339e1ffc4fc8d11bb1e02d1206e65"
    )


# Hashes are of decoded RGB pixels from the day 3 smoothed outputs, which day 4 replayed.
DAY3_RGB_HASHES = {
    "10488": "e47963bce8a9be395a69b7d3c5372992b22ebd209c0c8fc2c15bb4ce6bf93a4a",
    "12709": "95fbcf72216088378ae055fecec9f021e974a319997b5293c43b17f52d02bb3b",
    "12972": "0c410eafebdf27efbfb9e804d02bc24f4fad547d9c7adaf4a5646717ae1f8434",
}


def test_full_strength_replays_day3_development_outputs():
    checkpoint = json.loads((ROOT / "weights/source.json").read_text(encoding="utf-8"))
    development = json.loads((ROOT / "data/dev_manifest.json").read_text(encoding="utf-8"))
    weights = ROOT / checkpoint["file"]
    cases = [item for item in development["files"] if (ROOT / item["path"]).stem in DAY3_RGB_HASHES]
    assert len(cases) == len(DAY3_RGB_HASHES)
    if not torch.cuda.is_available():
        pytest.skip("Exact replay uses the original CUDA inference path.")
    if not weights.is_file() or any(not (ROOT / item["path"]).is_file() for item in cases):
        pytest.skip("Run tools/bootstrap_assets.py to get the weights and development images.")

    assert file_sha256(weights) == checkpoint["sha256"]
    enhancer = UnderwaterEnhancer(weights, "cuda")
    for item in cases:
        source_path = ROOT / item["path"]
        assert file_sha256(source_path) == item["sha256"]
        with Image.open(source_path) as image:
            _, source, prediction, _ = enhancer.predict(image)
        result, _ = limit_local_enhancement(source, prediction, 1.0)
        assert sha256(rgb_image(result).tobytes()).hexdigest() == DAY3_RGB_HASHES[source_path.stem]
