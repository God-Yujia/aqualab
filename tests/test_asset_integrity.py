"""Asset bootstrap must preserve the checked-in evidence of provenance."""

from hashlib import sha256
import json
from pathlib import Path

import pytest

from tools import bootstrap_assets as bootstrap


def write_json(path: Path, value: dict) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(value) + "\n").encode()
    path.write_bytes(content)
    return content


@pytest.mark.parametrize("changed", ["model.py", "checkpoint"])
def test_model_bootstrap_rejects_changed_asset_without_rewriting_provenance(tmp_path, monkeypatch, changed):
    monkeypatch.setattr(bootstrap, "ROOT", tmp_path)
    model_dir = tmp_path / "third_party/wwe_uie"
    model_dir.mkdir(parents=True)
    files = {"model.py": b"# official model\n" * 20, "LICENSE": b"official license\n" * 20}
    for name, content in files.items():
        (model_dir / name).write_bytes(content)
    checkpoint = tmp_path / "weights/wwe_uie_uieb.pth"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"PK\x03\x04" + b"x" * 200)
    common = {"revision": bootstrap.REVISION, "repository": bootstrap.REPOSITORY}
    model_metadata = model_dir / "source.json"
    checkpoint_metadata = checkpoint.parent / "source.json"
    model_before = write_json(model_metadata, {**common, "files": {
        name: sha256(content).hexdigest() for name, content in files.items()
    }})
    checkpoint_before = write_json(checkpoint_metadata, {**common, "sha256": bootstrap.sha256(checkpoint)})
    target = checkpoint if changed == "checkpoint" else model_dir / changed
    target.write_bytes(target.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        bootstrap.prepare_model()

    assert model_metadata.read_bytes() == model_before
    assert checkpoint_metadata.read_bytes() == checkpoint_before


@pytest.mark.parametrize("changed", [False, True])
def test_existing_development_images_are_verified_offline(tmp_path, monkeypatch, changed):
    monkeypatch.setattr(bootstrap, "ROOT", tmp_path)
    image = tmp_path / "data/dev/uieb_c60/example.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"frozen test image")
    manifest_path = tmp_path / "data/dev_manifest.json"
    original_manifest = write_json(manifest_path, {
        "selection": {"count": 1, "seed": 42},
        "files": [{"path": image.relative_to(tmp_path).as_posix(), "sha256": bootstrap.sha256(image)}],
    })
    monkeypatch.setattr(bootstrap, "download", lambda *args, **kwargs: pytest.fail("Existing images need no download"))
    if changed:
        image.write_bytes(b"changed image")
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            bootstrap.prepare_samples(1, 42)
    else:
        bootstrap.prepare_samples(1, 42)
    assert manifest_path.read_bytes() == original_manifest


def test_bootstrap_does_not_silently_replace_frozen_development_split(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "ROOT", tmp_path)
    write_json(tmp_path / "data/dev_manifest.json", {"selection": {"count": 10, "seed": 42}})
    with pytest.raises(ValueError, match="frozen development manifest"):
        bootstrap.prepare_samples(10, 43)


def test_offline_development_verification_rejects_extra_images(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "ROOT", tmp_path)
    image = tmp_path / "data/dev/uieb_c60/example.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"frozen test image")
    (image.parent / "unexpected.png").write_bytes(b"extra image")
    write_json(tmp_path / "data/dev_manifest.json", {
        "selection": {"count": 1, "seed": 42},
        "files": [{"path": image.relative_to(tmp_path).as_posix(), "sha256": bootstrap.sha256(image)}],
    })
    with pytest.raises(ValueError, match="outside the frozen development manifest"):
        bootstrap.prepare_samples(1, 42)
