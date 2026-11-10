"""Batch output isolation and desktop export/failure regressions."""

import json
from pathlib import Path
import sys
import time
import tkinter as tk

import numpy as np
from PIL import Image
import pytest
import torch

import app
import infer


def test_batch_run_refuses_to_mix_with_previous_outputs(tmp_path, monkeypatch):
    source = tmp_path / "photo.png"
    Image.new("RGB", (17, 19), (20, 70, 110)).save(source)
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", [
        "infer.py", "--input", str(source), "--output", str(output),
        "--method", "traditional", "--device", "cpu",
    ])
    assert infer.main() == 0
    summary = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert summary["count"] == 1
    assert summary["method"] == "traditional"
    with Image.open(summary["images"][0]["enhanced"]) as enhanced:
        assert enhanced.size == (17, 19)
        assert enhanced.mode == "RGB"

    # Changing input or method must not silently rewrite one part of a run.
    old_files = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    Image.new("RGB", (9, 7), (70, 20, 10)).save(source)
    with pytest.raises(SystemExit) as error:
        infer.main()
    assert error.value.code == 2
    assert old_files == {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}


@pytest.fixture(scope="module")
def window():
    try:
        root = tk.Tk()
    except tk.TclError as error:
        if "no display name" in str(error) or "couldn't connect to display" in str(error):
            pytest.skip(f"A Tk display is unavailable: {error}")
        raise
    root.withdraw()
    instance = app.AquaLabWindow(root, Path("unused-checkpoint.pth"), "cpu")
    yield instance
    if not instance.closing:
        instance.close()


def test_export_flushes_the_pending_slider_value(window, tmp_path, monkeypatch):
    pixels = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    original = Image.fromarray(pixels)
    source = torch.from_numpy(pixels.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255
    window.path = tmp_path / "input.png"
    original.save(window.path)
    window.tensors = source, source + 0.2
    window.result, window.strength_map = app.render_local(*window.tensors, 1)
    window._enable_result(True)
    window.strength.set(0)
    window.strength_changed("0")
    assert window.refresh_job is not None

    target = tmp_path / "export.png"
    monkeypatch.setattr(app.filedialog, "asksaveasfilename", lambda **kwargs: str(target))
    window.save_result()
    with Image.open(target) as exported:
        assert exported.tobytes() == original.tobytes()
    assert window.refresh_job is None
    assert not np.asarray(window.strength_map).any()
    assert not list(tmp_path.glob(".aqualab-*"))


def test_failed_image_load_clears_previous_preview(window, tmp_path, monkeypatch):
    window.path = tmp_path / "previous.png"
    window.result = Image.new("RGB", (5, 7))
    window.strength_map = Image.new("L", (5, 7))
    window.tensors = (torch.zeros(1, 3, 7, 5),) * 2
    window.original.show(window.result)
    window.enhanced.show(window.result)
    window.map_pane.show(window.strength_map)
    window._enable_result(True)
    errors = []
    monkeypatch.setattr(app.messagebox, "showerror", lambda *args, **kwargs: errors.append(args))
    damaged = tmp_path / "damaged.png"
    damaged.write_bytes(b"not an image")
    window.load_image(damaged)
    deadline = time.monotonic() + 5
    while window.pending is not None and time.monotonic() < deadline:
        window.root.update()
        time.sleep(0.01)

    assert window.pending is None
    assert errors
    assert window.path is window.result is window.strength_map is window.tensors is None
    assert window.original.image is window.enhanced.image is window.map_pane.image is None
    assert window.export_button.instate(["disabled"])
    assert window.open_button.instate(["!disabled"])
