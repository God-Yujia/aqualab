"""Prevent mislabeled comparisons from looking like frozen benchmark runs."""

from copy import deepcopy
import json

import pytest

from tools import evaluate


@pytest.mark.parametrize("change", ["wrong_clip", "duplicate_id", "changed_sweep"])
def test_evaluator_rejects_changed_comparisons(change):
    protocol = deepcopy(json.loads(evaluate.MANIFEST.read_text(encoding="utf-8"))["protocol"])
    if change == "wrong_clip":
        protocol["methods"][1]["method"] = "normalize"
    elif change == "duplicate_id":
        protocol["methods"][2]["id"] = "clip"
    else:
        protocol["interface"]["candidates"] = [0.5, 0.8, 1.0]
    with pytest.raises(ValueError, match="differ from the frozen manifest"):
        evaluate.validate_protocol(protocol)


def test_evaluate_rejects_existing_output_before_loading_model(tmp_path):
    evidence = tmp_path / "summary.json"
    evidence.write_text("historical evidence", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        evaluate.evaluate({}, "unused", "cpu", tmp_path)
    assert evidence.read_text(encoding="utf-8") == "historical evidence"
