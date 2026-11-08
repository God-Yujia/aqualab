"""Evaluate the frozen UIEB manifest and measure candidate UI strengths."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
from statistics import mean
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import skimage
from skimage.color import rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infer import UnderwaterEnhancer, file_sha256, prepare_image, rgb_image
from local_control import GUIDE_EPSILON, GUIDE_RADIUS, RANGE_KNEE, apply_output_control
from tools.fetch_test_assets import verify_manifest
from traditional import CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID, gray_world_clahe


MANIFEST = ROOT / "data/test_manifest.json"
SWEEP_STRENGTHS = (0.6, 0.8, 1.0)
METRIC_FIELDS = (
    "psnr_rgb", "ssim_rgb", "uciqe_lab", "rgb_endpoint_255_pct",
    "rgb_endpoint_0_pct", "model_forward_ms", "processing_ms",
)


def validate_protocol(protocol: dict) -> Path:
    """Check that runtime settings agree with the frozen algorithm."""
    expected_local = {
        "range_knee": RANGE_KNEE, "closing_kernel_size": 3,
        "guide_radius": GUIDE_RADIUS, "guide_epsilon": GUIDE_EPSILON,
    }
    expected_traditional = {
        "white_balance": "Gray World", "clahe_channel": "Lab L",
        "clahe_clip_limit": CLAHE_CLIP_LIMIT, "clahe_tile_grid": list(CLAHE_TILE_GRID),
    }
    if protocol["local_control"] != expected_local or protocol["traditional"] != expected_traditional:
        raise ValueError("Runtime enhancement parameters differ from the frozen manifest.")
    preprocessing = protocol["preprocessing"]
    if any(preprocessing[key] != value for key, value in {
        "apply_exif_orientation": True, "color_mode": "RGB", "resampling": "Lanczos",
        "preserve_aspect_ratio": True, "metric_image": "exported 8-bit RGB at processing dimensions",
    }.items()):
        raise ValueError("The evaluator requires the frozen RGB8 preprocessing protocol.")
    local = next(method for method in protocol["methods"] if method["id"] == "local")
    if local["method"] != "local" or local["strength"] != 1.0 or not local["smoothing"]:
        raise ValueError("The frozen local baseline must be the full module at strength 1.0.")
    uciqe = protocol["metrics"]["uciqe_lab"]
    if any(uciqe[key] != value for key, value in {
        "chroma": "hypot(a,b)",
        "lightness_contrast": "mean of largest k minus mean of smallest k, k=max(1,floor(N*0.01))",
        "saturation": "C/L for L>0, otherwise 0",
    }.items()):
        raise ValueError("The evaluator requires the frozen Lab UCIQE definitions.")
    timing = protocol["timing"]
    if timing["batch_size"] != 1 or not timing["synchronize_cuda_for_timing"]:
        raise ValueError("The evaluator requires batch size 1 and synchronized CUDA timing.")
    if skimage.__version__ != "0.24.0":
        raise RuntimeError("Install the pinned scikit-image 0.24.0 from requirements.txt.")
    checkpoint = (ROOT / protocol["checkpoint"]["path"]).resolve()
    if not checkpoint.is_relative_to(ROOT) or file_sha256(checkpoint) != protocol["checkpoint"]["sha256"]:
        raise ValueError("The checkpoint does not match the frozen manifest.")
    return checkpoint


def paired_metrics(reference: np.ndarray, result: np.ndarray, config: dict) -> dict:
    data_range = config["psnr_rgb"]["data_range"]
    psnr = math.inf if np.array_equal(reference, result) else float(
        peak_signal_noise_ratio(reference, result, data_range=data_range)
    )
    return {
        "psnr_rgb": psnr,
        "ssim_rgb": float(structural_similarity(reference, result, **config["ssim_rgb"])),
    }


def uciqe_lab(rgb: np.ndarray, config: dict) -> float:
    lab = rgb2lab(rgb, illuminant=config["illuminant"], observer=config["observer"]) / config["lab_scale"]
    lightness, a, b = lab.reshape(-1, 3).T
    chroma = np.hypot(a, b)
    saturation = np.divide(chroma, lightness, out=np.zeros_like(chroma), where=lightness > 0)
    k = max(1, lightness.size // 100)
    tails = np.partition(lightness, (k - 1, lightness.size - k))
    contrast = tails[-k:].mean() - tails[:k].mean()
    weights = config["coefficients"]
    return float(weights[0] * chroma.std(ddof=config["chroma_std_ddof"])
                 + weights[1] * contrast + weights[2] * saturation.mean())


def open_prepared(path: Path, max_side: int) -> Image.Image:
    with Image.open(path) as image:
        return prepare_image(image, max_side)


def warm_up(enhancer: UnderwaterEnhancer, size: list[int], passes: int) -> None:
    width, height = (max(16, math.ceil(side / 8) * 8) for side in size)
    tensor = torch.zeros((1, 3, height, width), device=enhancer.device)
    with torch.inference_mode():
        for _ in range(passes):
            enhancer.model(tensor)
    enhancer.synchronize()
    enhancer.warmed_up = True


def save_case(path: Path, title: str, panels: dict[str, Image.Image]) -> None:
    pictures = {label: image.copy() for label, image in panels.items()}
    for image in pictures.values():
        image.thumbnail((440, 340), Image.Resampling.LANCZOS)
    width, height = next(iter(pictures.values())).size
    canvas = Image.new("RGB", (len(pictures) * width, height + 58), "#15202b")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 7), title, fill="white")
    for index, (label, image) in enumerate(pictures.items()):
        offset = index * width
        draw.text((offset + 10, 34), label, fill="white")
        canvas.paste(image, (offset, 58))
    canvas.save(path)


def summarize(rows: list[dict], fields: tuple[str, ...] = METRIC_FIELDS) -> dict:
    averages = {"count": len(rows)}
    for field in fields:
        values = [row[field] for row in rows if row[field] is not None]
        if values:
            average = mean(values)
            averages[f"mean_{field}"] = "inf" if average == math.inf else average
    return averages


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_strength_curve(path: Path, scores: dict, selected: float) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    strengths = sorted(map(float, scores))
    x = [100 * strength for strength in strengths]
    count = scores[str(strengths[0])]["count"]
    figure = Figure(figsize=(9, 4), dpi=180, facecolor="#ffffff")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    figure.subplots_adjust(left=0.09, right=0.97, top=0.72, bottom=0.23, wspace=0.36)
    figure.text(0.09, 0.94, "Choosing the default enhancement strength", fontsize=17, weight="bold", color="#142b3b")
    figure.text(0.09, 0.87, f"UIEB paired subset  |  {count} image pairs  |  Selected default {selected:.0%}",
                fontsize=10, color="#425b6c")

    for axis, (metric, label, digits) in zip(axes, (
        ("mean_psnr_rgb", "Mean PSNR (dB)", 2), ("mean_ssim_rgb", "Mean SSIM", 4),
    )):
        values = [float(scores[str(strength)][metric]) for strength in strengths]
        finite = [value for value in values if math.isfinite(value)]
        axis.plot(x, [value if math.isfinite(value) else math.nan for value in values], "o-",
                  color="#197d82", linewidth=2, markersize=5)
        if finite:
            span = max(max(finite) - min(finite), 10**-digits)
            axis.set_ylim(min(finite) - 0.18 * span, max(finite) + 0.24 * span)
        for strength, position, value in zip(strengths, x, values):
            if math.isfinite(value):
                if strength == selected:
                    axis.scatter(position, value, color="#ba6428", s=58, zorder=4)
                axis.annotate(f"{value:.{digits}f}", (position, value), xytext=(0, 8),
                              textcoords="offset points", ha="center", fontsize=9, color="#142b3b")
            else:
                axis.annotate("inf", (position, 0.94), xycoords=("data", "axes fraction"), ha="center")
        axis.set(xlim=(53, 107), xticks=x, xticklabels=[f"{value:.0f}%" for value in x], ylabel=label)
        axis.set_xlabel("Requested local strength")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.23)
    figure.text(0.09, 0.055, "Selection uses mean SSIM, with mean PSNR as the tie-breaker.", fontsize=9, color="#425b6c")
    figure.savefig(path)


def evaluate(manifest: dict, manifest_hash: str, device: str, output: Path) -> None:
    protocol = manifest["protocol"]
    checkpoint = validate_protocol(protocol)
    verified_files = verify_manifest(manifest)
    print(f"Verified {verified_files} files against the frozen manifest.", flush=True)
    enhancer = UnderwaterEnhancer(checkpoint, device)
    subsets = manifest["subsets"]
    first = next(iter(subsets.values()))["files"][0]
    warm_up(enhancer, first["processing_size"], protocol["timing"]["model_warmup_passes"])
    max_side = protocol["preprocessing"]["max_side"]
    config = protocol["metrics"]
    methods = protocol["methods"]
    rows = []
    sweep_rows = []
    cases = {}

    for subset, dataset in subsets.items():
        for index, sample in enumerate(dataset["files"], start=1):
            image_path = ROOT / sample["input"]["path"]
            reference = open_prepared(ROOT / sample["reference"]["path"], max_side) if dataset["reference_images"] else None
            reference_rgb = np.asarray(reference) if reference is not None else None

            enhancer.synchronize()
            start = time.perf_counter()
            with Image.open(image_path) as image:
                working, source, prediction, forward_ms = enhancer.predict(image, max_side)
            shared_ms = (time.perf_counter() - start) * 1000

            sample_rows = {}
            case_images = {"Input": working}
            for method in methods:
                method_id = method["id"]
                conventional = method["method"] == "traditional"
                start = time.perf_counter()
                if conventional:
                    result = gray_world_clahe(open_prepared(image_path, max_side))
                else:
                    adjusted, _ = apply_output_control(
                        source, prediction, method["method"], method.get("strength", 1.0),
                        smooth=method.get("smoothing", True),
                    )
                    result = rgb_image(adjusted)
                    enhancer.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                rgb = np.asarray(result)
                row = {
                    "subset": subset, "id": sample["id"], "tags": "|".join(sample["tags"]),
                    "method": method_id, "width": working.width, "height": working.height,
                    "psnr_rgb": None, "ssim_rgb": None, "uciqe_lab": None,
                    "rgb_endpoint_255_pct": float(100 * np.any(rgb == 255, axis=-1).mean()),
                    "rgb_endpoint_0_pct": float(100 * np.any(rgb == 0, axis=-1).mean()),
                    "model_forward_ms": None if conventional else forward_ms,
                    "processing_ms": elapsed if conventional else shared_ms + elapsed,
                }
                if reference_rgb is None:
                    row["uciqe_lab"] = uciqe_lab(rgb, config["uciqe_lab"])
                else:
                    row.update(paired_metrics(reference_rgb, rgb, config))
                rows.append(row)
                sample_rows[method_id] = row
                if method_id in ("clip", "local"):
                    case_images[{"clip": "Hard clip", "local": "Local 100%"}[method_id]] = result

            baseline, local = sample_rows["clip"], sample_rows["local"]
            if reference_rgb is not None:
                for strength in SWEEP_STRENGTHS:
                    if strength == 1.0:
                        scores = {key: local[key] for key in ("psnr_rgb", "ssim_rgb")}
                    else:
                        adjusted, _ = apply_output_control(source, prediction, "local", strength)
                        scores = paired_metrics(reference_rgb, np.asarray(rgb_image(adjusted)), config)
                    sweep_rows.append({"id": sample["id"], "strength": strength, **scores})
                case_images["Reference"] = reference
                delta = local["ssim_rgb"] - baseline["ssim_rgb"]
                if "paired_lowest_ssim_delta" not in cases or delta < cases["paired_lowest_ssim_delta"][0]:
                    cases["paired_lowest_ssim_delta"] = (delta, sample["id"], case_images)
                if "paired_highest_ssim_delta" not in cases or delta > cases["paired_highest_ssim_delta"][0]:
                    cases["paired_highest_ssim_delta"] = (delta, sample["id"], case_images)
            else:
                reduction = baseline["rgb_endpoint_255_pct"] - local["rgb_endpoint_255_pct"]
                if "c60_largest_high_endpoint_reduction" not in cases or reduction > cases["c60_largest_high_endpoint_reduction"][0]:
                    cases["c60_largest_high_endpoint_reduction"] = (reduction, sample["id"], case_images)
            print(f"{subset} [{index}/{dataset['count']}] {sample['id']}", flush=True)

    summaries = {}
    for subset in subsets:
        subset_rows = [row for row in rows if row["subset"] == subset]
        tags = sorted({tag for row in subset_rows for tag in row["tags"].split("|")})
        summaries[subset] = {"methods": {}, "by_tag": {tag: {} for tag in tags}}
        for method in methods:
            method_id = method["id"]
            method_rows = [row for row in subset_rows if row["method"] == method_id]
            summaries[subset]["methods"][method_id] = summarize(method_rows)
            for tag in tags:
                summaries[subset]["by_tag"][tag][method_id] = summarize([
                    row for row in method_rows if tag in row["tags"].split("|")
                ])
    scores = {
        str(strength): summarize([row for row in sweep_rows if row["strength"] == strength], ("psnr_rgb", "ssim_rgb"))
        for strength in SWEEP_STRENGTHS
    }
    chosen = max(scores, key=lambda key: (scores[key]["mean_ssim_rgb"], float(scores[key]["mean_psnr_rgb"])))
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "per_image.csv", rows)
    write_csv(output / "strength_sweep.csv", sweep_rows)
    save_strength_curve(output / "strength_curve.png", scores, float(chosen))
    case_records = {}
    for label, (value, sample_id, panels) in cases.items():
        filename = f"{label}.png"
        save_case(output / filename, f"{sample_id}  |  {label.replace('_', ' ')}  {value:+.4f}", panels)
        case_records[label] = {"id": sample_id, "value": value, "image": filename}
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": manifest_hash,
        "verified_files": verified_files,
        "protocol": protocol,
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "scikit_image": skimage.__version__, "numpy": np.__version__,
            "device": str(enhancer.device),
            "device_name": torch.cuda.get_device_name(enhancer.device) if enhancer.device.type == "cuda" else platform.processor(),
        },
        "subsets": summaries,
        "strength_sweep": {
            "selection_rule": "Highest mean SSIM, then mean PSNR, then lower strength.",
            "purpose": "Choose the interface default on the paired subset. Its winning score is not independent validation.",
            "selected_strength": float(chosen),
            "scores": scores,
            "plot": "strength_curve.png",
        },
        "comparison_cases": case_records,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Finished {len(rows)} method rows and {len(sweep_rows)} strength rows. Selected {chosen}. Output: {output}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output", type=Path, help="New output directory. Defaults to a timestamped Day 5 run.")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_UTC")
    output = args.output or ROOT / "outputs/day5" / stamp
    if output.exists():
        parser.error(f"Output directory already exists: {output}")
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        evaluate(manifest, file_sha256(MANIFEST), args.device, output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
