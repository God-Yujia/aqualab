r"""AquaLab's shared batch entry point for underwater image enhancement.

Examples (run from this project directory):
  .\.venv\Scripts\python.exe infer.py
  .\.venv\Scripts\python.exe infer.py --input "D:\photos\海底.jpg"
  .\.venv\Scripts\python.exe infer.py --device cpu --max-side 256
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
import torch.nn.functional as F

from local_control import GUIDE_EPSILON, GUIDE_RADIUS, RANGE_KNEE, apply_output_control
from third_party.wwe_uie.model import myModel


ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
METHOD_LABELS = {
    "clip": "WWE-UIE (pretrained)",
    "fixed": "WWE-UIE + fixed strength",
    "normalize": "WWE-UIE + conditional normalization",
    "local": "WWE-UIE + local control",
    "traditional": "Gray World + CLAHE",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_image(image: Image.Image, max_side: int) -> Image.Image:
    """Orient and resize the image identically for every enhancement method."""
    if max_side < 0:
        raise ValueError("max_side must be 0 or a positive integer.")
    working = ImageOps.exif_transpose(image).convert("RGB")
    if max_side and max(working.size) > max_side:
        scale = max_side / max(working.size)
        target = tuple(max(1, round(dimension * scale)) for dimension in working.size)
        working = working.resize(target, Image.Resampling.LANCZOS)
    return working


class UnderwaterEnhancer:
    """Frozen official WWE-UIE with explicit, aspect-preserving preprocessing."""

    def __init__(self, checkpoint: Path, device: str = "auto") -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this Python environment. Install the CUDA requirements or select --device cpu.")
        self.device = torch.device(device)
        if self.device.type == "cpu":
            torch.set_num_threads(min(4, torch.get_num_threads()))
        self.model = myModel(in_channels=3, feature_channels=32, use_white_balance=True)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise ValueError("Expected a model state dictionary in the checkpoint.")
        if state and all(key.startswith("module.") for key in state):
            state = {key.removeprefix("module."): value for key, value in state.items()}
        # Missing or mismatched tensors must fail instead of silently using random parameters.
        self.model.load_state_dict(state, strict=True)
        self.model.requires_grad_(False).eval().to(self.device)
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        self.warmed_up = False

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def predict(
        self, image: Image.Image, max_side: int = 768,
    ) -> tuple[Image.Image, torch.Tensor, torch.Tensor, float]:
        """Return resized input, source tensor, unclipped RGB output, and milliseconds.

        max_side=0 preserves the input resolution. Images keep their aspect ratio;
        padding makes small/odd dimensions valid, and is removed after inference.
        """
        working = prepare_image(image, max_side)
        array = np.array(working, dtype=np.float32) / 255.0
        source = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous().to(self.device)
        height, width = source.shape[-2:]
        padded_height = max(16, math.ceil(height / 8) * 8)
        padded_width = max(16, math.ceil(width / 8) * 8)
        tensor = F.pad(source, (0, padded_width - width, 0, padded_height - height), mode="replicate")
        if not self.warmed_up:
            for _ in range(2):
                self.model(tensor)
            self.synchronize()
            self.warmed_up = True
        self.synchronize()
        start = time.perf_counter()
        prediction = self.model(tensor)
        self.synchronize()
        inference_ms = (time.perf_counter() - start) * 1000
        prediction = prediction[..., :height, :width]
        if not torch.isfinite(prediction).all().item():
            raise RuntimeError("Model output contains NaN or infinity.")
        return working, source, prediction, inference_ms

    def enhance(self, image: Image.Image, max_side: int = 768) -> tuple[Image.Image, Image.Image, float]:
        """Return the original clipped baseline through the existing API."""
        working, _, prediction, inference_ms = self.predict(image, max_side)
        return working, rgb_image(prediction), inference_ms


def rgb_image(prediction: torch.Tensor) -> Image.Image:
    output = prediction[0].permute(1, 2, 0).clamp(0, 1).mul(255).round().byte().cpu().numpy()
    return Image.fromarray(output)


def strength_image(strength_map: torch.Tensor) -> Image.Image:
    pixels = strength_map[0, 0].mul(255).round().byte().cpu().numpy()
    return Image.fromarray(pixels)


def comparison_image(
    before: Image.Image, after: Image.Image, *, panel_width: int = 512,
    after_label: str = METHOD_LABELS["clip"],
) -> Image.Image:
    """An internal development comparison, with matched views and no metrics."""
    scale = min(1.0, panel_width / before.width)
    panel_size = (max(1, round(before.width * scale)), max(1, round(before.height * scale)))
    header = 30
    canvas = Image.new("RGB", (panel_size[0] * 2, panel_size[1] + header), "#15202b")
    canvas.paste(before.resize(panel_size, Image.Resampling.LANCZOS), (0, header))
    canvas.paste(after.resize(panel_size, Image.Resampling.LANCZOS), (panel_size[0], header))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), "Input", fill="white")
    draw.text((panel_size[0] + 10, 8), after_label, fill="white")
    return canvas


def make_contact_sheet(comparisons: list[tuple[str, Path]], destination: Path) -> None:
    tile_width, tile_height = 640, 270
    columns = min(2, len(comparisons))
    rows = math.ceil(len(comparisons) / columns)
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "#e8edf1")
    draw = ImageDraw.Draw(sheet)
    for index, (name, path) in enumerate(comparisons):
        x, y = (index % columns) * tile_width, (index // columns) * tile_height
        draw.text((x + 10, y + 8), name, fill="#15202b")
        with Image.open(path) as picture:
            thumb = ImageOps.contain(picture.convert("RGB"), (tile_width - 20, tile_height - 38))
        sheet.paste(thumb, (x + (tile_width - thumb.width) // 2, y + 28))
    sheet.save(destination, quality=94)


def collect_images(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [path]
    if path.is_dir():
        return sorted((item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS), key=lambda item: item.name.casefold())
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Enhance a photo or folder with AquaLab.")
    parser.add_argument("--input", type=Path, default=ROOT / "data/dev/uieb_c60", help="An image or a folder (non-recursive).")
    parser.add_argument("--output", type=Path, default=None, help="Output folder; defaults to a new timestamped run.")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/wwe_uie_uieb.pth")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-side", type=int, default=768, help="Maximum processing side; 0 keeps the source resolution.")
    parser.add_argument("--method", choices=METHOD_LABELS, default="clip", help="Enhancement method (default: clip).")
    parser.add_argument("--strength", type=float, help="Requested strength in [0, 1] for fixed or local (default 1).")
    parser.add_argument("--no-smoothing", action="store_true", help="Skip closing and guided filtering in local mode for internal comparison.")
    args = parser.parse_args()
    if args.no_smoothing and args.method != "local":
        parser.error("--no-smoothing applies to --method local.")
    if args.strength is not None:
        if args.method not in ("fixed", "local"):
            parser.error("--strength applies to --method fixed or local.")
        if not 0 <= args.strength <= 1:
            parser.error("--strength must be in [0, 1].")
    strength = 1.0 if args.strength is None else args.strength
    images = collect_images(args.input.resolve())
    if not images:
        parser.error(f"No supported images found: {args.input}. Run tools/bootstrap_assets.py to obtain development samples.")
    if args.max_side < 0:
        parser.error("--max-side must be 0 or positive.")
    if len({path.stem.casefold() for path in images}) != len(images):
        parser.error("Input filenames have duplicate stems; use unique names to avoid output collisions.")
    output_dir = args.output or ROOT / "outputs" / datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_%f_UTC")
    output_dir = output_dir.resolve()
    # A run owns its complete directory. Reusing it can silently mix old images,
    # coefficient maps, and summaries with a different method or input batch.
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        parser.error(f"Output must be a new or empty directory: {output_dir}. Choose a different --output folder.")
    enhancer = None
    if args.method == "traditional":
        if args.device == "cuda":
            parser.error("The traditional method runs on CPU. Use --device cpu or auto.")
        try:
            import traditional
        except ModuleNotFoundError as error:
            if error.name != "cv2":
                raise
            parser.error("The traditional method requires OpenCV. Install requirements.txt first.")
    else:
        if not args.weights.is_file():
            parser.error(f"Missing weights: {args.weights}. Run tools/bootstrap_assets.py first.")
        print("Loading the official pretrained weights...", flush=True)
        enhancer = UnderwaterEnhancer(args.weights.resolve(), args.device)
    device = enhancer.device if enhancer is not None else torch.device("cpu")
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "CPU"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    method_label = "WWE-UIE + pixel limit only" if args.no_smoothing else METHOD_LABELS[args.method]
    print(f"Device: {device} / {device_name}; method: {method_label}", flush=True)

    for folder in (output_dir / "enhanced", output_dir / "comparisons"):
        folder.mkdir(parents=True, exist_ok=True)
    if args.method == "local":
        (output_dir / "strength_maps").mkdir(exist_ok=True)
    records = []
    comparisons = []
    for index, path in enumerate(images, start=1):
        start = time.perf_counter()
        with Image.open(path) as image:
            original_size = ImageOps.exif_transpose(image).size
            if enhancer is None:
                before = prepare_image(image, args.max_side)
                result = traditional.gray_world_clahe(before)
                elapsed_ms, strength_map = None, None
            else:
                before, source, prediction, elapsed_ms = enhancer.predict(image, args.max_side)
                adjusted, strength_map = apply_output_control(
                    source, prediction, args.method, strength, smooth=not args.no_smoothing,
                )
                result = rgb_image(adjusted)
        enhanced_path = output_dir / "enhanced" / f"{path.stem}_enhanced.png"
        comparison_path = output_dir / "comparisons" / f"{path.stem}_comparison.jpg"
        result.save(enhanced_path)
        comparison_image(before, result, after_label=method_label).save(comparison_path, quality=94)
        if strength_map is not None:
            strength_path = output_dir / "strength_maps" / f"{path.stem}_strength.png"
            strength_image(strength_map).save(strength_path)
        total_ms = (time.perf_counter() - start) * 1000
        record = {
            "input": str(path.resolve()), "input_sha256": file_sha256(path),
            "original_width": original_size[0], "original_height": original_size[1],
            "output_width": result.width, "output_height": result.height,
            "inference_ms": round(elapsed_ms, 3) if elapsed_ms is not None else None,
            "processing_and_save_ms": round(total_ms, 3),
            "enhanced": str(enhanced_path), "comparison": str(comparison_path),
        }
        if strength_map is not None:
            record["strength_map"] = str(strength_path)
        records.append(record)
        comparisons.append((path.name, comparison_path))
        timing = f"model {elapsed_ms:.1f} ms" if elapsed_ms is not None else f"processed and saved in {total_ms:.1f} ms"
        print(f"[{index}/{len(images)}] {path.name}: {before.width}x{before.height}, {timing} -> {enhanced_path.name}", flush=True)
    make_contact_sheet(comparisons, output_dir / "contact_sheet.jpg")
    with (output_dir / "timings.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    run = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Development smoke run; not a formal quality benchmark.",
        "command": sys.argv,
        "method": args.method,
        "model": "WWE-UIE, official UIEB pretrained checkpoint; no fine-tuning" if enhancer is not None else None,
        "output_control": None,
        "weights_sha256": file_sha256(args.weights) if enhancer is not None else None,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "device": str(device), "device_name": device_name},
        "preprocessing": {"color": "RGB", "range": "[0, 1]" if enhancer is not None else "[0, 255]", "max_side": args.max_side, "aspect_ratio_preserved": True, "output": "8-bit RGB PNG at processing resolution"},
        "parameter_count": enhancer.parameter_count if enhancer is not None else 0,
        "warmup_passes": 2 if enhancer is not None else 0,
        "mean_inference_ms": None,
        "peak_cuda_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 1048576, 2) if device.type == "cuda" else None,
        "count": len(records), "images": records,
    }
    if enhancer is None:
        run["traditional"] = {
            "white_balance": "Gray World",
            "clahe_channel": "Lab L",
            "clahe_clip_limit": traditional.CLAHE_CLIP_LIMIT,
            "clahe_tile_grid": traditional.CLAHE_TILE_GRID,
            "opencv_version": traditional.cv2.__version__,
        }
    else:
        run["preprocessing"]["padding"] = "replicate to a multiple of 8, minimum 16; removed after inference"
        run["preprocessing"]["output"] = "clamp [0, 1], round to 8-bit RGB PNG at processing resolution"
        run["mean_inference_ms"] = round(sum(record["inference_ms"] for record in records) / len(records), 3)
        run["output_control"] = {
            "method": args.method,
            "strength": strength if args.method in ("fixed", "local") else None,
        }
    if args.method == "local":
        run["output_control"].update(range_knee=RANGE_KNEE, smoothing=not args.no_smoothing)
        if not args.no_smoothing:
            run["output_control"].update(closing_kernel_size=3, guide_radius=GUIDE_RADIUS, guide_epsilon=GUIDE_EPSILON)
    (output_dir / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Finished {len(records)} images. Output: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
