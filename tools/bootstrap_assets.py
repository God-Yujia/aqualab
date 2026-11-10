"""Download the official model and a reproducible, local-only UIEB dev sample.

Uses the Python standard library and Windows' bundled bsdtar. Dataset images
and model weights are excluded from Git; their provenance is recorded in JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import random
import shutil
import subprocess
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
REVISION = "fd0697558203f9f849800cc13b804116bb3f921c"
REPOSITORY = "https://github.com/chingheng0808/WWE-UIE"
RAW = f"https://raw.githubusercontent.com/chingheng0808/WWE-UIE/{REVISION}"
MODEL_ID = "1c80kkDbgpLD1MtqVLql1Y1NQVD7IzxiO"
C60_ID = "1Ew_r83nXzVk0hlkfuomWqsAIxuq6kaN4"
UIEB_PAGE = "https://li-chongyi.github.io/proj_benchmark.html"
C60_PASSWORD = "5678901"  # Publicly provided by the official UIEB download page.


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, path: Path, *, binary: bool = False) -> None:
    if path.is_file() and path.stat().st_size > 100:
        print(f"Already downloaded: {path.relative_to(ROOT)}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "AquaLab/0.1"})
            with urllib.request.urlopen(request, timeout=45) as response:
                expected = int(response.headers.get("Content-Length", "0"))
                print(f"Downloading {path.name} ({expected / 1048576:.1f} MiB)", flush=True)
                first = response.read(1024)
                if binary and (b"<html" in first.lower() or b"<!doctype" in first.lower()):
                    raise RuntimeError("The download returned an HTML page instead of a file.")
                written = len(first)
                last_report = time.monotonic()
                with partial.open("wb") as stream:
                    stream.write(first)
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                        written += len(chunk)
                        if time.monotonic() - last_report > 10:
                            print(f"  {path.name}: {written / 1048576:.1f} MiB", flush=True)
                            last_report = time.monotonic()
                if expected and written != expected:
                    raise RuntimeError(f"Incomplete download: {written} / {expected} bytes")
                if written < 100:
                    raise RuntimeError("The downloaded file is unexpectedly small.")
            partial.replace(path)
            return
        except Exception as error:
            if attempt == 2:
                raise
            print(f"Retry {attempt + 1}: {error}", flush=True)
            time.sleep(2)


def drive_url(file_id: str) -> str:
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def verify_frozen_file(path: Path, expected_hash: str) -> None:
    """Never replace the trusted digest with a digest of a changed local file."""
    if sha256(path) != expected_hash:
        raise ValueError(
            f"SHA-256 mismatch: {path}. Restore the frozen file or remove the "
            "damaged download and rerun; the recorded digest was not changed."
        )


def prepare_model() -> None:
    model_source = json.loads((ROOT / "third_party/wwe_uie/source.json").read_text(encoding="utf-8"))
    checkpoint_source = json.loads((ROOT / "weights/source.json").read_text(encoding="utf-8"))
    if any(source["revision"] != REVISION or source["repository"] != REPOSITORY
           for source in (model_source, checkpoint_source)):
        raise ValueError("Asset provenance differs from the frozen WWE-UIE revision.")
    paths = {
        "model.py": ROOT / "third_party/wwe_uie/model.py",
        "LICENSE": ROOT / "third_party/wwe_uie/LICENSE",
    }
    for name, path in paths.items():
        download(f"{RAW}/{name}", path)
        verify_frozen_file(path, model_source["files"][name])
    checkpoint = ROOT / "weights/wwe_uie_uieb.pth"
    download(drive_url(MODEL_ID), checkpoint, binary=True)
    verify_frozen_file(checkpoint, checkpoint_source["sha256"])
    with checkpoint.open("rb") as stream:
        if stream.read(4) != b"PK\x03\x04":
            raise RuntimeError("Unexpected checkpoint format; expected a PyTorch ZIP checkpoint.")


def prepare_samples(count: int, seed: int) -> None:
    manifest = json.loads((ROOT / "data/dev_manifest.json").read_text(encoding="utf-8"))
    if manifest["selection"]["count"] != count or manifest["selection"]["seed"] != seed:
        raise ValueError("The requested sample selection differs from the frozen development manifest.")
    expected_files = {row["path"]: row["sha256"] for row in manifest["files"]}
    if len(expected_files) != count or len(manifest["files"]) != count:
        raise ValueError("The development manifest contains missing or duplicate files.")
    sample_dir = ROOT / "data/dev/uieb_c60"
    if sample_dir.is_dir():
        expected_names = {PurePosixPath(path).name for path in expected_files}
        existing_names = {path.name for path in sample_dir.iterdir() if path.is_file()}
        if existing_names - expected_names:
            raise ValueError("The sample folder contains files outside the frozen development manifest.")
    for relative, expected_hash in expected_files.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to((ROOT / "data/dev/uieb_c60").resolve()):
            raise ValueError(f"Unexpected development path: {relative}")
        if path.exists():
            verify_frozen_file(path, expected_hash)
    if all((ROOT / relative).is_file() for relative in expected_files):
        print(f"Verified {len(expected_files)} frozen development images.", flush=True)
        return
    tar = shutil.which("tar")
    if tar is None:
        raise RuntimeError("A bsdtar executable is needed to read the official UIEB RAR archive.")
    archive = ROOT / ".cache/downloads/uieb_challenging_60.rar"
    download(drive_url(C60_ID), archive, binary=True)
    verify_frozen_file(archive, manifest["archive_sha256"])
    listing = subprocess.run([tar, "-tf", str(archive)], check=True, capture_output=True)
    members = []
    for name in listing.stdout.decode("utf-8").splitlines():
        member = PurePosixPath(name.replace("\\", "/"))
        if member.is_absolute() or ".." in member.parts:
            raise RuntimeError(f"Unexpected archive member: {name}")
        if member.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"} and "__MACOSX" not in name:
            members.append(name)
    members.sort()
    if not 1 <= count <= len(members):
        raise ValueError(f"Sample count must be between 1 and {len(members)}.")
    selected = sorted(random.Random(seed).sample(members, count))
    selected_names = {PurePosixPath(name.replace("\\", "/")).name for name in selected}
    selected_paths = {f"data/dev/uieb_c60/{name}" for name in selected_names}
    if selected_paths != set(expected_files):
        raise ValueError("Archive selection differs from the frozen development files.")
    sample_dir.mkdir(parents=True, exist_ok=True)
    unrar = prepare_unrar(tar)
    for name in selected:
        basename = PurePosixPath(name.replace("\\", "/")).name
        path = sample_dir / basename
        if not path.exists():
            # Stream only the selected member to stdout; never extract arbitrary archive paths.
            member_name = name.replace("/", "\\") if os.name == "nt" else name
            result = subprocess.run([str(unrar), "p", "-inul", f"-p{C60_PASSWORD}", str(archive), member_name], capture_output=True)
            if result.returncode:
                raise RuntimeError(f"UnRAR failed for {name} (exit {result.returncode}): {result.stderr.decode('utf-8', errors='replace')}")
            content = result.stdout
            if not content:
                raise RuntimeError(f"Empty archive member: {name}")
            if hashlib.sha256(content).hexdigest() != expected_files[path.relative_to(ROOT).as_posix()]:
                raise ValueError(f"Archive member SHA-256 mismatch: {name}")
            path.write_bytes(content)
        verify_frozen_file(path, expected_files[path.relative_to(ROOT).as_posix()])
        print(f"Dev sample: {path.name}", flush=True)


def prepare_unrar(tar: str) -> Path:
    """Unpack the official portable UnRAR, without executing its SFX installer.

    The UIEB archive is encrypted; Windows bsdtar can list it but cannot decrypt
    its members. The public password is listed on the official dataset page.
    """
    destination = ROOT / ".cache/tools/UnRAR.exe"
    if destination.is_file():
        return destination
    package = ROOT / ".cache/tools/unrarw64.exe"
    download("https://www.rarlab.com/rar/unrarw64.exe", package, binary=True)
    blob = package.read_bytes()
    offsets = [blob.find(magic) for magic in (b"Rar!\x1a\x07\x00", b"Rar!\x1a\x07\x01\x00")]
    offsets = [offset for offset in offsets if offset >= 0]
    if not offsets:
        raise RuntimeError("Could not find the archive in the official UnRAR distribution.")
    archive = package.with_suffix(".rar")
    archive.write_bytes(blob[min(offsets):])
    for member, path in (("UnRAR.exe", destination), ("license.txt", destination.with_name("UnRAR-license.txt"))):
        result = subprocess.run([tar, "-xOf", str(archive), member], check=True, capture_output=True)
        path.write_bytes(result.stdout)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10, help="Frozen development sample count (10).")
    parser.add_argument("--seed", type=int, default=42, help="Frozen development selection seed (42).")
    parser.add_argument("--model-only", action="store_true")
    args = parser.parse_args()
    try:
        prepare_model()
        if not args.model_only:
            prepare_samples(args.count, args.seed)
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Asset preparation failed: {error}\n")
    print("Official assets are ready.", flush=True)


if __name__ == "__main__":
    main()
