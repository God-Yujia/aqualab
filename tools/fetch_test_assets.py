"""Fetch missing formal-test images and verify the frozen test manifest."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
from tempfile import NamedTemporaryFile
import time
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

import numpy as np
from PIL import Image, ImageOps

if __package__:
    from .bootstrap_assets import C60_ID, C60_PASSWORD, ROOT, download, drive_url, prepare_unrar, sha256
else:
    from bootstrap_assets import C60_ID, C60_PASSWORD, ROOT, download, drive_url, prepare_unrar, sha256


MANIFEST = ROOT / "data/test_manifest.json"
PAIRED_ID = "1ceeWdqRdiIortVTSH0UNTZamHdMsZ3aA"
CHUNK_SIZE = 8 * 1024**2


@dataclass(frozen=True)
class Asset:
    path: Path
    member: str
    sha256: str
    size: tuple[int, int]
    processing_size: tuple[int, int]
    tags: tuple[str, ...] | None


def read_manifest(path: Path = MANIFEST) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_assets(manifest: dict, root: Path) -> Iterable[tuple[str, Asset]]:
    if manifest["version"] != 1:
        raise ValueError("Unsupported test manifest version")
    data_dir = root / "data/test"
    if not data_dir.resolve().is_relative_to(root.resolve()):
        raise ValueError("Test image directory resolves outside the project")
    seen_paths = set()
    seen_ids = set()
    for subset, roles in (
        ("uieb_paired", (("input", "input", "UIEB/test/input"), ("reference", "reference", "UIEB/test/GT"))),
        ("uieb_c60", (("input", "", "challenging-60"),)),
    ):
        section = manifest["subsets"][subset]
        if len(section["files"]) != section["count"] or section["reference_images"] != (len(roles) == 2):
            raise ValueError(f"Invalid manifest subset: {subset}")
        for row in section["files"]:
            filename = PurePosixPath(row["input"]["path"]).name
            if not re.fullmatch(r"[a-zA-Z0-9_][a-zA-Z0-9_.-]*\.(png|jpe?g|bmp)", filename, re.IGNORECASE):
                raise ValueError(f"Invalid test image name: {filename}")
            if row["id"] in seen_ids:
                raise ValueError(f"Duplicate test ID: {row['id']}")
            seen_ids.add(row["id"])
            for role, folder, prefix in roles:
                entry = row[role]
                relative = PurePosixPath("data/test", subset, folder, filename)
                member = f"{prefix}/{filename}"
                if entry["path"] != relative.as_posix() or entry["archive_member"] != member:
                    raise ValueError(f"Unexpected {role} path for {row['id']}")
                path = (root / relative).resolve()
                if not path.is_relative_to(data_dir.resolve()) or path in seen_paths:
                    raise ValueError(f"Unsafe or duplicate test path: {path}")
                seen_paths.add(path)
                yield subset, Asset(path, member, entry["sha256"], tuple(row["input_size"]),
                                    tuple(row["processing_size"]), tuple(row["tags"]) if role == "input" else None)


def scene_tags(image: Image.Image, weights: np.ndarray) -> set[str]:
    pixels = np.asarray(image, dtype=np.float32)
    red, green, blue = pixels.mean(axis=(0, 1), dtype=np.float64)
    luma = pixels @ weights
    tags = set()
    if blue - red >= 15 and blue - green >= 5:
        tags.add("blue_cast")
    if green - red >= 15 and green - blue >= 5:
        tags.add("green_cast")
    if np.median(luma) < 64:
        tags.add("low_light")
    if (luma >= 235).mean() >= 0.01:
        tags.add("highlights")
    return tags or {"other"}


def verify_asset(asset: Asset, max_side: int, weights: np.ndarray) -> None:
    if sha256(asset.path) != asset.sha256:
        raise ValueError(f"SHA-256 mismatch: {asset.path}")
    with Image.open(asset.path) as image:
        working = ImageOps.exif_transpose(image).convert("RGB")
    if working.size != asset.size:
        raise ValueError(f"Oriented image dimensions changed: {asset.path}")
    if max_side and max(working.size) > max_side:
        scale = max_side / max(working.size)
        size = tuple(max(1, round(dimension * scale)) for dimension in working.size)
        working = working.resize(size, Image.Resampling.LANCZOS)
    if working.size != asset.processing_size:
        raise ValueError(f"Processing dimensions changed: {asset.path}")
    if asset.tags is not None and scene_tags(working, weights) != set(asset.tags):
        raise ValueError(f"Scene tags changed: {asset.path}")


def verification_settings(manifest: dict) -> tuple[int, np.ndarray]:
    config = manifest["protocol"]["preprocessing"]
    max_side = config["max_side"]
    if (not isinstance(max_side, int) or max_side < 0 or config["resampling"] != "Lanczos"
        or config["color_mode"] != "RGB" or not config["apply_exif_orientation"] or not config["preserve_aspect_ratio"]):
        raise ValueError("Unsupported preprocessing in the test manifest")
    weights = np.asarray(manifest["tagging"]["luma_proxy_weights"], dtype=np.float32)
    if weights.shape != (3,) or not np.isfinite(weights).all():
        raise ValueError("Invalid luminance weights in the test manifest")
    return max_side, weights


def verify_manifest(manifest: dict, root: Path = ROOT) -> int:
    """Verify the frozen image files offline and return the number checked."""
    assets = list(iter_assets(manifest, root))
    settings = verification_settings(manifest)
    for _, asset in assets:
        verify_asset(asset, *settings)
    return len(assets)


def fetch_range(url: str, start: int, stop: int, archive_size: int | None = None) -> tuple[bytes, int]:
    end = stop - 1
    request = Request(f"{url}&aqualab_range={start}-{end}", headers={
        "Range": f"bytes={start}-{end}", "User-Agent": "AquaLab/0.1",
    })
    for attempt in range(3):
        try:
            with urlopen(request, timeout=45) as response:
                content_range = response.headers.get("Content-Range", "")
                match = re.fullmatch(rf"bytes {start}-{end}/(\d+)", content_range)
                if response.status != 206 or not match:
                    raise ValueError("The upstream ZIP server did not honor the byte range. Supply --paired-zip.")
                size = int(match.group(1))
                if archive_size is not None and size != archive_size:
                    raise ValueError("The upstream ZIP archive size changed. Supply the frozen --paired-zip.")
                data = response.read(stop - start + 1)
            if len(data) != stop - start:
                raise OSError("Incomplete ZIP range download")
            return data, size
        except OSError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def install_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class RangedZip(io.RawIOBase):
    """Seek over the upstream ZIP, caching only accessed byte blocks."""

    def __init__(self, cache: Path) -> None:
        self.cache = cache
        self.url = drive_url(PAIRED_ID)
        self.position = 0
        index = cache / "segments.json"
        if index.is_file():
            self.size = int(json.loads(index.read_text(encoding="utf-8"))["archive_size"])
        else:
            _, self.size = fetch_range(self.url, 0, 4)
            install_bytes(index, json.dumps({"archive_size": self.size}).encode())
        self.segments = {}
        for block in cache.glob("*.bin"):
            match = re.fullmatch(r"(\d+)-(\d+)\.bin", block.name)
            if match:
                start, end = map(int, match.groups())
                if 0 <= start <= end < self.size and block.stat().st_size == end - start + 1:
                    self.segments[start] = (end + 1, block)

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        bases = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}
        position = bases[whence] + offset
        if position < 0:
            raise OSError("ZIP seek before start of file")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        stop = self.size if size < 0 else min(self.size, self.position + size)
        parts = []
        while self.position < stop:
            start = max((offset for offset in self.segments if offset <= self.position), default=-1)
            if start < 0 or self.position >= self.segments[start][0]:
                start = self.position
                next_block = min((offset for offset in self.segments if offset > start), default=self.size)
                end = min(start + CHUNK_SIZE, next_block)
                data, _ = fetch_range(self.url, start, end, self.size)
                block = self.cache / f"{start}-{end - 1}.bin"
                install_bytes(block, data)
                self.segments[start] = (end, block)
            end, block = self.segments[start]
            count = min(stop, end) - self.position
            with block.open("rb") as stream:
                stream.seek(self.position - start)
                data = stream.read(count)
            if len(data) != count:
                raise OSError(f"Incomplete cached ZIP block: {block}")
            parts.append(data)
            self.position += count
        return b"".join(parts)


def extract_assets(assets: Iterable[Asset], read_member: Callable[[str], bytes]) -> None:
    for asset in assets:
        data = read_member(asset.member)
        if hashlib.sha256(data).hexdigest() != asset.sha256:
            raise ValueError(f"Archive member SHA-256 mismatch: {asset.member}")
        if asset.path.exists():
            if sha256(asset.path) != asset.sha256:
                raise ValueError(f"Destination changed during fetch: {asset.path}")
        else:
            install_bytes(asset.path, data)


def prepare_test_assets(manifest: dict, root: Path, paired_zip: Path | None, c60_rar: Path | None) -> int:
    assets = list(iter_assets(manifest, root))
    settings = verification_settings(manifest)
    missing = {"uieb_paired": [], "uieb_c60": []}
    for subset, asset in assets:
        if asset.path.exists():
            verify_asset(asset, *settings)
        else:
            missing[subset].append(asset)

    downloads = root / ".cache/downloads"
    if missing["uieb_paired"]:
        print(f"Preparing {len(missing['uieb_paired'])} paired files", flush=True)
        if paired_zip is None:
            with RangedZip(downloads / "wwe_uie_ui_test_ranges") as source, ZipFile(source) as archive:
                ordered = sorted(missing["uieb_paired"], key=lambda asset: archive.getinfo(asset.member).header_offset)
                extract_assets(ordered, archive.read)
        else:
            with ZipFile(paired_zip) as archive:
                extract_assets(missing["uieb_paired"], archive.read)
    if missing["uieb_c60"]:
        print(f"Preparing {len(missing['uieb_c60'])} C60 files", flush=True)
        if c60_rar is None:
            c60_rar = downloads / "uieb_challenging_60.rar"
            download(drive_url(C60_ID), c60_rar, binary=True)
        if sha256(c60_rar) != manifest["subsets"]["uieb_c60"]["archive_sha256"]:
            raise ValueError(f"C60 archive SHA-256 mismatch: {c60_rar}")
        unrar = shutil.which("unrar")
        if unrar is None and os.name == "nt":
            tar = shutil.which("tar")
            if tar is not None:
                unrar = str(prepare_unrar(tar))
        if unrar is None:
            raise OSError("UnRAR is required to read the encrypted official C60 archive")

        def read_c60(member: str) -> bytes:
            name = member.replace("/", "\\") if os.name == "nt" else member
            return subprocess.check_output([unrar, "p", "-inul", f"-p{C60_PASSWORD}", str(c60_rar), name])

        extract_assets(missing["uieb_c60"], read_c60)

    for group in missing.values():
        for asset in group:
            verify_asset(asset, *settings)
    return len(assets)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Verify the local files without any downloads.")
    parser.add_argument("--paired-zip", type=Path, help="Use a local copy of the author's paired archive.")
    parser.add_argument("--c60-rar", type=Path, help="Use a local copy of the official C60 archive.")
    args = parser.parse_args()
    if args.verify_only and (args.paired_zip is not None or args.c60_rar is not None):
        parser.error("Archive arguments apply only when fetching missing files")
    try:
        manifest = read_manifest()
        if args.verify_only:
            count = verify_manifest(manifest)
        else:
            count = prepare_test_assets(manifest, ROOT, args.paired_zip, args.c60_rar)
    except (OSError, ValueError, KeyError, BadZipFile, subprocess.CalledProcessError) as error:
        print(f"Test asset preparation failed: {error}", file=sys.stderr)
        return 1
    print(f"Verified {count} test image files against {MANIFEST.relative_to(ROOT)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
