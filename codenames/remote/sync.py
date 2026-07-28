"""Pull run artifacts from Drive into the local ``output/`` tree.

Replaces the download script that was lost in the 2026-06-18 cleanup. Skipping
is checksum-based where Drive provides an md5, because size-and-mtime
comparison across a FUSE mount and a Drive round-trip is not trustworthy.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from .drive import FOLDER_MIME, download_file, list_folder

# The .npz vector subsamples and the .npy lens dumps are the multi-GB files.
# codenames/viz and the boards/trust/examples aggregate steps require them, so
# skip_heavy is for quick iteration only, never for a final pull.
_HEAVY_SUFFIXES = ("_f16.npz", "_f16.npy")


@dataclass(frozen=True)
class RemoteFile:
    file_id: str
    name: str
    size: int
    md5: Optional[str] = None


@dataclass
class SyncReport:
    downloaded: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    bytes_downloaded: int = 0


def is_heavy_artifact(name: str) -> bool:
    return name.endswith(_HEAVY_SUFFIXES)


def filter_files(
    files: Sequence[RemoteFile], *, skip_heavy: bool = False, only_heavy: bool = False
) -> List[RemoteFile]:
    if skip_heavy and only_heavy:
        raise ValueError("skip_heavy and only_heavy are mutually exclusive")
    if skip_heavy:
        return [f for f in files if not is_heavy_artifact(f.name)]
    if only_heavy:
        return [f for f in files if is_heavy_artifact(f.name)]
    return list(files)


def _local_md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def needs_download(remote: RemoteFile, local_path: Path) -> bool:
    local_path = Path(local_path)
    if not local_path.exists():
        return True
    if remote.md5:
        return _local_md5(local_path) != remote.md5
    return local_path.stat().st_size != remote.size


def sync_model_outputs(
    service,
    drive_root_id: str,
    local_root: Path,
    models: Sequence[str],
    *,
    skip_heavy: bool = False,
    only_heavy: bool = False,
    dry_run: bool = False,
) -> SyncReport:
    """Mirror ``<model>_outputs/`` folders from Drive into ``local_root``."""
    report = SyncReport()
    local_root = Path(local_root)
    wanted = {f"{m}_outputs" for m in models}

    for entry in list_folder(service, drive_root_id):
        if entry["name"] not in wanted:
            continue
        dest_dir = local_root / entry["name"]
        files = [
            RemoteFile(
                file_id=child["id"],
                name=child["name"],
                size=int(child.get("size", 0) or 0),
                md5=child.get("md5Checksum"),
            )
            for child in list_folder(service, entry["id"])
            if child.get("mimeType") != FOLDER_MIME
        ]
        for remote in filter_files(files, skip_heavy=skip_heavy, only_heavy=only_heavy):
            dest = dest_dir / remote.name
            if not needs_download(remote, dest):
                report.skipped.append(remote.name)
                continue
            if not dry_run:
                download_file(service, remote.file_id, dest)
            report.downloaded.append(remote.name)
            report.bytes_downloaded += remote.size
    return report
