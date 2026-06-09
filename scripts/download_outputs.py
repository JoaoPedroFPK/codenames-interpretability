#!/usr/bin/env python3
"""Download the per-model experiment outputs from Google Drive into ``output/``.

The seven notebooks under ``notebooks/`` run on Colab and write their final
result files to Google Drive at::

    MyDrive/<drive-root>/<prefix>_outputs/<prefix>_<kind>_<mode>.<ext>

where

* ``<drive-root>``  defaults to ``TCC`` (see the notebooks' ``BASE_DIR``),
* ``<prefix>``      is one of the seven model prefixes (see ``MODEL_PREFIXES``),
* ``<kind>``        is ``general`` / ``metrics`` / ``generation`` / ``errors``,
* ``<mode>``        is ``no_social`` / ``with_social``,
* ``<ext>``         is ``csv`` (general/generation/errors) or ``parquet`` (metrics).

This script mirrors each ``<prefix>_outputs/`` folder verbatim into the local
``output/<prefix>_outputs/`` directory. It does NOT fetch the ``_checkpoints``
folders (intermediate state) unless ``--include-checkpoints`` is passed.

Authentication
--------------
Uses the Google OAuth 2.0 *installed application* flow with the desktop client
in ``client_secret.json`` (repo root). The first run opens a browser for you to
grant **read-only** Drive access; the resulting token is cached in ``token.json``
so subsequent runs are non-interactive. Both files are gitignored.

Because the first run needs a browser, run it yourself from the repo root::

    pip install -e ".[download]"
    python scripts/download_outputs.py

Common options::

    python scripts/download_outputs.py --models qwen random_qwen   # subset
    python scripts/download_outputs.py --drive-root TCC            # Drive folder
    python scripts/download_outputs.py --force                     # re-download all
    python scripts/download_outputs.py --dry-run                   # list, don't fetch
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from typing import Dict, List, Optional

# The seven model prefixes, mirroring the ``"prefix"`` metadata in
# ``codenames/models/*.py``. Hardcoded (rather than imported) so this script
# stays standalone and does not drag in the torch/transformers stack.
MODEL_PREFIXES = [
    "mistral",
    "qwen",
    "random_qwen",
    "bert",
    "random_bert",
    "t5",
    "modernbert",
]

# Read-only is all we need to download. Narrow scope = the token can never be
# used to mutate or delete anything in the user's Drive.
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

FOLDER_MIME = "application/vnd.google-apps.folder"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CLIENT_SECRET = os.path.join(REPO_ROOT, "client_secret.json")
DEFAULT_TOKEN = os.path.join(REPO_ROOT, "token.json")
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "output")


def _eprint(*args, **kwargs) -> None:
    print(*args, file=sys.stderr, **kwargs)


def _import_google():
    """Import the optional Google libraries with a helpful error if missing."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
        from googleapiclient.http import MediaIoBaseDownload
    except ImportError as exc:  # pragma: no cover - environment guard
        _eprint(
            "Missing Google API libraries. Install the downloader extra:\n"
            '    pip install -e ".[download]"\n'
            f"(original import error: {exc})"
        )
        sys.exit(2)
    return {
        "Request": Request,
        "Credentials": Credentials,
        "InstalledAppFlow": InstalledAppFlow,
        "build": build,
        "HttpError": HttpError,
        "MediaIoBaseDownload": MediaIoBaseDownload,
    }


def get_service(client_secret: str, token_path: str, google):
    """Build an authenticated Drive v3 service, caching the token locally."""
    Credentials = google["Credentials"]
    Request = google["Request"]
    InstalledAppFlow = google["InstalledAppFlow"]
    build = google["build"]

    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Refreshing cached Drive token...")
            creds.refresh(Request())
        else:
            if not os.path.exists(client_secret):
                _eprint(
                    f"Client secret not found at {client_secret}.\n"
                    "Place your Google OAuth desktop client JSON there "
                    "(or pass --client-secret)."
                )
                sys.exit(2)
            print(
                "Opening a browser for Google Drive consent "
                "(read-only access)..."
            )
            flow = InstalledAppFlow.from_client_secrets_file(client_secret, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as fh:
            fh.write(creds.to_json())
        print(f"Saved Drive token to {token_path}")

    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _list_children(service, parent_id: str, *, only_folders: bool = False) -> List[Dict]:
    """List non-trashed direct children of ``parent_id`` (handles pagination)."""
    q = f"'{parent_id}' in parents and trashed = false"
    if only_folders:
        q += f" and mimeType = '{FOLDER_MIME}'"
    items: List[Dict] = []
    page_token: Optional[str] = None
    while True:
        resp = (
            service.files()
            .list(
                q=q,
                spaces="drive",
                fields="nextPageToken, files(id, name, mimeType, size, md5Checksum)",
                pageSize=1000,
                pageToken=page_token,
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
            )
            .execute()
        )
        items.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return items


def find_folder(service, name: str, parent_id: str = "root") -> Optional[str]:
    """Return the id of the child folder named ``name`` under ``parent_id``."""
    safe = name.replace("'", "\\'")
    q = (
        f"name = '{safe}' and mimeType = '{FOLDER_MIME}' "
        f"and '{parent_id}' in parents and trashed = false"
    )
    resp = (
        service.files()
        .list(
            q=q,
            spaces="drive",
            fields="files(id, name)",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        )
        .execute()
    )
    files = resp.get("files", [])
    if not files:
        return None
    if len(files) > 1:
        _eprint(
            f"WARNING: {len(files)} folders named '{name}' under parent "
            f"'{parent_id}'. Using the first ({files[0]['id']})."
        )
    return files[0]["id"]


def download_file(service, file_meta: Dict, dest_path: str, google) -> bool:
    """Download one Drive file to ``dest_path``. Returns True if bytes were written."""
    MediaIoBaseDownload = google["MediaIoBaseDownload"]
    request = service.files().get_media(fileId=file_meta["id"], supportsAllDrives=True)
    tmp_path = dest_path + ".part"
    with io.FileIO(tmp_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=8 * 1024 * 1024)
        done = False
        while not done:
            _status, done = downloader.next_chunk()
    os.replace(tmp_path, dest_path)
    return True


def _should_skip(file_meta: Dict, dest_path: str, force: bool) -> bool:
    """Skip an existing local file when its size matches the Drive size."""
    if force or not os.path.exists(dest_path):
        return False
    remote_size = file_meta.get("size")
    if remote_size is None:
        return False  # size unknown (e.g. Google-native doc) -> re-fetch to be safe
    return os.path.getsize(dest_path) == int(remote_size)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drive-root", default="TCC", help="Drive folder holding the *_outputs dirs (default: TCC)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Local destination (default: ./output)")
    parser.add_argument("--client-secret", default=DEFAULT_CLIENT_SECRET, help="OAuth desktop client JSON")
    parser.add_argument("--token", default=DEFAULT_TOKEN, help="Cached OAuth token path")
    parser.add_argument("--models", nargs="*", default=MODEL_PREFIXES, metavar="PREFIX",
                        help=f"Subset of model prefixes (default: all 7 -> {', '.join(MODEL_PREFIXES)})")
    parser.add_argument("--include-checkpoints", action="store_true",
                        help="Also mirror each <prefix>_checkpoints/ folder")
    parser.add_argument("--skip-vectors", action="store_true",
                        help="Skip the heavy *_f16.npz hidden-state vector dumps "
                             "(~3.7 GB). The per-layer summaries and metrics still "
                             "download; vectors stay on Drive, re-fetchable later.")
    parser.add_argument("--force", action="store_true", help="Re-download even if a same-size local copy exists")
    parser.add_argument("--dry-run", action="store_true", help="List what would be downloaded, fetch nothing")
    args = parser.parse_args(argv)

    google = _import_google()
    HttpError = google["HttpError"]

    try:
        service = get_service(args.client_secret, args.token, google)
    except Exception as exc:  # pragma: no cover - auth/network failure surface
        _eprint(f"Authentication failed: {exc}")
        return 2

    root_id = find_folder(service, args.drive_root, "root")
    if root_id is None:
        _eprint(f"Drive root folder 'MyDrive/{args.drive_root}' not found.")
        return 1
    print(f"Drive root 'MyDrive/{args.drive_root}' -> {root_id}")

    suffixes = ["_outputs"]
    if args.include_checkpoints:
        suffixes.append("_checkpoints")

    total_files = 0
    total_bytes = 0
    skipped = 0
    missing_folders: List[str] = []

    for prefix in args.models:
        for suffix in suffixes:
            folder_name = f"{prefix}{suffix}"
            folder_id = find_folder(service, folder_name, root_id)
            if folder_id is None:
                missing_folders.append(folder_name)
                _eprint(f"  [skip] '{folder_name}' not found under '{args.drive_root}'")
                continue

            local_dir = os.path.join(args.output_dir, folder_name)
            try:
                files = _list_children(service, folder_id)
            except HttpError as exc:
                _eprint(f"  [error] listing '{folder_name}': {exc}")
                continue

            data_files = [f for f in files if f.get("mimeType") != FOLDER_MIME]
            if args.skip_vectors:
                data_files = [f for f in data_files if not f["name"].endswith(".npz")]
            print(f"\n{folder_name}/  ({len(data_files)} files) -> {os.path.relpath(local_dir, REPO_ROOT)}/")
            if not args.dry_run:
                os.makedirs(local_dir, exist_ok=True)

            for fmeta in sorted(data_files, key=lambda f: f["name"]):
                dest = os.path.join(local_dir, fmeta["name"])
                size = int(fmeta["size"]) if fmeta.get("size") is not None else 0
                if _should_skip(fmeta, dest, args.force):
                    print(f"    = {fmeta['name']} ({size:,} B, up-to-date)")
                    skipped += 1
                    continue
                if args.dry_run:
                    print(f"    + {fmeta['name']} ({size:,} B)")
                    total_files += 1
                    total_bytes += size
                    continue
                try:
                    download_file(service, fmeta, dest, google)
                    print(f"    ↓ {fmeta['name']} ({size:,} B)")
                    total_files += 1
                    total_bytes += size
                except HttpError as exc:
                    _eprint(f"    [error] {fmeta['name']}: {exc}")

    print("\n" + "=" * 60)
    verb = "Would download" if args.dry_run else "Downloaded"
    print(f"{verb}: {total_files} files, {total_bytes / 1e6:.1f} MB")
    if skipped:
        print(f"Skipped (already current): {skipped} files")
    if missing_folders:
        print(f"Missing on Drive: {', '.join(missing_folders)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
