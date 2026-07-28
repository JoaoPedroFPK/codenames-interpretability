"""Drive API backend for the job store, plus its OAuth bootstrap.

Only the local client uses this. The Colab runner sees the same tree through a
FUSE mount and uses ``LocalDirStore``.

Scope note: ``drive.file`` is insufficient. It grants access only to files this
client created, but status documents and logs are created by the Colab Drive
mount — a different application — so the client would be blind to exactly the
files it needs to read.
"""

import io
import json
from pathlib import Path
from typing import List, Optional, Sequence

from .protocol import Heartbeat, Job, JobStatus
from .store import JobStore

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]
FOLDER_MIME = "application/vnd.google-apps.folder"
_SUBDIRS = ("queue", "claimed", "status", "logs", "runner", "control")


def load_credentials(token_path: Path, client_secret_path: Path, *, scopes=DRIVE_SCOPES):
    """Load cached credentials, refreshing or re-consenting as needed.

    The token cached by the old read-only tooling carries a narrower scope, so
    the first call after this feature lands triggers a consent screen.
    """
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    token_path = Path(token_path)
    creds = None
    if token_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), scopes)
        except ValueError:
            creds = None  # cached token was issued for different scopes

    has_scope = creds is not None and set(scopes) <= set(creds.scopes or [])
    if creds and creds.valid and has_scope:
        return creds
    if creds and has_scope and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_path), scopes)
        creds = flow.run_local_server(port=0)
    token_path.write_text(creds.to_json(), encoding="utf-8")
    return creds


def build_drive_service(creds):
    from googleapiclient.discovery import build

    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _escape(name: str) -> str:
    return name.replace("'", "\\'")


def text_media(text: str):
    """Upload body for a small text payload.

    The raw text is also kept on the object as ``_content`` so a test double can
    see what would be uploaded without reimplementing ``MediaIoBaseUpload``.
    """
    from googleapiclient.http import MediaIoBaseUpload

    media = MediaIoBaseUpload(
        io.BytesIO(text.encode("utf-8")), mimetype="text/plain", resumable=False
    )
    media._content = text
    return media


def find_file(service, folder_id: str, name: str) -> Optional[dict]:
    resp = (
        service.files()
        .list(
            q=f"'{folder_id}' in parents and name = '{_escape(name)}' and trashed = false",
            fields="files(id,name,mimeType,size,md5Checksum,modifiedTime)",
            pageSize=10,
        )
        .execute()
    )
    files = resp.get("files", [])
    return files[0] if files else None


def list_folder(service, folder_id: str) -> List[dict]:
    out: List[dict] = []
    token = None
    while True:
        kwargs = dict(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken, files(id,name,mimeType,size,md5Checksum,modifiedTime)",
            pageSize=1000,
        )
        if token:
            kwargs["pageToken"] = token
        resp = service.files().list(**kwargs).execute()
        out.extend(resp.get("files", []))
        token = resp.get("nextPageToken")
        if not token:
            return out


def resolve_folder(
    service,
    path_parts: Sequence[str],
    *,
    create: bool = False,
    parent: str = "root",
) -> Optional[str]:
    """Walk a folder path from ``parent``, optionally creating missing levels."""
    current = parent
    for part in path_parts:
        found = find_file(service, current, part)
        if found is not None and found.get("mimeType") == FOLDER_MIME:
            current = found["id"]
            continue
        if not create:
            return None
        created = (
            service.files()
            .create(
                body={"name": part, "mimeType": FOLDER_MIME, "parents": [current]},
                fields="id",
            )
            .execute()
        )
        current = created["id"]
    return current


def upload_text(service, folder_id: str, name: str, text: str) -> str:
    """Create or update a small text file; returns its file id."""
    media = text_media(text)
    existing = find_file(service, folder_id, name)
    if existing is not None:
        service.files().update(
            fileId=existing["id"], media_body=media, fields="id"
        ).execute()
        return existing["id"]
    created = (
        service.files()
        .create(
            body={"name": name, "parents": [folder_id]}, media_body=media, fields="id"
        )
        .execute()
    )
    return created["id"]


def read_text(service, file_id: str) -> str:
    request = service.files().get_media(fileId=file_id)
    # Test doubles hand back the bytes directly; the real client hands back an
    # HttpRequest that has to be streamed.
    if hasattr(request, "content"):
        return request.content
    from googleapiclient.http import MediaIoBaseDownload

    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _status, done = downloader.next_chunk()
    return buf.getvalue().decode("utf-8", "replace")


def download_file(service, file_id: str, dest: Path) -> None:
    """Stream a file to ``dest``, via a .part file so partials are never mistaken
    for complete downloads."""
    from googleapiclient.http import MediaIoBaseDownload

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, service.files().get_media(fileId=file_id))
        done = False
        while not done:
            _status, done = downloader.next_chunk()
    tmp.replace(dest)


class DriveApiStore(JobStore):
    """JobStore over the Drive v3 API, mirroring LocalDirStore's semantics."""

    def __init__(self, service, root_folder_id: str):
        self.service = service
        self.root_folder_id = root_folder_id
        self._folders = {}

    def _folder(self, name: str) -> str:
        if name not in self._folders:
            self._folders[name] = resolve_folder(
                self.service, [name], create=True, parent=self.root_folder_id
            )
        return self._folders[name]

    def ensure_layout(self) -> None:
        for name in _SUBDIRS:
            self._folder(name)

    # --- jobs -------------------------------------------------------------
    def put_job(self, job: Job) -> None:
        upload_text(
            self.service,
            self._folder("queue"),
            f"{job.job_id}.json",
            json.dumps(job.to_dict(), indent=2),
        )

    def list_queued(self) -> List[Job]:
        jobs = []
        for meta in sorted(
            list_folder(self.service, self._folder("queue")), key=lambda m: m["name"]
        ):
            if not meta["name"].endswith(".json"):
                continue
            try:
                jobs.append(
                    Job.from_dict(json.loads(read_text(self.service, meta["id"])))
                )
            except (json.JSONDecodeError, KeyError):
                continue
        return jobs

    def claim(self, job_id: str) -> Optional[Job]:
        meta = find_file(self.service, self._folder("queue"), f"{job_id}.json")
        if meta is None:
            return None
        text = read_text(self.service, meta["id"])
        upload_text(self.service, self._folder("claimed"), f"{job_id}.json", text)
        self.service.files().delete(fileId=meta["id"]).execute()
        return Job.from_dict(json.loads(text))

    def read_job(self, job_id: str) -> Optional[Job]:
        for folder in ("queue", "claimed"):
            meta = find_file(self.service, self._folder(folder), f"{job_id}.json")
            if meta is not None:
                try:
                    return Job.from_dict(json.loads(read_text(self.service, meta["id"])))
                except (json.JSONDecodeError, KeyError):
                    return None
        return None

    # --- status -----------------------------------------------------------
    def write_status(self, status: JobStatus) -> None:
        upload_text(
            self.service,
            self._folder("status"),
            f"{status.job_id}.json",
            json.dumps(status.to_dict(), indent=2),
        )

    def read_status(self, job_id: str) -> Optional[JobStatus]:
        meta = find_file(self.service, self._folder("status"), f"{job_id}.json")
        if meta is None:
            return None
        try:
            return JobStatus.from_dict(json.loads(read_text(self.service, meta["id"])))
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def list_statuses(self, limit: int = 20) -> List[JobStatus]:
        metas = sorted(
            list_folder(self.service, self._folder("status")),
            key=lambda m: m["name"],
            reverse=True,
        )[:limit]
        out = []
        for meta in metas:
            try:
                out.append(
                    JobStatus.from_dict(json.loads(read_text(self.service, meta["id"])))
                )
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        return out

    # --- heartbeat --------------------------------------------------------
    def write_heartbeat(self, hb: Heartbeat) -> None:
        upload_text(
            self.service,
            self._folder("runner"),
            "heartbeat.json",
            json.dumps(hb.to_dict(), indent=2),
        )

    def read_heartbeat(self) -> Optional[Heartbeat]:
        meta = find_file(self.service, self._folder("runner"), "heartbeat.json")
        if meta is None:
            return None
        try:
            return Heartbeat.from_dict(json.loads(read_text(self.service, meta["id"])))
        except (json.JSONDecodeError, KeyError):
            return None

    # --- logs -------------------------------------------------------------
    def put_log(self, job_id: str, local_path: Path) -> None:
        text = Path(local_path).read_text(encoding="utf-8", errors="replace")
        upload_text(self.service, self._folder("logs"), f"{job_id}.log", text)

    def read_log(self, job_id: str) -> str:
        meta = find_file(self.service, self._folder("logs"), f"{job_id}.log")
        return read_text(self.service, meta["id"]) if meta is not None else ""

    # --- cancellation -----------------------------------------------------
    def request_cancel(self, job_id: str) -> None:
        upload_text(self.service, self._folder("control"), f"cancel-{job_id}", "cancel")

    def is_cancel_requested(self, job_id: str) -> bool:
        return (
            find_file(self.service, self._folder("control"), f"cancel-{job_id}")
            is not None
        )

    def clear_cancel(self, job_id: str) -> None:
        meta = find_file(self.service, self._folder("control"), f"cancel-{job_id}")
        if meta is not None:
            self.service.files().delete(fileId=meta["id"]).execute()
