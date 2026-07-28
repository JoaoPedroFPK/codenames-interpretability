"""Job-store interface plus its filesystem backend.

The runner sees Drive as a mounted filesystem, so it uses ``LocalDirStore`` and
needs no credentials at all. The Drive-API backend (``codenames/remote/drive.py``)
implements the same interface for the local client. Tests use ``LocalDirStore``
throughout, which is why almost none of them need network or mocks.
"""

import abc
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from .protocol import Heartbeat, Job, JobPaths, JobStatus


def atomic_write_text(path: Path, text: str) -> None:
    """Write via a temp file in the same directory, then replace.

    Colab's Drive FUSE mount does not guarantee much, but a reader that opens a
    path either sees the old bytes or the new ones — never a half-written file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _read_json(path: Path) -> Optional[dict]:
    """Read JSON, treating unreadable or partial content as absent."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


class JobStore(abc.ABC):
    """Everything both sides of the queue need, and nothing more."""

    @abc.abstractmethod
    def ensure_layout(self) -> None: ...

    @abc.abstractmethod
    def put_job(self, job: Job) -> None: ...

    @abc.abstractmethod
    def list_queued(self) -> List[Job]: ...

    @abc.abstractmethod
    def claim(self, job_id: str) -> Optional[Job]: ...

    @abc.abstractmethod
    def read_job(self, job_id: str) -> Optional[Job]: ...

    @abc.abstractmethod
    def write_status(self, status: JobStatus) -> None: ...

    @abc.abstractmethod
    def read_status(self, job_id: str) -> Optional[JobStatus]: ...

    @abc.abstractmethod
    def list_statuses(self, limit: int = 20) -> List[JobStatus]: ...

    @abc.abstractmethod
    def write_heartbeat(self, hb: Heartbeat) -> None: ...

    @abc.abstractmethod
    def read_heartbeat(self) -> Optional[Heartbeat]: ...

    @abc.abstractmethod
    def put_log(self, job_id: str, local_path: Path) -> None: ...

    @abc.abstractmethod
    def read_log(self, job_id: str) -> str: ...

    @abc.abstractmethod
    def request_cancel(self, job_id: str) -> None: ...

    @abc.abstractmethod
    def is_cancel_requested(self, job_id: str) -> bool: ...

    @abc.abstractmethod
    def clear_cancel(self, job_id: str) -> None: ...


class LocalDirStore(JobStore):
    """Job store backed by an ordinary directory tree."""

    def __init__(self, root: Path):
        self.paths = JobPaths(Path(root))

    def ensure_layout(self) -> None:
        for d in self.paths.all_dirs():
            d.mkdir(parents=True, exist_ok=True)

    def put_job(self, job: Job) -> None:
        atomic_write_text(
            self.paths.job_file(job.job_id), json.dumps(job.to_dict(), indent=2)
        )

    def list_queued(self) -> List[Job]:
        if not self.paths.queue.is_dir():
            return []
        jobs = []
        for path in sorted(self.paths.queue.glob("*.json")):
            raw = _read_json(path)
            if raw is not None:
                jobs.append(Job.from_dict(raw))
        return jobs

    def claim(self, job_id: str) -> Optional[Job]:
        src = self.paths.job_file(job_id)
        raw = _read_json(src)
        if raw is None:
            return None
        dst = self.paths.claimed_file(job_id)
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(src, dst)
        except OSError:
            return None
        return Job.from_dict(raw)

    def read_job(self, job_id: str) -> Optional[Job]:
        for path in (self.paths.job_file(job_id), self.paths.claimed_file(job_id)):
            raw = _read_json(path)
            if raw is not None:
                return Job.from_dict(raw)
        return None

    def write_status(self, status: JobStatus) -> None:
        atomic_write_text(
            self.paths.status_file(status.job_id),
            json.dumps(status.to_dict(), indent=2),
        )

    def read_status(self, job_id: str) -> Optional[JobStatus]:
        raw = _read_json(self.paths.status_file(job_id))
        return JobStatus.from_dict(raw) if raw is not None else None

    def list_statuses(self, limit: int = 20) -> List[JobStatus]:
        if not self.paths.status.is_dir():
            return []
        out = []
        for path in sorted(self.paths.status.glob("*.json"), reverse=True)[:limit]:
            raw = _read_json(path)
            if raw is not None:
                out.append(JobStatus.from_dict(raw))
        return out

    def write_heartbeat(self, hb: Heartbeat) -> None:
        atomic_write_text(
            self.paths.heartbeat_file(), json.dumps(hb.to_dict(), indent=2)
        )

    def read_heartbeat(self) -> Optional[Heartbeat]:
        raw = _read_json(self.paths.heartbeat_file())
        return Heartbeat.from_dict(raw) if raw is not None else None

    def put_log(self, job_id: str, local_path: Path) -> None:
        dst = self.paths.log_file(job_id)
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Whole-file replacement, never append: appending through Drive's FUSE
        # mount is the unreliable operation this design exists to avoid.
        shutil.copyfile(local_path, dst)

    def read_log(self, job_id: str) -> str:
        try:
            return self.paths.log_file(job_id).read_text(
                encoding="utf-8", errors="replace"
            )
        except (FileNotFoundError, OSError):
            return ""

    def request_cancel(self, job_id: str) -> None:
        path = self.paths.cancel_file(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("cancel", encoding="utf-8")

    def is_cancel_requested(self, job_id: str) -> bool:
        return self.paths.cancel_file(job_id).exists()

    def clear_cancel(self, job_id: str) -> None:
        try:
            self.paths.cancel_file(job_id).unlink()
        except FileNotFoundError:
            pass
