"""Local-side operations against a job store.

Backend-agnostic by construction: every function here takes a ``JobStore``, so
the same code path runs against a local directory in tests and against Drive in
production.
"""

import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence

from .protocol import (
    TERMINAL_STATES,
    Heartbeat,
    Job,
    JobStatus,
    make_job_id,
    parse_iso,
    utc_now_iso,
    validate_job,
)
from .store import JobStore

DEFAULT_TIMEOUT_S = 6 * 3600
STALE_AFTER_S = 120.0


def submit(
    store: JobStore,
    argv: Sequence[str],
    *,
    git_ref: str,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    expect_gpu: Optional[str] = None,
    now: Optional[datetime] = None,
) -> Job:
    """Validate and enqueue a job. Raises ValueError if it would be rejected."""
    now = now or datetime.now(timezone.utc)
    job = Job(
        job_id=make_job_id(argv, now),
        created_at=utc_now_iso(now),
        git_ref=git_ref,
        argv=tuple(argv),
        timeout_s=timeout_s,
        expect_gpu=expect_gpu,
    )
    reason = validate_job(job)
    if reason is not None:
        # Fail here rather than letting a doomed job sit in the queue until a
        # runner picks it up and rejects it minutes later.
        raise ValueError(f"refusing to submit: {reason}")
    store.ensure_layout()
    store.put_job(job)
    return job


@dataclass(frozen=True)
class RunnerHealth:
    alive: bool
    age_s: Optional[float]
    heartbeat: Optional[Heartbeat]


def runner_health(
    store: JobStore,
    now: Optional[datetime] = None,
    stale_after_s: float = STALE_AFTER_S,
) -> RunnerHealth:
    now = now or datetime.now(timezone.utc)
    hb = store.read_heartbeat()
    if hb is None:
        return RunnerHealth(alive=False, age_s=None, heartbeat=None)
    try:
        stamped = parse_iso(hb.updated_at)
    except ValueError:
        return RunnerHealth(alive=False, age_s=None, heartbeat=hb)
    age = (now - stamped).total_seconds()
    return RunnerHealth(alive=age < stale_after_s, age_s=age, heartbeat=hb)


def logs(store: JobStore, job_id: str, tail: Optional[int] = None) -> str:
    text = store.read_log(job_id)
    if tail is None:
        return text
    lines = [ln for ln in text.splitlines() if ln != ""]
    return "\n".join(lines[-tail:])


def cancel(store: JobStore, job_id: str) -> bool:
    """Request cancellation. Returns False if the job already finished."""
    status = store.read_status(job_id)
    if status is not None and status.state in TERMINAL_STATES:
        return False
    store.request_cancel(job_id)
    return True


def has_unfinished_work(store: JobStore, limit: int = 50) -> bool:
    """True if anything is queued or still short of a terminal state."""
    if store.list_queued():
        return True
    return any(s.state not in TERMINAL_STATES for s in store.list_statuses(limit=limit))


def format_status_line(status: JobStatus) -> str:
    bits = [f"{status.job_id:<44}", f"{status.state.value:<10}"]
    if status.exit_code is not None:
        bits.append(f"exit={status.exit_code}")
    if status.gpu_name:
        bits.append(status.gpu_name)
    if status.reason:
        bits.append(f"({status.reason})")
    return "  ".join(bits)


class KeepAwake:
    """Hold macOS awake for the duration of a monitored job.

    Colab Pro has no background execution, so a sleeping Mac drops the tab and
    the session with it. No-op on other platforms.
    """

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None

    def __enter__(self) -> "KeepAwake":
        if sys.platform == "darwin":
            try:
                self.process = subprocess.Popen(
                    ["caffeinate", "-dimsu"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except (FileNotFoundError, OSError):
                self.process = None
        return self

    def __exit__(self, *exc_info) -> None:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
