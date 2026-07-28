"""Schemas, identifiers, and the job state machine.

Pure data: this module imports nothing from the rest of the package and does no
I/O, so both sides of the queue — the Colab runner and the local client — agree
on the wire format without agreeing on anything else.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

SCHEMA_VERSION = 1

# Only these may be executed by a runner. The queue lives in a personal Drive
# folder; without a whitelist, anything able to write a JSON file there would
# have arbitrary code execution on the GPU session.
ALLOWED_SUBCOMMANDS: FrozenSet[str] = frozenset(
    {
        "run",
        "lens-extract",
        "lens-tune",
        "doctor",
        "sanity",
        "preflight",
        "validate",
        "compare",
    }
)

_ID_TIME_FORMAT = "%Y%m%dT%H%M%SZ"
ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def utc_now_iso(now: datetime) -> str:
    """Format an aware UTC datetime as ISO-8601 with a trailing Z."""
    return now.strftime(ISO_FORMAT)


def parse_iso(stamp: str) -> datetime:
    """Inverse of :func:`utc_now_iso`. Raises ValueError on malformed input."""
    from datetime import timezone

    return datetime.strptime(stamp, ISO_FORMAT).replace(tzinfo=timezone.utc)


class JobState(str, Enum):
    QUEUED = "queued"
    CLAIMED = "claimed"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


TERMINAL_STATES: FrozenSet[JobState] = frozenset(
    {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED, JobState.TIMED_OUT}
)

_LEGAL_TRANSITIONS: Dict[JobState, FrozenSet[JobState]] = {
    JobState.QUEUED: frozenset({JobState.CLAIMED, JobState.FAILED, JobState.CANCELLED}),
    JobState.CLAIMED: frozenset({JobState.RUNNING, JobState.FAILED, JobState.CANCELLED}),
    JobState.RUNNING: frozenset(TERMINAL_STATES),
}


def is_legal_transition(src: JobState, dst: JobState) -> bool:
    return dst in _LEGAL_TRANSITIONS.get(src, frozenset())


@dataclass(frozen=True)
class Job:
    job_id: str
    created_at: str
    git_ref: str
    argv: Tuple[str, ...]
    timeout_s: int
    expect_gpu: Optional[str] = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        # JSON round-trips give lists; normalise so equality and hashing hold.
        object.__setattr__(self, "argv", tuple(self.argv))

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "created_at": self.created_at,
            "git_ref": self.git_ref,
            "argv": list(self.argv),
            "timeout_s": self.timeout_s,
            "expect_gpu": self.expect_gpu,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        return cls(
            job_id=d["job_id"],
            created_at=d["created_at"],
            git_ref=d["git_ref"],
            argv=tuple(d["argv"]),
            timeout_s=int(d["timeout_s"]),
            expect_gpu=d.get("expect_gpu"),
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class JobStatus:
    job_id: str
    state: JobState
    updated_at: str
    exit_code: Optional[int] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    repo_sha: Optional[str] = None
    gpu_name: Optional[str] = None
    reason: Optional[str] = None
    log_tail: str = ""

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "job_id": self.job_id,
            "state": self.state.value,
            "updated_at": self.updated_at,
            "exit_code": self.exit_code,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "repo_sha": self.repo_sha,
            "gpu_name": self.gpu_name,
            "reason": self.reason,
            "log_tail": self.log_tail,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "JobStatus":
        return cls(
            job_id=d["job_id"],
            state=JobState(d["state"]),
            updated_at=d["updated_at"],
            exit_code=d.get("exit_code"),
            started_at=d.get("started_at"),
            ended_at=d.get("ended_at"),
            repo_sha=d.get("repo_sha"),
            gpu_name=d.get("gpu_name"),
            reason=d.get("reason"),
            log_tail=d.get("log_tail", ""),
        )


@dataclass(frozen=True)
class Heartbeat:
    runner_id: str
    updated_at: str
    gpu_name: Optional[str] = None
    vram_free_mb: Optional[int] = None
    vram_total_mb: Optional[int] = None
    current_job: Optional[str] = None
    repo_sha: Optional[str] = None
    uptime_s: float = 0.0
    last_log_line: str = ""

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "runner_id": self.runner_id,
            "updated_at": self.updated_at,
            "gpu_name": self.gpu_name,
            "vram_free_mb": self.vram_free_mb,
            "vram_total_mb": self.vram_total_mb,
            "current_job": self.current_job,
            "repo_sha": self.repo_sha,
            "uptime_s": self.uptime_s,
            "last_log_line": self.last_log_line,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Heartbeat":
        return cls(
            runner_id=d["runner_id"],
            updated_at=d["updated_at"],
            gpu_name=d.get("gpu_name"),
            vram_free_mb=d.get("vram_free_mb"),
            vram_total_mb=d.get("vram_total_mb"),
            current_job=d.get("current_job"),
            repo_sha=d.get("repo_sha"),
            uptime_s=float(d.get("uptime_s", 0.0)),
            last_log_line=d.get("last_log_line", ""),
        )


def make_job_id(argv: Sequence[str], now: datetime) -> str:
    """Sortable, human-readable id: timestamp + subcommand + model if present."""
    stamp = now.strftime(_ID_TIME_FORMAT)
    argv = list(argv)
    parts = [stamp, argv[0] if argv else "job"]
    if "--model" in argv:
        idx = argv.index("--model")
        if idx + 1 < len(argv):
            parts.append(argv[idx + 1])
    return "-".join(parts)


def validate_job(job: Job) -> Optional[str]:
    """Return a human-readable rejection reason, or None if the job is runnable."""
    if job.schema_version > SCHEMA_VERSION:
        return (
            f"job schema version {job.schema_version} is newer than this runner "
            f"supports ({SCHEMA_VERSION}); update the runner"
        )
    if not job.argv:
        return "argv is empty"
    if job.argv[0] not in ALLOWED_SUBCOMMANDS:
        return (
            f"subcommand '{job.argv[0]}' is not in the runner whitelist "
            f"({', '.join(sorted(ALLOWED_SUBCOMMANDS))})"
        )
    if job.timeout_s <= 0:
        return "timeout_s must be positive"
    return None


@dataclass(frozen=True)
class JobPaths:
    """Directory layout of the jobs tree, relative to its root."""

    root: Path

    @property
    def queue(self) -> Path:
        return self.root / "queue"

    @property
    def claimed(self) -> Path:
        return self.root / "claimed"

    @property
    def status(self) -> Path:
        return self.root / "status"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def runner(self) -> Path:
        return self.root / "runner"

    @property
    def control(self) -> Path:
        return self.root / "control"

    def all_dirs(self) -> List[Path]:
        return [
            self.queue,
            self.claimed,
            self.status,
            self.logs,
            self.runner,
            self.control,
        ]

    def job_file(self, job_id: str) -> Path:
        return self.queue / f"{job_id}.json"

    def claimed_file(self, job_id: str) -> Path:
        return self.claimed / f"{job_id}.json"

    def status_file(self, job_id: str) -> Path:
        return self.status / f"{job_id}.json"

    def log_file(self, job_id: str) -> Path:
        return self.logs / f"{job_id}.log"

    def cancel_file(self, job_id: str) -> Path:
        return self.control / f"cancel-{job_id}"

    def heartbeat_file(self) -> Path:
        return self.runner / "heartbeat.json"
