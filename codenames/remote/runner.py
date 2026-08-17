"""The GPU-side agent loop.

Runs inside a Colab session with Drive mounted. Claims jobs from the queue,
executes whitelisted ``codenames-experiment`` subcommands as subprocesses,
mirrors their output to Drive, and publishes a heartbeat so the client can tell
a live session from a dead one.

Two rules shape the implementation:

1. Nothing is executed through a shell, and ``argv[0]`` must be whitelisted.
2. Logs are written to local disk and copied to Drive whole, on a timer.
   Appending through the Drive FUSE mount is the operation that fails in
   practice.
"""

import os
import shlex
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Tuple

from .protocol import (
    Heartbeat,
    Job,
    JobState,
    JobStatus,
    parse_iso,
    utc_now_iso,
    validate_job,
)
from .store import JobStore, LocalDirStore

LOG_TAIL_LINES = 40
HEARTBEAT_STALE_AFTER_S = 120.0


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class RunnerConfig:
    repo_dir: Path
    jobs_root: Path
    local_log_dir: Path
    runner_id: str = field(default_factory=lambda: f"r-{uuid.uuid4().hex[:10]}")
    executable: str = "codenames-experiment"
    poll_interval_s: float = 5.0
    heartbeat_interval_s: float = 30.0
    log_flush_interval_s: float = 15.0
    idle_shutdown_minutes: float = 30.0
    max_session_hours: float = 11.0
    grace_period_s: float = 30.0
    skip_git: bool = False


def gpu_info() -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """Return (device name, free MiB, total MiB), or (None, None, None) on CPU."""
    try:
        import torch

        if not torch.cuda.is_available():
            return (None, None, None)
        free_b, total_b = torch.cuda.mem_get_info()
        return (
            torch.cuda.get_device_name(0),
            free_b // (1024 * 1024),
            total_b // (1024 * 1024),
        )
    except Exception:
        return (None, None, None)


def checkout_ref(repo_dir: Path, git_ref: str) -> str:
    """Fetch and hard-reset the repo to ``git_ref``; return the resolved SHA."""
    repo = str(repo_dir)
    subprocess.run(
        ["git", "-C", repo, "fetch", "origin", git_ref], check=True, capture_output=True
    )
    subprocess.run(
        ["git", "-C", repo, "checkout", git_ref], check=True, capture_output=True
    )
    subprocess.run(
        ["git", "-C", repo, "reset", "--hard", f"origin/{git_ref}"],
        check=True,
        capture_output=True,
    )
    out = subprocess.run(
        ["git", "-C", repo, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def tail_text(text: str, n_lines: int) -> str:
    lines = [ln for ln in text.splitlines() if ln != ""]
    return "\n".join(lines[-n_lines:])


def _terminate(proc: subprocess.Popen, grace_period_s: float) -> None:
    """SIGINT first so checkpoint code can flush, SIGKILL only if it must."""
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=grace_period_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=grace_period_s)
        except subprocess.TimeoutExpired:
            pass
    except (ProcessLookupError, OSError):
        pass


def execute_job(
    job: Job,
    cfg: RunnerConfig,
    store: JobStore,
    now_fn: Callable[[], datetime] = _utc_now,
    repo_sha: Optional[str] = None,
    started_mono: Optional[float] = None,
) -> JobStatus:
    """Run one job to completion and return its terminal status."""
    gpu_name, _free, _total = gpu_info()

    def _fail(reason: str) -> JobStatus:
        status = JobStatus(
            job_id=job.job_id,
            state=JobState.FAILED,
            updated_at=utc_now_iso(now_fn()),
            reason=reason,
            gpu_name=gpu_name,
            repo_sha=repo_sha,
        )
        store.write_status(status)
        return status

    reason = validate_job(job)
    if reason is not None:
        return _fail(reason)

    if job.expect_gpu:
        if gpu_name is None:
            return _fail(
                f"job expects GPU '{job.expect_gpu}' but no CUDA device is visible"
            )
        if job.expect_gpu.lower() not in gpu_name.lower():
            return _fail(
                f"job expects GPU '{job.expect_gpu}' but this session has '{gpu_name}'"
            )

    started_at = utc_now_iso(now_fn())
    store.write_status(
        JobStatus(
            job_id=job.job_id,
            state=JobState.RUNNING,
            updated_at=started_at,
            started_at=started_at,
            gpu_name=gpu_name,
            repo_sha=repo_sha,
        )
    )

    cfg.local_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg.local_log_dir / f"{job.job_id}.log"
    argv = shlex.split(cfg.executable) + list(job.argv)

    deadline = time.monotonic() + job.timeout_s
    next_flush = time.monotonic()
    next_heartbeat = time.monotonic()
    outcome = JobState.SUCCEEDED
    exit_code: Optional[int] = None
    fail_reason: Optional[str] = None

    with open(log_path, "w", encoding="utf-8") as log_fh:
        proc = subprocess.Popen(
            argv,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=str(cfg.repo_dir) if cfg.repo_dir.is_dir() else None,
            shell=False,
        )
        while True:
            try:
                exit_code = proc.wait(timeout=cfg.poll_interval_s)
                break
            except subprocess.TimeoutExpired:
                pass

            now_mono = time.monotonic()
            if now_mono >= next_flush:
                log_fh.flush()
                store.put_log(job.job_id, log_path)
                next_flush = now_mono + cfg.log_flush_interval_s
            if now_mono >= next_heartbeat:
                # Keep the heartbeat alive DURING the job: without this a
                # multi-hour patch run reads as STALE from the client even
                # while its log is advancing, and the operator's next move
                # ("re-run the runner cell") would start a second runner on
                # the same output directory.
                write_heartbeat(cfg, store, now_fn, current_job=job.job_id,
                                repo_sha=repo_sha, started_mono=started_mono)
                next_heartbeat = now_mono + cfg.heartbeat_interval_s

            if store.is_cancel_requested(job.job_id):
                _terminate(proc, cfg.grace_period_s)
                exit_code = proc.returncode
                outcome = JobState.CANCELLED
                fail_reason = "cancelled by request"
                break

            if now_mono >= deadline:
                _terminate(proc, cfg.grace_period_s)
                exit_code = proc.returncode
                outcome = JobState.TIMED_OUT
                fail_reason = f"exceeded timeout_s={job.timeout_s}"
                break

    store.put_log(job.job_id, log_path)
    store.clear_cancel(job.job_id)

    if outcome is JobState.SUCCEEDED and exit_code != 0:
        outcome = JobState.FAILED
        fail_reason = f"subcommand exited with code {exit_code}"

    ended_at = utc_now_iso(now_fn())
    status = JobStatus(
        job_id=job.job_id,
        state=outcome,
        updated_at=ended_at,
        exit_code=exit_code,
        started_at=started_at,
        ended_at=ended_at,
        gpu_name=gpu_name,
        repo_sha=repo_sha,
        reason=fail_reason,
        log_tail=tail_text(store.read_log(job.job_id), LOG_TAIL_LINES),
    )
    store.write_status(status)
    return status


def export_hf_token(jobs_root: Path) -> bool:
    """Put a Hugging Face token into the environment jobs inherit.

    Gated Hub repos (meta-llama/Llama-3.1-8B-Instruct) need a token. An
    explicit ``HF_TOKEN`` in the environment (Colab Secret via the notebook's
    Cell 3c) wins; otherwise ``<jobs_root>/../_secrets/hf_token`` -- a file
    uploaded once to the same private Drive folder as the job queue -- is
    used. Returns True if a token is set afterwards.
    """
    if os.environ.get("HF_TOKEN"):
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])
        return True
    path = Path(jobs_root).parent / "_secrets" / "hf_token"
    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    if not token:
        return False
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    print("[runner] HF token loaded from Drive secrets file (gated repos reachable).")
    return True


def write_heartbeat(
    cfg: RunnerConfig,
    store: JobStore,
    now_fn: Callable[[], datetime] = _utc_now,
    current_job: Optional[str] = None,
    repo_sha: Optional[str] = None,
    started_mono: Optional[float] = None,
    last_log_line: str = "",
) -> Heartbeat:
    gpu_name, free_mb, total_mb = gpu_info()
    hb = Heartbeat(
        runner_id=cfg.runner_id,
        updated_at=utc_now_iso(now_fn()),
        gpu_name=gpu_name,
        vram_free_mb=free_mb,
        vram_total_mb=total_mb,
        current_job=current_job,
        repo_sha=repo_sha,
        uptime_s=(time.monotonic() - started_mono) if started_mono else 0.0,
        last_log_line=last_log_line,
    )
    store.write_heartbeat(hb)
    return hb


def another_runner_is_alive(
    store: JobStore, cfg: RunnerConfig, now_fn: Callable[[], datetime] = _utc_now
) -> bool:
    """True if a *different* runner published a heartbeat recently.

    This is the whole mutual-exclusion story. Two runners sharing one output
    directory would corrupt artifacts that cost hundreds of thousands of A100
    forward passes, so a second session declines to start rather than racing.
    """
    hb = store.read_heartbeat()
    if hb is None or hb.runner_id == cfg.runner_id:
        return False
    try:
        age = (now_fn() - parse_iso(hb.updated_at)).total_seconds()
    except ValueError:
        return False
    return age < HEARTBEAT_STALE_AFTER_S


def run_agent_loop(
    cfg: RunnerConfig,
    store: Optional[JobStore] = None,
    now_fn: Callable[[], datetime] = _utc_now,
    max_iterations: Optional[int] = None,
) -> int:
    """Poll for jobs until the idle budget or the session limit runs out.

    Returns 0 on a clean shutdown, 2 if it declined to start beside another
    live runner.
    """
    store = store or LocalDirStore(cfg.jobs_root)
    store.ensure_layout()

    if another_runner_is_alive(store, cfg, now_fn):
        hb = store.read_heartbeat()
        print(f"[runner] another runner is live ({hb.runner_id}); refusing to start.")
        return 2

    export_hf_token(cfg.jobs_root)

    started_mono = time.monotonic()
    idle_since = started_mono
    next_heartbeat = 0.0
    iterations = 0
    print(f"[runner] {cfg.runner_id} polling {cfg.jobs_root}")

    while True:
        if max_iterations is not None and iterations >= max_iterations:
            return 0
        iterations += 1

        now_mono = time.monotonic()
        if (now_mono - started_mono) >= cfg.max_session_hours * 3600.0:
            print("[runner] session hour limit reached; shutting down.")
            return 0

        queued = store.list_queued()
        if not queued:
            # Heartbeat on its own cadence: polling is frequent, but writing
            # every poll would hammer the Drive mount for no added liveness.
            if now_mono >= next_heartbeat:
                write_heartbeat(cfg, store, now_fn, started_mono=started_mono)
                next_heartbeat = now_mono + cfg.heartbeat_interval_s
            if (now_mono - idle_since) >= cfg.idle_shutdown_minutes * 60.0:
                print(
                    "[runner] idle budget exhausted; shutting down to save compute units."
                )
                return 0
            time.sleep(cfg.poll_interval_s)
            continue

        job = store.claim(queued[0].job_id)
        if job is None:
            continue

        repo_sha = None
        if not cfg.skip_git:
            try:
                repo_sha = checkout_ref(cfg.repo_dir, job.git_ref)
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or b"").decode("utf-8", "replace")[-500:]
                store.write_status(
                    JobStatus(
                        job_id=job.job_id,
                        state=JobState.FAILED,
                        updated_at=utc_now_iso(now_fn()),
                        reason=f"git checkout of '{job.git_ref}' failed: {stderr}",
                    )
                )
                idle_since = time.monotonic()
                continue

        write_heartbeat(
            cfg,
            store,
            now_fn,
            current_job=job.job_id,
            repo_sha=repo_sha,
            started_mono=started_mono,
        )
        print(f"[runner] running {job.job_id}: {' '.join(job.argv)}")
        status = execute_job(job, cfg, store, now_fn, repo_sha=repo_sha,
                             started_mono=started_mono)
        print(f"[runner] {job.job_id} -> {status.state.value}")

        idle_since = time.monotonic()
        next_heartbeat = 0.0
        write_heartbeat(
            cfg,
            store,
            now_fn,
            repo_sha=repo_sha,
            started_mono=started_mono,
            last_log_line=tail_text(status.log_tail, 1),
        )
