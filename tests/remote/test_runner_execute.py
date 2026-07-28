"""execute_job: subprocess handling, status transitions, log capture, guards."""

import sys
from datetime import datetime, timedelta, timezone

import pytest

from codenames.remote.protocol import Job, JobState
from codenames.remote.runner import RunnerConfig, execute_job, tail_text
from codenames.remote.store import LocalDirStore

NOW = datetime(2026, 7, 27, 19, 30, 0, tzinfo=timezone.utc)


class Clock:
    """Deterministic UTC clock that advances a fixed step per call."""

    def __init__(self, start=NOW, step_s=1):
        self.t = start
        self.step = timedelta(seconds=step_s)

    def __call__(self):
        self.t += self.step
        return self.t


@pytest.fixture
def store(tmp_path):
    s = LocalDirStore(tmp_path / "_jobs")
    s.ensure_layout()
    return s


@pytest.fixture
def cfg(tmp_path):
    """Runner config whose 'executable' is a stub, not the real CLI.

    The stub reads its first argument as a subcommand name and behaves
    accordingly, so the runner's process handling is tested without ever
    loading a 7B model.
    """
    stub = tmp_path / "stub_cli.py"
    stub.write_text(
        "import sys, time\n"
        "cmd = sys.argv[1]\n"
        "if cmd == 'doctor':\n"
        "    print('stub ok'); sys.exit(0)\n"
        "if cmd == 'sanity':\n"
        "    print('stub boom', file=sys.stderr); sys.exit(3)\n"
        "if cmd == 'run':\n"
        "    print('starting', flush=True); time.sleep(30); sys.exit(0)\n",
        encoding="utf-8",
    )
    return RunnerConfig(
        repo_dir=tmp_path / "repo",
        jobs_root=tmp_path / "_jobs",
        local_log_dir=tmp_path / "logs",
        runner_id="r-test",
        executable=f"{sys.executable} {stub}",
        skip_git=True,
        poll_interval_s=0.05,
        log_flush_interval_s=0.05,
        grace_period_s=1.0,
    )


def _job(argv, job_id="j1", timeout_s=600, expect_gpu=None) -> Job:
    return Job(
        job_id=job_id,
        created_at="2026-07-27T19:30:00Z",
        git_ref="probing",
        argv=argv,
        timeout_s=timeout_s,
        expect_gpu=expect_gpu,
    )


def test_successful_job_reaches_succeeded_with_exit_zero(store, cfg):
    status = execute_job(_job(("doctor",)), cfg, store, Clock())
    assert status.state is JobState.SUCCEEDED
    assert status.exit_code == 0


def test_successful_job_captures_stdout_into_the_log(store, cfg):
    execute_job(_job(("doctor",)), cfg, store, Clock())
    assert "stub ok" in store.read_log("j1")


def test_failing_job_reaches_failed_and_records_the_exit_code(store, cfg):
    status = execute_job(_job(("sanity",)), cfg, store, Clock())
    assert status.state is JobState.FAILED
    assert status.exit_code == 3


def test_stderr_is_merged_into_the_log(store, cfg):
    execute_job(_job(("sanity",)), cfg, store, Clock())
    assert "stub boom" in store.read_log("j1")


def test_status_is_persisted_not_just_returned(store, cfg):
    execute_job(_job(("doctor",)), cfg, store, Clock())
    assert store.read_status("j1").state is JobState.SUCCEEDED


def test_running_status_is_written_before_completion(store, cfg):
    # A caller polling mid-run must see 'running', so the runner has to publish
    # that state before waiting on the child.
    seen = []
    original = store.write_status

    def spy(status):
        seen.append(status.state)
        original(status)

    store.write_status = spy
    execute_job(_job(("doctor",)), cfg, store, Clock())
    assert JobState.RUNNING in seen
    assert seen[-1] is JobState.SUCCEEDED


def test_status_carries_start_and_end_timestamps(store, cfg):
    status = execute_job(_job(("doctor",)), cfg, store, Clock())
    assert status.started_at is not None
    assert status.ended_at is not None


def test_log_tail_is_embedded_in_the_final_status(store, cfg):
    status = execute_job(_job(("doctor",)), cfg, store, Clock())
    assert "stub ok" in status.log_tail


def test_whitelist_rejection_fails_without_launching_anything(store, cfg):
    status = execute_job(_job(("rm", "-rf", "/")), cfg, store, Clock())
    assert status.state is JobState.FAILED
    assert "whitelist" in status.reason.lower()
    assert status.exit_code is None


def test_timeout_kills_the_child_and_reports_timed_out(store, cfg):
    status = execute_job(_job(("run",), timeout_s=1), cfg, store, Clock())
    assert status.state is JobState.TIMED_OUT


def test_cancellation_stops_a_running_job(store, cfg):
    store.request_cancel("j1")
    status = execute_job(_job(("run",), timeout_s=600), cfg, store, Clock())
    assert status.state is JobState.CANCELLED


def test_cancel_sentinel_is_cleared_afterwards(store, cfg):
    store.request_cancel("j1")
    execute_job(_job(("run",), timeout_s=600), cfg, store, Clock())
    assert not store.is_cancel_requested("j1")


def test_expect_gpu_mismatch_fails_before_running(store, cfg, monkeypatch):
    monkeypatch.setattr(
        "codenames.remote.runner.gpu_info", lambda: ("Tesla T4", 100, 15360)
    )
    status = execute_job(_job(("doctor",), expect_gpu="A100"), cfg, store, Clock())
    assert status.state is JobState.FAILED
    assert "A100" in status.reason and "T4" in status.reason
    assert store.read_log("j1") == ""


def test_expect_gpu_match_is_case_insensitive_substring(store, cfg, monkeypatch):
    monkeypatch.setattr(
        "codenames.remote.runner.gpu_info",
        lambda: ("NVIDIA A100-SXM4-40GB", 39000, 40960),
    )
    status = execute_job(_job(("doctor",), expect_gpu="a100"), cfg, store, Clock())
    assert status.state is JobState.SUCCEEDED


def test_tail_text_keeps_only_the_last_lines():
    assert tail_text("a\nb\nc\nd\n", 2) == "c\nd"


def test_tail_text_handles_short_input():
    assert tail_text("a\n", 5) == "a"
