"""CLI wiring for the job verbs, driven through --jobs-dir (no Drive)."""

from datetime import datetime, timezone

import pytest

from codenames.cli import main
from codenames.remote.protocol import JobState, JobStatus, utc_now_iso
from codenames.remote.store import LocalDirStore

NOW = datetime(2026, 7, 27, 19, 30, 0, tzinfo=timezone.utc)


@pytest.fixture
def jobs_dir(tmp_path):
    d = tmp_path / "_jobs"
    LocalDirStore(d).ensure_layout()
    return d


def test_job_submit_enqueues_a_job(jobs_dir):
    rc = main(
        [
            "job-submit",
            "--jobs-dir",
            str(jobs_dir),
            "--git-ref",
            "probing",
            "--",
            "doctor",
            "--model",
            "mistral",
        ]
    )
    assert rc == 0
    assert len(LocalDirStore(jobs_dir).list_queued()) == 1


def test_job_submit_prints_the_job_id(jobs_dir, capsys):
    main(
        ["job-submit", "--jobs-dir", str(jobs_dir), "--git-ref", "probing", "--", "doctor"]
    )
    assert "doctor" in capsys.readouterr().out


def test_job_submit_rejects_a_non_whitelisted_subcommand(jobs_dir, capsys):
    rc = main(
        [
            "job-submit",
            "--jobs-dir",
            str(jobs_dir),
            "--git-ref",
            "probing",
            "--",
            "visualize",
        ]
    )
    assert rc == 1
    assert "whitelist" in capsys.readouterr().err.lower()


def test_job_submit_passes_expect_gpu_through(jobs_dir):
    main(
        [
            "job-submit",
            "--jobs-dir",
            str(jobs_dir),
            "--git-ref",
            "probing",
            "--expect-gpu",
            "A100",
            "--",
            "run",
            "--model",
            "mistral",
        ]
    )
    assert LocalDirStore(jobs_dir).list_queued()[0].expect_gpu == "A100"


def test_job_status_reports_no_runner_when_heartbeat_is_absent(jobs_dir, capsys):
    rc = main(["job-status", "--jobs-dir", str(jobs_dir)])
    assert rc == 0
    assert "no runner" in capsys.readouterr().out.lower()


def test_job_status_lists_known_jobs(jobs_dir, capsys):
    LocalDirStore(jobs_dir).write_status(
        JobStatus(job_id="j1", state=JobState.SUCCEEDED, updated_at=utc_now_iso(NOW))
    )
    main(["job-status", "--jobs-dir", str(jobs_dir)])
    out = capsys.readouterr().out
    assert "j1" in out and "succeeded" in out


def test_job_logs_prints_the_log(jobs_dir, tmp_path, capsys):
    local = tmp_path / "x.log"
    local.write_text("hello world\n", encoding="utf-8")
    LocalDirStore(jobs_dir).put_log("j1", local)
    main(["job-logs", "--jobs-dir", str(jobs_dir), "--job", "j1"])
    assert "hello world" in capsys.readouterr().out


def test_job_logs_of_unknown_job_returns_nonzero(jobs_dir):
    assert main(["job-logs", "--jobs-dir", str(jobs_dir), "--job", "nope"]) == 1


def test_job_cancel_creates_the_sentinel(jobs_dir):
    store = LocalDirStore(jobs_dir)
    store.write_status(
        JobStatus(job_id="j1", state=JobState.RUNNING, updated_at=utc_now_iso(NOW))
    )
    assert main(["job-cancel", "--jobs-dir", str(jobs_dir), "--job", "j1"]) == 0
    assert store.is_cancel_requested("j1")


def test_job_cancel_of_finished_job_returns_nonzero(jobs_dir):
    store = LocalDirStore(jobs_dir)
    store.write_status(
        JobStatus(job_id="j1", state=JobState.SUCCEEDED, updated_at=utc_now_iso(NOW))
    )
    assert main(["job-cancel", "--jobs-dir", str(jobs_dir), "--job", "j1"]) == 1


def test_job_runner_starts_the_loop_and_exits_when_idle(jobs_dir, tmp_path):
    rc = main(
        [
            "job-runner",
            "--jobs-dir",
            str(jobs_dir),
            "--repo-dir",
            str(tmp_path),
            "--log-dir",
            str(tmp_path / "logs"),
            "--idle-shutdown-minutes",
            "0",
            "--skip-git",
        ]
    )
    assert rc == 0
