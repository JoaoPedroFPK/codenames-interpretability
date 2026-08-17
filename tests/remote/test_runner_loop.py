"""run_agent_loop: claiming, heartbeat, idle shutdown, runner mutex."""

import sys
from datetime import datetime, timedelta, timezone

import pytest

from codenames.remote.protocol import Heartbeat, Job, JobState, utc_now_iso
from codenames.remote.runner import (
    RunnerConfig,
    another_runner_is_alive,
    run_agent_loop,
    write_heartbeat,
)
from codenames.remote.store import LocalDirStore

NOW = datetime(2026, 7, 27, 19, 30, 0, tzinfo=timezone.utc)


class Clock:
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
    stub = tmp_path / "stub_cli.py"
    stub.write_text("import sys\nprint('ok')\nsys.exit(0)\n", encoding="utf-8")
    return RunnerConfig(
        repo_dir=tmp_path / "repo",
        jobs_root=tmp_path / "_jobs",
        local_log_dir=tmp_path / "logs",
        runner_id="r-test",
        executable=f"{sys.executable} {stub}",
        skip_git=True,
        poll_interval_s=0.01,
        heartbeat_interval_s=0.0,
        log_flush_interval_s=0.01,
        idle_shutdown_minutes=0.0,
        grace_period_s=1.0,
    )


def _job(job_id="j1", argv=("doctor",)) -> Job:
    return Job(
        job_id=job_id,
        created_at="2026-07-27T19:30:00Z",
        git_ref="probing",
        argv=argv,
        timeout_s=600,
        expect_gpu=None,
    )


def test_write_heartbeat_records_identity_and_time(store, cfg):
    hb = write_heartbeat(cfg, store, Clock())
    assert hb.runner_id == "r-test"
    assert store.read_heartbeat().runner_id == "r-test"
    assert hb.updated_at.endswith("Z")


def test_write_heartbeat_carries_the_current_job(store, cfg):
    write_heartbeat(cfg, store, Clock(), current_job="j1")
    assert store.read_heartbeat().current_job == "j1"


def test_loop_runs_a_queued_job_to_completion(store, cfg):
    store.put_job(_job())
    run_agent_loop(cfg, store=store, now_fn=Clock(), max_iterations=5)
    assert store.read_status("j1").state is JobState.SUCCEEDED


def test_loop_claims_the_job_out_of_the_queue(store, cfg):
    store.put_job(_job())
    run_agent_loop(cfg, store=store, now_fn=Clock(), max_iterations=5)
    assert store.list_queued() == []


def test_loop_processes_jobs_in_id_order(store, cfg):
    store.put_job(_job(job_id="20260727T190000Z-a"))
    store.put_job(_job(job_id="20260727T200000Z-b"))
    run_agent_loop(cfg, store=store, now_fn=Clock(), max_iterations=10)
    statuses = {s.job_id: s.state for s in store.list_statuses()}
    assert statuses["20260727T190000Z-a"] is JobState.SUCCEEDED
    assert statuses["20260727T200000Z-b"] is JobState.SUCCEEDED


def test_loop_exits_cleanly_when_idle_budget_is_zero(store, cfg):
    # idle_shutdown_minutes=0.0 means: with nothing queued, stop immediately.
    assert run_agent_loop(cfg, store=store, now_fn=Clock(), max_iterations=50) == 0


def test_loop_writes_a_heartbeat_even_when_idle(store, cfg):
    run_agent_loop(cfg, store=store, now_fn=Clock(), max_iterations=3)
    assert store.read_heartbeat() is not None


def test_idle_timer_resets_after_a_job(store, cfg):
    # A job arriving before the idle budget expires must not be starved by a
    # timer that started when the loop did.
    cfg.idle_shutdown_minutes = 1.0
    store.put_job(_job())
    run_agent_loop(cfg, store=store, now_fn=Clock(), max_iterations=3)
    assert store.read_status("j1").state is JobState.SUCCEEDED


def test_another_runner_is_alive_detects_a_fresh_foreign_heartbeat(store, cfg):
    store.write_heartbeat(Heartbeat(runner_id="r-other", updated_at=utc_now_iso(NOW)))
    assert another_runner_is_alive(store, cfg, lambda: NOW + timedelta(seconds=10))


def test_our_own_heartbeat_is_not_another_runner(store, cfg):
    store.write_heartbeat(Heartbeat(runner_id="r-test", updated_at=utc_now_iso(NOW)))
    assert not another_runner_is_alive(store, cfg, lambda: NOW + timedelta(seconds=10))


def test_stale_foreign_heartbeat_is_not_alive(store, cfg):
    store.write_heartbeat(Heartbeat(runner_id="r-other", updated_at=utc_now_iso(NOW)))
    assert not another_runner_is_alive(store, cfg, lambda: NOW + timedelta(seconds=600))


def test_loop_refuses_to_start_beside_a_live_runner(store, cfg):
    store.put_job(_job())
    store.write_heartbeat(Heartbeat(runner_id="r-other", updated_at=utc_now_iso(NOW)))
    rc = run_agent_loop(
        cfg, store=store, now_fn=lambda: NOW + timedelta(seconds=5), max_iterations=5
    )
    assert rc == 2
    assert store.read_status("j1") is None


def test_loop_exports_hf_token_from_the_drive_secrets_file(store, cfg, tmp_path, monkeypatch):
    """Gated Hub models (Llama) need HF_TOKEN in the job's environment. The
    runner reads it from <jobs_root>/../_secrets/hf_token when the env has
    none, so a token uploaded once to Drive serves every session without
    touching Colab Secrets. Env wins over the file; a missing file is fine."""
    import os
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    secrets = tmp_path / "_secrets"
    secrets.mkdir()
    (secrets / "hf_token").write_text("hf_testtoken\n")
    run_agent_loop(cfg, store=store, now_fn=Clock(), max_iterations=1)
    assert os.environ.get("HF_TOKEN") == "hf_testtoken"
    assert os.environ.get("HUGGING_FACE_HUB_TOKEN") == "hf_testtoken"
    # an explicit environment token is never overwritten
    os.environ["HF_TOKEN"] = "hf_fromenv"
    run_agent_loop(cfg, store=store, now_fn=Clock(), max_iterations=1)
    assert os.environ["HF_TOKEN"] == "hf_fromenv"
