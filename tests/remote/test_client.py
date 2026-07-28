"""Client-side operations, exercised entirely over LocalDirStore."""

from datetime import datetime, timedelta, timezone

import pytest

from codenames.remote import client
from codenames.remote.protocol import Heartbeat, JobState, JobStatus, utc_now_iso
from codenames.remote.store import LocalDirStore

NOW = datetime(2026, 7, 27, 19, 30, 0, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path):
    s = LocalDirStore(tmp_path / "_jobs")
    s.ensure_layout()
    return s


def test_submit_writes_a_queued_job(store):
    job = client.submit(
        store, ["doctor", "--model", "mistral"], git_ref="probing", now=NOW
    )
    assert [j.job_id for j in store.list_queued()] == [job.job_id]


def test_submit_derives_the_job_id_from_time_and_argv(store):
    job = client.submit(
        store, ["lens-extract", "--model", "qwen"], git_ref="probing", now=NOW
    )
    assert job.job_id == "20260727T193000Z-lens-extract-qwen"


def test_submit_records_the_git_ref_and_timeout(store):
    job = client.submit(store, ["doctor"], git_ref="main", timeout_s=99, now=NOW)
    assert job.git_ref == "main"
    assert job.timeout_s == 99


def test_submit_rejects_a_non_whitelisted_subcommand(store):
    with pytest.raises(ValueError, match="whitelist"):
        client.submit(store, ["rm", "-rf", "/"], git_ref="probing", now=NOW)


def test_submit_rejects_empty_argv(store):
    with pytest.raises(ValueError):
        client.submit(store, [], git_ref="probing", now=NOW)


def test_runner_health_reports_dead_when_no_heartbeat(store):
    health = client.runner_health(store, now=NOW)
    assert health.alive is False
    assert health.heartbeat is None


def test_runner_health_reports_alive_for_a_fresh_heartbeat(store):
    store.write_heartbeat(Heartbeat(runner_id="r1", updated_at=utc_now_iso(NOW)))
    health = client.runner_health(store, now=NOW + timedelta(seconds=30))
    assert health.alive is True
    assert health.age_s == pytest.approx(30.0)


def test_runner_health_reports_dead_past_the_staleness_threshold(store):
    store.write_heartbeat(Heartbeat(runner_id="r1", updated_at=utc_now_iso(NOW)))
    assert client.runner_health(store, now=NOW + timedelta(seconds=300)).alive is False


def test_logs_returns_the_stored_text(store, tmp_path):
    local = tmp_path / "x.log"
    local.write_text("line1\nline2\nline3\n", encoding="utf-8")
    store.put_log("j1", local)
    assert client.logs(store, "j1") == "line1\nline2\nline3\n"


def test_logs_tail_limits_the_line_count(store, tmp_path):
    local = tmp_path / "x.log"
    local.write_text("line1\nline2\nline3\n", encoding="utf-8")
    store.put_log("j1", local)
    assert client.logs(store, "j1", tail=2) == "line2\nline3"


def test_cancel_creates_the_sentinel_for_a_known_job(store):
    store.write_status(
        JobStatus(job_id="j1", state=JobState.RUNNING, updated_at=utc_now_iso(NOW))
    )
    assert client.cancel(store, "j1") is True
    assert store.is_cancel_requested("j1")


def test_cancel_of_a_finished_job_is_a_no_op(store):
    store.write_status(
        JobStatus(job_id="j1", state=JobState.SUCCEEDED, updated_at=utc_now_iso(NOW))
    )
    assert client.cancel(store, "j1") is False
    assert not store.is_cancel_requested("j1")


def test_format_status_line_mentions_id_and_state(store):
    line = client.format_status_line(
        JobStatus(job_id="j1", state=JobState.RUNNING, updated_at=utc_now_iso(NOW))
    )
    assert "j1" in line and "running" in line


def test_pending_jobs_counts_queued_and_unfinished(store):
    client.submit(store, ["doctor"], git_ref="probing", now=NOW)
    store.write_status(
        JobStatus(job_id="j-old", state=JobState.SUCCEEDED, updated_at=utc_now_iso(NOW))
    )
    assert client.has_unfinished_work(store) is True


def test_has_unfinished_work_is_false_when_everything_is_terminal(store):
    store.write_status(
        JobStatus(job_id="j1", state=JobState.SUCCEEDED, updated_at=utc_now_iso(NOW))
    )
    assert client.has_unfinished_work(store) is False


def test_keep_awake_is_a_no_op_off_darwin(monkeypatch):
    monkeypatch.setattr(client.sys, "platform", "linux")
    with client.KeepAwake() as ka:
        assert ka.process is None
