"""Protocol-level invariants: schemas, ids, paths, legal state transitions."""

from datetime import datetime, timezone

import pytest

from codenames.remote.protocol import (
    ALLOWED_SUBCOMMANDS,
    SCHEMA_VERSION,
    TERMINAL_STATES,
    Heartbeat,
    Job,
    JobPaths,
    JobState,
    JobStatus,
    is_legal_transition,
    make_job_id,
    validate_job,
)

NOW = datetime(2026, 7, 27, 19, 30, 0, tzinfo=timezone.utc)


def _job(**overrides):
    base = dict(
        job_id="20260727T193000Z-lens-extract-mistral",
        created_at="2026-07-27T19:30:00Z",
        git_ref="probing",
        argv=("lens-extract", "--model", "mistral", "--full"),
        timeout_s=21600,
        expect_gpu="A100",
    )
    base.update(overrides)
    return Job(**base)


def test_job_round_trips_through_dict():
    job = _job()
    assert Job.from_dict(job.to_dict()) == job


def test_job_dict_carries_schema_version():
    assert _job().to_dict()["schema_version"] == SCHEMA_VERSION


def test_job_argv_is_a_tuple_after_round_trip():
    # JSON gives back a list; the dataclass must normalise so equality holds.
    raw = _job().to_dict()
    assert isinstance(raw["argv"], list)
    assert isinstance(Job.from_dict(raw).argv, tuple)


def test_make_job_id_is_sortable_and_descriptive():
    job_id = make_job_id(["lens-extract", "--model", "mistral"], NOW)
    assert job_id == "20260727T193000Z-lens-extract-mistral"


def test_make_job_id_without_model_flag_uses_subcommand_only():
    assert make_job_id(["doctor"], NOW) == "20260727T193000Z-doctor"


def test_validate_job_accepts_a_whitelisted_subcommand():
    assert validate_job(_job()) is None


def test_validate_job_rejects_a_subcommand_outside_the_whitelist():
    reason = validate_job(_job(argv=("rm", "-rf", "/")))
    assert reason is not None
    assert "whitelist" in reason.lower()


def test_validate_job_rejects_empty_argv():
    assert validate_job(_job(argv=())) is not None


def test_validate_job_rejects_a_future_schema_version():
    reason = validate_job(_job(schema_version=SCHEMA_VERSION + 1))
    assert reason is not None
    assert "schema" in reason.lower()


def test_whitelist_covers_the_gpu_stages_and_nothing_destructive():
    assert {"run", "lens-extract", "lens-tune", "doctor"} <= ALLOWED_SUBCOMMANDS
    assert "visualize" not in ALLOWED_SUBCOMMANDS


def test_whitelist_covers_the_causal_gpu_stages():
    """The causal tier's GPU stages are dispatchable (causal_spec.md §12)."""
    assert {
        "causal-extract", "causal-scan", "causal-patch",
        "causal-steer", "causal-pilot",
    } <= ALLOWED_SUBCOMMANDS


def test_offline_causal_stage_is_not_dispatchable():
    """causal-analyze needs no GPU; it runs locally and stays off the queue.

    The whitelist is a security boundary on a personal Drive folder, so it
    admits only what genuinely needs the GPU session.
    """
    assert "causal-analyze" not in ALLOWED_SUBCOMMANDS


@pytest.mark.parametrize(
    "src,dst",
    [
        (JobState.QUEUED, JobState.CLAIMED),
        (JobState.CLAIMED, JobState.RUNNING),
        (JobState.RUNNING, JobState.SUCCEEDED),
        (JobState.RUNNING, JobState.FAILED),
        (JobState.RUNNING, JobState.CANCELLED),
        (JobState.RUNNING, JobState.TIMED_OUT),
        (JobState.CLAIMED, JobState.FAILED),
    ],
)
def test_legal_transitions(src, dst):
    assert is_legal_transition(src, dst)


@pytest.mark.parametrize(
    "src,dst",
    [
        (JobState.SUCCEEDED, JobState.RUNNING),
        (JobState.QUEUED, JobState.RUNNING),
        (JobState.FAILED, JobState.SUCCEEDED),
    ],
)
def test_illegal_transitions(src, dst):
    assert not is_legal_transition(src, dst)


def test_terminal_states_are_exactly_the_four_endings():
    assert TERMINAL_STATES == frozenset(
        {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED, JobState.TIMED_OUT}
    )


def test_status_round_trips_and_preserves_state_type():
    status = JobStatus(
        job_id="j1",
        state=JobState.RUNNING,
        updated_at="2026-07-27T19:31:00Z",
        started_at="2026-07-27T19:30:30Z",
        repo_sha="abc1234",
        gpu_name="NVIDIA A100-SXM4-40GB",
    )
    restored = JobStatus.from_dict(status.to_dict())
    assert restored == status
    assert isinstance(restored.state, JobState)


def test_heartbeat_round_trips():
    hb = Heartbeat(
        runner_id="r-0001",
        updated_at="2026-07-27T19:31:00Z",
        gpu_name="NVIDIA A100-SXM4-40GB",
        vram_free_mb=39000,
        vram_total_mb=40960,
        current_job="j1",
        repo_sha="abc1234",
        uptime_s=120.0,
        last_log_line="board 200/7703",
    )
    assert Heartbeat.from_dict(hb.to_dict()) == hb


def test_job_paths_are_all_under_the_root(tmp_path):
    paths = JobPaths(tmp_path / "_jobs")
    assert paths.job_file("j1") == tmp_path / "_jobs" / "queue" / "j1.json"
    assert paths.claimed_file("j1") == tmp_path / "_jobs" / "claimed" / "j1.json"
    assert paths.status_file("j1") == tmp_path / "_jobs" / "status" / "j1.json"
    assert paths.log_file("j1") == tmp_path / "_jobs" / "logs" / "j1.log"
    assert paths.cancel_file("j1") == tmp_path / "_jobs" / "control" / "cancel-j1"
    assert paths.heartbeat_file() == tmp_path / "_jobs" / "runner" / "heartbeat.json"


def test_job_paths_all_dirs_lists_every_directory(tmp_path):
    dirs = JobPaths(tmp_path / "_jobs").all_dirs()
    assert {d.name for d in dirs} == {
        "queue",
        "claimed",
        "status",
        "logs",
        "runner",
        "control",
    }
