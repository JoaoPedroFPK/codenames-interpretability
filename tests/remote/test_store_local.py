"""LocalDirStore behaviour: claiming, status, logs, cancellation, heartbeat."""

from codenames.remote.protocol import Heartbeat, Job, JobState, JobStatus
from codenames.remote.store import LocalDirStore, atomic_write_text

NOW_ISO = "2026-07-27T19:30:00Z"


def _store(tmp_path) -> LocalDirStore:
    store = LocalDirStore(tmp_path / "_jobs")
    store.ensure_layout()
    return store


def _job(job_id="j1", argv=("doctor",)) -> Job:
    return Job(
        job_id=job_id,
        created_at=NOW_ISO,
        git_ref="probing",
        argv=argv,
        timeout_s=600,
        expect_gpu=None,
    )


def test_ensure_layout_creates_every_directory(tmp_path):
    store = _store(tmp_path)
    for d in store.paths.all_dirs():
        assert d.is_dir()


def test_put_then_list_queued_returns_the_job(tmp_path):
    store = _store(tmp_path)
    store.put_job(_job())
    assert [j.job_id for j in store.list_queued()] == ["j1"]


def test_list_queued_is_sorted_by_job_id(tmp_path):
    store = _store(tmp_path)
    store.put_job(_job(job_id="20260727T200000Z-b"))
    store.put_job(_job(job_id="20260727T190000Z-a"))
    assert [j.job_id for j in store.list_queued()] == [
        "20260727T190000Z-a",
        "20260727T200000Z-b",
    ]


def test_claim_moves_the_document_out_of_the_queue(tmp_path):
    store = _store(tmp_path)
    store.put_job(_job())
    claimed = store.claim("j1")
    assert claimed is not None and claimed.job_id == "j1"
    assert store.list_queued() == []
    assert store.paths.claimed_file("j1").exists()


def test_second_claim_of_the_same_job_returns_none(tmp_path):
    store = _store(tmp_path)
    store.put_job(_job())
    assert store.claim("j1") is not None
    assert store.claim("j1") is None


def test_read_job_finds_it_in_queue_or_claimed(tmp_path):
    store = _store(tmp_path)
    store.put_job(_job())
    assert store.read_job("j1").argv == ("doctor",)
    store.claim("j1")
    assert store.read_job("j1").argv == ("doctor",)


def test_read_job_returns_none_when_absent(tmp_path):
    assert _store(tmp_path).read_job("nope") is None


def test_status_round_trips_through_the_store(tmp_path):
    store = _store(tmp_path)
    status = JobStatus(job_id="j1", state=JobState.RUNNING, updated_at=NOW_ISO)
    store.write_status(status)
    assert store.read_status("j1") == status


def test_list_statuses_is_newest_first_and_respects_limit(tmp_path):
    store = _store(tmp_path)
    for i in range(5):
        store.write_status(
            JobStatus(job_id=f"2026072{i}", state=JobState.SUCCEEDED, updated_at=NOW_ISO)
        )
    ids = [s.job_id for s in store.list_statuses(limit=3)]
    assert ids == ["20260724", "20260723", "20260722"]


def test_put_log_copies_the_local_file(tmp_path):
    store = _store(tmp_path)
    local = tmp_path / "local.log"
    local.write_text("board 1/10\nboard 2/10\n", encoding="utf-8")
    store.put_log("j1", local)
    assert store.read_log("j1") == "board 1/10\nboard 2/10\n"


def test_put_log_overwrites_rather_than_appends(tmp_path):
    store = _store(tmp_path)
    local = tmp_path / "local.log"
    local.write_text("first\n", encoding="utf-8")
    store.put_log("j1", local)
    local.write_text("first\nsecond\n", encoding="utf-8")
    store.put_log("j1", local)
    assert store.read_log("j1") == "first\nsecond\n"


def test_read_log_of_unknown_job_is_empty(tmp_path):
    assert _store(tmp_path).read_log("nope") == ""


def test_cancel_request_lifecycle(tmp_path):
    store = _store(tmp_path)
    assert not store.is_cancel_requested("j1")
    store.request_cancel("j1")
    assert store.is_cancel_requested("j1")
    store.clear_cancel("j1")
    assert not store.is_cancel_requested("j1")


def test_heartbeat_round_trips(tmp_path):
    store = _store(tmp_path)
    assert store.read_heartbeat() is None
    hb = Heartbeat(runner_id="r1", updated_at=NOW_ISO, gpu_name="A100")
    store.write_heartbeat(hb)
    assert store.read_heartbeat() == hb


def test_atomic_write_leaves_no_temp_files(tmp_path):
    target = tmp_path / "x.json"
    atomic_write_text(target, "{}")
    assert target.read_text(encoding="utf-8") == "{}"
    assert list(tmp_path.iterdir()) == [target]


def test_corrupt_status_file_is_reported_as_missing(tmp_path):
    # A half-written file on a FUSE mount must not crash the client.
    store = _store(tmp_path)
    store.paths.status_file("j1").write_text("{not json", encoding="utf-8")
    assert store.read_status("j1") is None
