"""DriveApiStore against a fake Drive service — no network, no credentials."""

import pytest

from codenames.remote.drive import FOLDER_MIME, DriveApiStore, resolve_folder
from codenames.remote.protocol import Heartbeat, Job, JobState, JobStatus

NOW_ISO = "2026-07-27T19:30:00Z"


class _Exec:
    """Stand-in for a googleapiclient request object."""

    def __init__(self, payload):
        self.payload = payload

    def execute(self):
        return self.payload


class _Media:
    """Stand-in for a get_media() response; the store reads .content."""

    def __init__(self, content):
        self.content = content


class FakeDrive:
    """In-memory Drive v3 ``files()`` resource.

    Models only what the store uses: list by parent (optionally by name),
    create, update, delete, and media download. Folders are files carrying the
    Drive folder mime type.
    """

    def __init__(self):
        self.items = {
            "root": {
                "id": "root",
                "name": "root",
                "parents": [],
                "mimeType": FOLDER_MIME,
                "content": "",
            }
        }
        self._next = 0

    def _new_id(self) -> str:
        self._next += 1
        return f"id{self._next}"

    def add_folder(self, name, parent="root") -> str:
        fid = self._new_id()
        self.items[fid] = {
            "id": fid,
            "name": name,
            "parents": [parent],
            "mimeType": FOLDER_MIME,
            "content": "",
        }
        return fid

    # --- the surface the store calls -------------------------------------
    def files(self):
        return self

    @staticmethod
    def _parse_query(q):
        parent = name = None
        for clause in (q or "").split(" and "):
            clause = clause.strip()
            if clause.endswith("in parents"):
                parent = clause.split("'")[1]
            elif clause.startswith("name ="):
                name = clause.split("'")[1]
        return parent, name

    def list(self, q=None, fields=None, pageSize=None, pageToken=None, **kw):
        parent, name = self._parse_query(q)
        results = []
        for item in self.items.values():
            if item["id"] == "root":
                continue
            if parent is not None and parent not in item["parents"]:
                continue
            if name is not None and item["name"] != name:
                continue
            results.append(
                {
                    "id": item["id"],
                    "name": item["name"],
                    "mimeType": item["mimeType"],
                    "size": str(len(item["content"])),
                    "md5Checksum": f"md5-{item['content']}",
                }
            )
        return _Exec({"files": results})

    def create(self, body=None, media_body=None, fields=None, **kw):
        fid = self._new_id()
        content = getattr(media_body, "_content", "")
        self.items[fid] = {
            "id": fid,
            "name": body["name"],
            "parents": body.get("parents", []),
            "mimeType": body.get("mimeType", "text/plain"),
            "content": content,
        }
        return _Exec({"id": fid})

    def update(self, fileId=None, media_body=None, fields=None, **kw):
        self.items[fileId]["content"] = getattr(media_body, "_content", "")
        return _Exec({"id": fileId})

    def delete(self, fileId=None, **kw):
        self.items.pop(fileId, None)
        return _Exec({})

    def get_media(self, fileId=None, **kw):
        return _Media(self.items[fileId]["content"])


@pytest.fixture
def service():
    return FakeDrive()


@pytest.fixture
def store(service):
    root = service.add_folder("_jobs", parent="root")
    s = DriveApiStore(service, root)
    s.ensure_layout()
    return s


def _job(job_id="j1") -> Job:
    return Job(
        job_id=job_id,
        created_at=NOW_ISO,
        git_ref="probing",
        argv=("doctor",),
        timeout_s=600,
        expect_gpu=None,
    )


def test_ensure_layout_creates_the_six_subfolders(service, store):
    names = {
        it["name"] for it in service.items.values() if it["mimeType"] == FOLDER_MIME
    }
    assert {"queue", "claimed", "status", "logs", "runner", "control"} <= names


def test_ensure_layout_is_idempotent(service, store):
    before = len(service.items)
    store.ensure_layout()
    assert len(service.items) == before


def test_put_then_list_queued(store):
    store.put_job(_job())
    assert [j.job_id for j in store.list_queued()] == ["j1"]


def test_claim_removes_from_queue_and_adds_to_claimed(store):
    store.put_job(_job())
    assert store.claim("j1").job_id == "j1"
    assert store.list_queued() == []
    assert store.read_job("j1") is not None


def test_second_claim_returns_none(store):
    store.put_job(_job())
    store.claim("j1")
    assert store.claim("j1") is None


def test_status_round_trip(store):
    status = JobStatus(job_id="j1", state=JobState.RUNNING, updated_at=NOW_ISO)
    store.write_status(status)
    assert store.read_status("j1") == status


def test_write_status_twice_updates_rather_than_duplicates(service, store):
    store.write_status(
        JobStatus(job_id="j1", state=JobState.RUNNING, updated_at=NOW_ISO)
    )
    store.write_status(
        JobStatus(job_id="j1", state=JobState.SUCCEEDED, updated_at=NOW_ISO)
    )
    matching = [it for it in service.items.values() if it["name"] == "j1.json"]
    assert len(matching) == 1
    assert store.read_status("j1").state is JobState.SUCCEEDED


def test_list_statuses_is_newest_first(store):
    for jid in ("20260721", "20260723", "20260722"):
        store.write_status(
            JobStatus(job_id=jid, state=JobState.SUCCEEDED, updated_at=NOW_ISO)
        )
    assert [s.job_id for s in store.list_statuses()] == [
        "20260723",
        "20260722",
        "20260721",
    ]


def test_heartbeat_round_trip(store):
    hb = Heartbeat(runner_id="r1", updated_at=NOW_ISO, gpu_name="A100")
    store.write_heartbeat(hb)
    assert store.read_heartbeat() == hb


def test_read_heartbeat_is_none_when_absent(store):
    assert store.read_heartbeat() is None


def test_log_upload_and_read(store, tmp_path):
    local = tmp_path / "j1.log"
    local.write_text("hello\n", encoding="utf-8")
    store.put_log("j1", local)
    assert store.read_log("j1") == "hello\n"


def test_read_log_of_unknown_job_is_empty(store):
    assert store.read_log("nope") == ""


def test_cancel_lifecycle(store):
    assert not store.is_cancel_requested("j1")
    store.request_cancel("j1")
    assert store.is_cancel_requested("j1")
    store.clear_cancel("j1")
    assert not store.is_cancel_requested("j1")


def test_corrupt_status_document_reads_as_missing(service, store):
    store.write_status(
        JobStatus(job_id="j1", state=JobState.RUNNING, updated_at=NOW_ISO)
    )
    for item in service.items.values():
        if item["name"] == "j1.json":
            item["content"] = "{not json"
    assert store.read_status("j1") is None


def test_resolve_folder_walks_a_path(service):
    a = service.add_folder("Codenames-Research", parent="root")
    service.add_folder("_jobs", parent=a)
    assert resolve_folder(service, ["Codenames-Research", "_jobs"]) is not None


def test_resolve_folder_returns_none_for_a_missing_path(service):
    assert resolve_folder(service, ["Nope", "_jobs"]) is None


def test_resolve_folder_creates_when_asked(service):
    fid = resolve_folder(service, ["Codenames-Research", "_jobs"], create=True)
    assert fid is not None
    assert resolve_folder(service, ["Codenames-Research", "_jobs"]) == fid
