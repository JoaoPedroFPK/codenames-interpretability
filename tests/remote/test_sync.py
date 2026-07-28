"""Retrieval: which files come down, which are skipped, and why."""

import hashlib

import pytest

from codenames.remote.sync import (
    RemoteFile,
    filter_files,
    is_heavy_artifact,
    needs_download,
    sync_model_outputs,
)


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _rf(name, content="x", file_id="id1") -> RemoteFile:
    return RemoteFile(file_id=file_id, name=name, size=len(content), md5=_md5(content))


def test_vector_subsample_npz_is_heavy():
    assert is_heavy_artifact("mistral_vectors_subsample_no_social_f16.npz")


def test_lens_hidden_dump_is_heavy():
    assert is_heavy_artifact("mistral_lens_hidden_no_social_f16.npy")


def test_summary_csv_is_not_heavy():
    assert not is_heavy_artifact("mistral_general_no_social.csv")


def test_metrics_parquet_is_not_heavy():
    assert not is_heavy_artifact("mistral_metrics_no_social.parquet")


def test_filter_skip_heavy_drops_only_heavy_files():
    files = [_rf("a_general.csv"), _rf("a_lens_hidden_no_social_f16.npy")]
    assert [f.name for f in filter_files(files, skip_heavy=True)] == ["a_general.csv"]


def test_filter_only_heavy_keeps_only_heavy_files():
    files = [_rf("a_general.csv"), _rf("a_lens_hidden_no_social_f16.npy")]
    assert [f.name for f in filter_files(files, only_heavy=True)] == [
        "a_lens_hidden_no_social_f16.npy"
    ]


def test_filter_rejects_both_flags_at_once():
    with pytest.raises(ValueError):
        filter_files([], skip_heavy=True, only_heavy=True)


def test_needs_download_true_when_local_is_absent(tmp_path):
    assert needs_download(_rf("a.csv"), tmp_path / "a.csv")


def test_needs_download_false_when_md5_matches(tmp_path):
    local = tmp_path / "a.csv"
    local.write_text("x", encoding="utf-8")
    assert not needs_download(_rf("a.csv", content="x"), local)


def test_needs_download_true_when_md5_differs(tmp_path):
    local = tmp_path / "a.csv"
    local.write_text("different", encoding="utf-8")
    assert needs_download(_rf("a.csv", content="x"), local)


def test_needs_download_falls_back_to_size_when_md5_is_absent(tmp_path):
    local = tmp_path / "a.csv"
    local.write_text("xy", encoding="utf-8")
    same = RemoteFile(file_id="i", name="a.csv", size=2, md5=None)
    bigger = RemoteFile(file_id="i", name="a.csv", size=99, md5=None)
    assert not needs_download(same, local)
    assert needs_download(bigger, local)


class FakeSyncService:
    """Minimal Drive stand-in for sync: one model folder holding two files."""

    def __init__(self):
        self.downloads = []

    def list_children(self, folder_id):
        return {
            "root": [
                {
                    "id": "f-mistral",
                    "name": "mistral_outputs",
                    "mimeType": "application/vnd.google-apps.folder",
                }
            ],
            "f-mistral": [
                {
                    "id": "a",
                    "name": "mistral_general_no_social.csv",
                    "mimeType": "text/csv",
                    "size": "1",
                    "md5Checksum": _md5("x"),
                },
                {
                    "id": "b",
                    "name": "mistral_lens_hidden_no_social_f16.npy",
                    "mimeType": "application/octet-stream",
                    "size": "1",
                    "md5Checksum": _md5("x"),
                },
            ],
        }[folder_id]


@pytest.fixture
def fake(monkeypatch):
    svc = FakeSyncService()

    def fake_list_folder(service, folder_id):
        return service.list_children(folder_id)

    def fake_download(service, file_id, dest):
        service.downloads.append(file_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("x", encoding="utf-8")

    monkeypatch.setattr("codenames.remote.sync.list_folder", fake_list_folder)
    monkeypatch.setattr("codenames.remote.sync.download_file", fake_download)
    return svc


def test_sync_downloads_both_files(fake, tmp_path):
    report = sync_model_outputs(fake, "root", tmp_path, ["mistral"])
    assert set(fake.downloads) == {"a", "b"}
    assert len(report.downloaded) == 2


def test_sync_writes_into_the_model_output_directory(fake, tmp_path):
    sync_model_outputs(fake, "root", tmp_path, ["mistral"])
    assert (tmp_path / "mistral_outputs" / "mistral_general_no_social.csv").exists()


def test_sync_skip_heavy_leaves_the_dump_behind(fake, tmp_path):
    sync_model_outputs(fake, "root", tmp_path, ["mistral"], skip_heavy=True)
    assert fake.downloads == ["a"]


def test_sync_second_run_skips_identical_files(fake, tmp_path):
    sync_model_outputs(fake, "root", tmp_path, ["mistral"])
    fake.downloads.clear()
    report = sync_model_outputs(fake, "root", tmp_path, ["mistral"])
    assert fake.downloads == []
    assert len(report.skipped) == 2


def test_dry_run_downloads_nothing_but_still_reports(fake, tmp_path):
    report = sync_model_outputs(fake, "root", tmp_path, ["mistral"], dry_run=True)
    assert fake.downloads == []
    assert len(report.downloaded) == 2


def test_unknown_model_folder_is_skipped_without_error(fake, tmp_path):
    report = sync_model_outputs(fake, "root", tmp_path, ["qwen"])
    assert report.downloaded == []
