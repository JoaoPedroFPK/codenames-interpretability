import pytest

from codenames.remote.prewarm import MODEL_REPOS, prewarm_models, repo_for


def test_every_causal_model_has_a_repo():
    for key in ("mistral", "qwen", "qwen_random"):
        assert repo_for(key)


def test_random_init_shares_the_qwen_repo():
    """The random-init decoder instantiates Qwen's architecture, so warming
    Qwen warms it too - no separate 15 GB download."""
    assert repo_for("qwen_random") == repo_for("qwen")


def test_unknown_model_is_a_clear_error():
    with pytest.raises(KeyError, match="unknown model"):
        repo_for("gpt5")


def test_prewarm_deduplicates_shared_repos(monkeypatch):
    """qwen and qwen_random share a repo; it must be fetched once, not twice."""
    calls = []

    def fake_snapshot(repo_id, **kwargs):
        calls.append(repo_id)
        return f"/cache/{repo_id}"

    monkeypatch.setattr("codenames.remote.prewarm._snapshot_download", fake_snapshot)
    out = prewarm_models(["qwen", "qwen_random"])
    assert calls == [MODEL_REPOS["qwen"]]
    assert set(out) == {"qwen", "qwen_random"}


def test_prewarm_reports_failures_without_raising(monkeypatch):
    """A warm failure must not abort the runner cell - the job can still fetch
    the weights itself, just more slowly."""
    def boom(repo_id, **kwargs):
        raise OSError("network down")

    monkeypatch.setattr("codenames.remote.prewarm._snapshot_download", boom)
    out = prewarm_models(["qwen"])
    assert out["qwen"].startswith("FAILED")


def test_prewarm_accepts_an_empty_selection(monkeypatch):
    monkeypatch.setattr("codenames.remote.prewarm._snapshot_download",
                        lambda repo_id, **kw: "/cache")
    assert prewarm_models([]) == {}
