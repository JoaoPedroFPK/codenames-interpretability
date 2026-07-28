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


# --- do not transfer weight formats the loader will never read -------------
#
# Mistral-7B-Instruct-v0.2 publishes a full .bin copy beside its safetensors:
# 29.5 GB in the repo for the 14.5 GB from_pretrained actually loads. Warming
# both decoders moved 44.7 GB instead of 29.7 GB, which is why the pre-warm
# cell ran long enough to look hung.

def test_duplicate_weight_formats_are_skipped(monkeypatch):
    monkeypatch.setattr("codenames.remote.prewarm._repo_files",
                        lambda repo: ["config.json",
                                      "model-00001-of-00003.safetensors",
                                      "pytorch_model-00001-of-00003.bin"])
    from codenames.remote.prewarm import ignore_patterns_for
    assert "*.bin" in ignore_patterns_for("any/repo")


def test_a_repo_with_only_bin_weights_is_fetched_whole(monkeypatch):
    """Skipping .bin there would warm a cache containing no weights at all,
    and the job would then race the download this module exists to avoid."""
    monkeypatch.setattr("codenames.remote.prewarm._repo_files",
                        lambda repo: ["config.json", "pytorch_model.bin"])
    from codenames.remote.prewarm import ignore_patterns_for
    assert ignore_patterns_for("any/repo") == []


def test_an_unreachable_listing_fetches_everything(monkeypatch):
    """A slow warm is a cheaper mistake than an incomplete one."""
    def boom(repo):
        raise OSError("rate limited")
    monkeypatch.setattr("codenames.remote.prewarm._repo_files", boom)
    from codenames.remote.prewarm import ignore_patterns_for
    assert ignore_patterns_for("any/repo") == []


def test_prewarm_passes_the_skip_list_to_the_download(monkeypatch):
    seen = {}

    def fake(repo_id, **kwargs):
        seen.update(kwargs)
        return "/cache"

    monkeypatch.setattr("codenames.remote.prewarm._repo_files",
                        lambda repo: ["model-00001-of-00002.safetensors",
                                      "pytorch_model.bin"])
    monkeypatch.setattr("codenames.remote.prewarm._snapshot_download", fake)
    prewarm_models(["mistral"])
    assert "*.bin" in seen["ignore_patterns"]
