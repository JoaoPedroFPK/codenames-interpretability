"""Pre-fetch model weights into the Colab session cache before claiming jobs.

A job that starts against a cold cache races a ~15 GB download inside its own
timeout, and a session drop mid-download strands the job with nothing measured
(observed twice on Qwen: the log stopped at "Downloading shards: 0%" and the
heartbeat went stale). Warming the cache first turns that download into a
separate, restartable step that costs the job nothing.

``snapshot_download`` is used rather than ``from_pretrained`` deliberately: it
fetches files without instantiating the model, so it needs no GPU memory and no
15 GB of host RAM, and it resumes rather than restarting after an interruption.
"""

from typing import Dict, Iterable, List

# Registry key -> HF repo. The random-init decoder instantiates Qwen's
# architecture from config, so warming Qwen covers it as well.
MODEL_REPOS: Dict[str, str] = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "qwen_random": "Qwen/Qwen2.5-7B-Instruct",
}


def repo_for(model_key: str) -> str:
    """HF repo id backing a registry key."""
    try:
        return MODEL_REPOS[model_key]
    except KeyError:
        raise KeyError(
            f"unknown model {model_key!r}; expected one of {sorted(MODEL_REPOS)}"
        ) from None


# Weight formats ``from_pretrained`` will not load when safetensors are
# present. Mistral-7B-Instruct-v0.2 ships a full ``.bin`` copy alongside its
# safetensors, so a bare snapshot of that repo transfers 29.5 GB to obtain the
# 14.5 GB the loader actually reads.
_REDUNDANT_WEIGHTS = ("*.bin", "*.pth", "*.h5", "*.msgpack", "consolidated*")


def _repo_files(repo_id: str) -> List[str]:
    """Filenames in the repo; empty when the listing cannot be fetched."""
    from huggingface_hub import HfApi

    return [s.rfilename for s in HfApi().model_info(repo_id).siblings]


def ignore_patterns_for(repo_id: str) -> List[str]:
    """Glob patterns to skip, or ``[]`` when everything is needed.

    Duplicate weight formats are skipped only after confirming the repo really
    does publish sharded safetensors. A repo that ships ``.bin`` alone would
    otherwise be warmed into a cache with no weights in it, and the job would
    then download them itself -- the exact failure this module exists to
    prevent. When the listing cannot be fetched, nothing is skipped: a slower
    warm is a far cheaper mistake than an incomplete one.
    """
    try:
        files = _repo_files(repo_id)
    except Exception:  # noqa: BLE001 - offline/rate-limited; fetch everything
        return []
    has_sharded_safetensors = any(
        f.endswith(".safetensors") and not f.startswith("consolidated")
        for f in files
    )
    return list(_REDUNDANT_WEIGHTS) if has_sharded_safetensors else []


def _snapshot_download(repo_id: str, **kwargs):
    """Indirection so tests can substitute the network call."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id, **kwargs)


def prewarm_models(model_keys: Iterable[str], *, quiet: bool = False) -> Dict[str, str]:
    """Fetch each model's weights into the local cache.

    Returns ``{model_key: cache_path or "FAILED: ..."}``. A failure is reported
    rather than raised: warming is an optimisation, and the job can still fetch
    the weights itself, just more slowly and with more exposure to a drop.
    """
    keys: List[str] = [k for k in model_keys]
    results: Dict[str, str] = {}
    seen: Dict[str, str] = {}

    for key in keys:
        repo = repo_for(key)
        if repo in seen:
            results[key] = seen[repo]
            if not quiet:
                print(f"  {key}: already warmed via {repo}")
            continue
        try:
            skip = ignore_patterns_for(repo)
            if not quiet:
                note = " (skipping duplicate weight formats)" if skip else ""
                print(f"  {key}: fetching {repo}{note} ...")
            # max_workers raises shard throughput on Colab, where the bottleneck
            # is per-connection rather than total bandwidth.
            path = _snapshot_download(repo, ignore_patterns=skip, max_workers=8)
            seen[repo] = str(path)
            results[key] = str(path)
            if not quiet:
                print(f"  {key}: cached at {path}")
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            message = f"FAILED: {type(exc).__name__}: {exc}"
            results[key] = message
            if not quiet:
                print(f"  {key}: {message}")
    return results
