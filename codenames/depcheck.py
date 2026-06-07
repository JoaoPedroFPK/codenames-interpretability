"""Dependency verification ("doctor").

Read-only check that the *installed* environment satisfies the package's
pinned dependencies and exposes the ``transformers`` model classes each loader
needs. It mutates nothing and downloads no model weights, so it is safe to run
as the first cell after ``pip install`` in Colab to gain confidence before a
long run.

Why this exists: the package pins exact dependency versions in
``pyproject.toml`` for reproducibility, but Colab ships its own pre-installed
``torch``/``transformers``. This check makes any drift between the pinned and
installed versions loud and immediate instead of letting it surface as a
cryptic failure deep inside a model load or — worse — as silent numerical
drift in the results.

The expected versions are read from the installed package metadata
(``importlib.metadata``), so ``pyproject.toml`` stays the single source of
truth; this module never hard-codes a version number.
"""

from __future__ import annotations

import importlib
import importlib.metadata as ilm
import platform
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

DIST_NAME = "codenames-interpretability"

# Minimum Python the package supports (mirrors requires-python in pyproject).
MIN_PYTHON: Tuple[int, int] = (3, 10)

# transformers top-level symbols each model loader depends on. Importability
# of these is the version-sensitive signal: e.g. ``ModernBertModel`` only
# exists in transformers>=4.48.0, so a too-old transformers fails here loudly
# instead of inside ``load_modernbert``. Keep this in sync with the
# ``from transformers import ...`` lines in codenames/models/*.py.
MODEL_TRANSFORMERS_SYMBOLS: Dict[str, List[str]] = {
    "mistral":     ["AutoModelForCausalLM", "AutoTokenizer", "MistralForCausalLM"],
    "qwen":        ["AutoModelForCausalLM", "AutoTokenizer", "Qwen2ForCausalLM"],
    "qwen_random": ["AutoConfig", "AutoModelForCausalLM", "AutoTokenizer", "Qwen2ForCausalLM"],
    "bert":        ["BertModel", "BertTokenizerFast"],
    "bert_random": ["BertConfig", "BertModel", "BertTokenizerFast"],
    "t5":          ["T5EncoderModel", "T5TokenizerFast"],
    "modernbert":  ["AutoModel", "AutoTokenizer", "ModernBertModel"],
}

# Models whose loaders probe for optional flash_attn (FA2). Informational only:
# the loaders fall back to eager attention when flash_attn is absent.
FLASH_ATTN_MODELS = ("mistral", "qwen", "modernbert")

# Matches "name", "name==1.2.3", "name>=1.0,<2.0", etc. Captures name + spec.
_REQ_RE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*(.*)$")


# ---------------------------------------------------------------------------
# Pin parsing / comparison
# ---------------------------------------------------------------------------

def _parse_requirement(req: str) -> Tuple[str, Optional[str]]:
    """Split a requirement string into (dist_name, version_spec_or_None).

    Strips environment markers and extras. ``"transformers==4.48.0"`` ->
    ``("transformers", "==4.48.0")``; ``"torch"`` -> ``("torch", None)``.
    """
    base = req.split(";", 1)[0].strip()
    # Drop any extras bracket, e.g. "accelerate[foo]==1.0" -> "accelerate==1.0".
    base = re.sub(r"\[[^\]]*\]", "", base)
    m = _REQ_RE.match(base)
    if not m:
        return base, None
    name = m.group(1)
    spec = m.group(2).strip() or None
    return name, spec


def expected_dependencies() -> List[Tuple[str, Optional[str]]]:
    """Return [(dist_name, version_spec)] from the installed package metadata.

    Returns an empty list if the package is not pip-installed (metadata
    unavailable) — the caller reports that as its own failure mode.
    """
    try:
        reqs = ilm.requires(DIST_NAME) or []
    except ilm.PackageNotFoundError:
        return []
    out: List[Tuple[str, Optional[str]]] = []
    for r in reqs:
        # Skip optional-extra requirements (they carry an 'extra ==' marker).
        marker = r.split(";", 1)[1] if ";" in r else ""
        if "extra" in marker:
            continue
        out.append(_parse_requirement(r))
    return out


def _satisfies(installed: str, spec: str) -> bool:
    """Whether the installed version satisfies the spec.

    Uses ``packaging`` when available (correct PEP 440 semantics); falls back
    to a literal equality check for the exact ``==`` pins this project uses.
    """
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version

        return Version(installed) in SpecifierSet(spec)
    except Exception:
        s = spec.strip()
        if s.startswith("=="):
            return installed == s[2:].strip()
        # Unknown operator and no packaging available: don't raise a false
        # alarm — report as satisfied and let the explicit version printout
        # speak for itself.
        return True


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

@dataclass
class DepResult:
    name: str
    expected: Optional[str]
    installed: Optional[str]
    status: str  # "ok" | "drift" | "missing" | "unpinned"


def check_packages() -> List[DepResult]:
    """Compare every declared dependency against what is installed."""
    results: List[DepResult] = []
    for name, spec in expected_dependencies():
        try:
            inst = ilm.version(name)
        except ilm.PackageNotFoundError:
            results.append(DepResult(name, spec, None, "missing"))
            continue
        if spec is None:
            results.append(DepResult(name, None, inst, "unpinned"))
        elif _satisfies(inst, spec):
            results.append(DepResult(name, spec, inst, "ok"))
        else:
            results.append(DepResult(name, spec, inst, "drift"))
    return results


@dataclass
class SymbolResult:
    model: str
    missing: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing


def check_model_symbols(models: Optional[List[str]] = None) -> List[SymbolResult]:
    """Check that the installed transformers exposes each model's classes.

    Importing the symbols (rather than calling ``from_pretrained``) keeps this
    offline and weight-free while still catching a transformers too old to
    know about the architecture.
    """
    targets = models or list(MODEL_TRANSFORMERS_SYMBOLS)
    try:
        transformers = importlib.import_module("transformers")
    except Exception as exc:  # transformers itself missing/broken
        return [SymbolResult(m, [f"transformers import failed: {exc}"]) for m in targets]

    out: List[SymbolResult] = []
    for m in targets:
        wanted = MODEL_TRANSFORMERS_SYMBOLS.get(m, [])
        missing = [s for s in wanted if not hasattr(transformers, s)]
        out.append(SymbolResult(m, missing))
    return out


def python_ok() -> Tuple[bool, str]:
    v = sys.version_info
    return (v[:2] >= MIN_PYTHON), platform.python_version()


def torch_cuda_status() -> Tuple[Optional[bool], str]:
    """(cuda_available_or_None, detail). None means torch failed to import."""
    try:
        import torch
    except Exception as exc:
        return None, f"torch import failed: {exc}"
    try:
        avail = bool(torch.cuda.is_available())
        dev = torch.cuda.get_device_name(0) if avail else "cpu only"
        return avail, f"torch {torch.__version__}, CUDA available={avail} ({dev})"
    except Exception as exc:
        return False, f"torch {getattr(torch, '__version__', '?')}, CUDA probe failed: {exc}"


def flash_attn_available() -> bool:
    try:
        importlib.import_module("flash_attn")
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Orchestration / reporting
# ---------------------------------------------------------------------------

@dataclass
class DoctorReport:
    packages: List[DepResult]
    symbols: List[SymbolResult]
    python_version: str
    python_ok: bool
    cuda: Optional[bool]
    cuda_detail: str
    flash_attn: bool
    metadata_found: bool
    allow_drift: bool
    require_cuda: bool

    @property
    def hard_failures(self) -> List[str]:
        fails: List[str] = []
        if not self.metadata_found:
            fails.append(
                f"package metadata for '{DIST_NAME}' not found — run `pip install -e .` first"
            )
        if not self.python_ok:
            fails.append(
                f"Python {self.python_version} < required {MIN_PYTHON[0]}.{MIN_PYTHON[1]}"
            )
        for p in self.packages:
            if p.status == "missing":
                fails.append(f"missing dependency: {p.name} (expected {p.expected})")
            elif p.status == "drift" and not self.allow_drift:
                fails.append(
                    f"version drift: {p.name} installed {p.installed}, pinned {p.expected}"
                )
        for s in self.symbols:
            if not s.ok:
                fails.append(
                    f"transformers missing {s.model} class(es): {', '.join(s.missing)}"
                )
        if self.require_cuda and self.cuda is not True:
            fails.append(f"CUDA required but unavailable ({self.cuda_detail})")
        return fails

    @property
    def ok(self) -> bool:
        return not self.hard_failures


def build_report(
    models: Optional[List[str]] = None,
    allow_drift: bool = False,
    require_cuda: bool = False,
) -> DoctorReport:
    expected = expected_dependencies()
    py_ok, py_ver = python_ok()
    cuda, cuda_detail = torch_cuda_status()
    return DoctorReport(
        packages=check_packages(),
        symbols=check_model_symbols(models),
        python_version=py_ver,
        python_ok=py_ok,
        cuda=cuda,
        cuda_detail=cuda_detail,
        flash_attn=flash_attn_available(),
        metadata_found=bool(expected),
        allow_drift=allow_drift,
        require_cuda=require_cuda,
    )


def _fmt_pkg(p: DepResult) -> str:
    glyph = {"ok": "OK  ", "unpinned": "WARN", "drift": "FAIL", "missing": "FAIL"}[p.status]
    if p.status == "missing":
        return f"  [{glyph}] {p.name:<14} not installed (pinned {p.expected})"
    if p.status == "unpinned":
        return f"  [{glyph}] {p.name:<14} {p.installed} (no pin declared)"
    if p.status == "drift":
        return f"  [{glyph}] {p.name:<14} {p.installed}  != pinned {p.expected}"
    return f"  [{glyph}] {p.name:<14} {p.installed} (== {p.expected})"


def run(
    models: Optional[List[str]] = None,
    allow_drift: bool = False,
    require_cuda: bool = False,
) -> bool:
    """Print a human-readable dependency report. Return True if all hard checks pass."""
    report = build_report(models=models, allow_drift=allow_drift, require_cuda=require_cuda)

    print("=" * 70)
    print("codenames-experiment doctor — dependency verification")
    print("=" * 70)

    pyglyph = "OK  " if report.python_ok else "FAIL"
    print(f"  [{pyglyph}] Python         {report.python_version} (require >= "
          f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]})")

    if not report.metadata_found:
        print("  [FAIL] package metadata not found — is `pip install -e .` done?")

    print("\nPinned dependencies (installed vs pyproject):")
    for p in report.packages:
        print(_fmt_pkg(p))

    scope = ", ".join(models) if models else "all models"
    print(f"\ntransformers model classes ({scope}):")
    for s in report.symbols:
        if s.ok:
            print(f"  [OK  ] {s.model:<14} all classes importable")
        else:
            print(f"  [FAIL] {s.model:<14} missing: {', '.join(s.missing)}")

    print("\nRuntime:")
    cglyph = "OK  " if report.cuda else ("FAIL" if report.require_cuda else "WARN")
    print(f"  [{cglyph}] {report.cuda_detail}")
    faglyph = "OK  " if report.flash_attn else "INFO"
    fa_note = "importable" if report.flash_attn else "absent (loaders fall back to eager)"
    print(f"  [{faglyph}] flash_attn     {fa_note}")

    print("-" * 70)
    if report.ok:
        print("RESULT: PASS — environment matches the pinned, reproducible set.")
    else:
        print("RESULT: FAIL")
        for f in report.hard_failures:
            print(f"  - {f}")
        print("\nFix: re-run `pip install -e .` (it enforces the pinned versions). "
              "If Colab refuses to change a pre-installed package, restart the "
              "runtime and run the setup cells again.")
    print("=" * 70)
    return report.ok
