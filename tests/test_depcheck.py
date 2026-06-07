"""Unit tests for the dependency doctor (codenames.depcheck).

These test the pure parsing/comparison/report logic without touching the
network or requiring the heavy ML dependencies to be installed.
"""

import importlib.metadata as ilm

import pytest

from codenames import depcheck


# --- requirement parsing ---------------------------------------------------

@pytest.mark.parametrize(
    "req, expected",
    [
        ("transformers==4.48.0", ("transformers", "==4.48.0")),
        ("torch", ("torch", None)),
        ("numpy>=1.23,<2.3", ("numpy", ">=1.23,<2.3")),
        ("accelerate[foo]==1.2.1", ("accelerate", "==1.2.1")),
        ("pandas==2.2.3 ; python_version >= '3.10'", ("pandas", "==2.2.3")),
    ],
)
def test_parse_requirement(req, expected):
    assert depcheck._parse_requirement(req) == expected


# --- version satisfaction --------------------------------------------------

def test_satisfies_exact_pin():
    assert depcheck._satisfies("4.48.0", "==4.48.0")
    assert not depcheck._satisfies("4.49.0", "==4.48.0")


def test_satisfies_range_when_packaging_available():
    # packaging is a transitive dependency of pip and effectively always present.
    assert depcheck._satisfies("2.1.3", ">=2.0,<3.0")
    assert not depcheck._satisfies("1.26.0", ">=2.0,<3.0")


# --- expected_dependencies reads installed metadata ------------------------

def test_expected_dependencies_uses_metadata(monkeypatch):
    monkeypatch.setattr(
        depcheck.ilm, "requires",
        lambda name: ["torch==2.5.1", "transformers==4.48.0",
                      "pytest==1.0 ; extra == 'dev'"],
    )
    deps = depcheck.expected_dependencies()
    # The extra-only requirement is filtered out.
    assert ("torch", "==2.5.1") in deps
    assert ("transformers", "==4.48.0") in deps
    assert all(name != "pytest" for name, _ in deps)


def test_expected_dependencies_missing_metadata(monkeypatch):
    def _raise(_name):
        raise ilm.PackageNotFoundError()
    monkeypatch.setattr(depcheck.ilm, "requires", _raise)
    assert depcheck.expected_dependencies() == []


# --- package check classifies drift / missing / ok -------------------------

def test_check_packages_classifies(monkeypatch):
    monkeypatch.setattr(
        depcheck, "expected_dependencies",
        lambda: [("torch", "==2.5.1"), ("numpy", "==2.1.3"), ("ghost", "==9.9")],
    )

    def fake_version(name):
        table = {"torch": "2.5.1", "numpy": "2.0.0"}
        if name not in table:
            raise ilm.PackageNotFoundError()
        return table[name]

    monkeypatch.setattr(depcheck.ilm, "version", fake_version)

    by_name = {r.name: r for r in depcheck.check_packages()}
    assert by_name["torch"].status == "ok"
    assert by_name["numpy"].status == "drift"
    assert by_name["ghost"].status == "missing"


# --- report hard-failure aggregation ---------------------------------------

def _report(**overrides):
    base = dict(
        packages=[depcheck.DepResult("torch", "==2.5.1", "2.5.1", "ok")],
        symbols=[depcheck.SymbolResult("bert", [])],
        python_version="3.12.0",
        python_ok=True,
        cuda=True,
        cuda_detail="cuda ok",
        flash_attn=False,
        metadata_found=True,
        allow_drift=False,
        require_cuda=False,
    )
    base.update(overrides)
    return depcheck.DoctorReport(**base)


def test_report_ok_when_all_pass():
    assert _report().ok


def test_report_fails_on_drift_unless_allowed():
    drifted = [depcheck.DepResult("numpy", "==2.1.3", "2.0.0", "drift")]
    assert not _report(packages=drifted).ok
    assert _report(packages=drifted, allow_drift=True).ok


def test_report_fails_on_missing_symbol():
    bad = [depcheck.SymbolResult("modernbert", ["ModernBertModel"])]
    assert not _report(symbols=bad).ok


def test_report_require_cuda():
    assert not _report(cuda=False, require_cuda=True).ok
    assert _report(cuda=False, require_cuda=False).ok


def test_report_missing_metadata_fails():
    assert not _report(metadata_found=False).ok


# --- end-to-end CLI invocation --------------------------------------------

def test_cli_doctor_runs_and_returns_int(capsys):
    """`codenames-experiment doctor` dispatches and prints a report without raising.

    Exercises the real CLI -> depcheck path against the live interpreter. The
    return code depends on whether the package is pip-installed in this env
    (metadata present), so we only assert it is a valid process exit code and
    that the report header was printed.
    """
    from codenames.cli import main

    rc = main(["doctor", "--allow-drift"])
    out = capsys.readouterr().out
    assert rc in (0, 1)
    assert "doctor — dependency verification" in out
    assert "Pinned dependencies" in out


def test_cli_doctor_single_model(capsys):
    from codenames.cli import main

    rc = main(["doctor", "--model", "bert", "--allow-drift"])
    out = capsys.readouterr().out
    assert rc in (0, 1)
    # Only the requested model's class block should appear.
    assert "bert " in out
    assert "modernbert" not in out
