"""Credential loading: scope widening must trigger consent, not a doomed refresh.

Regression: ``Credentials.from_authorized_user_file(path, scopes)`` reports the
scopes you *asked for*, not the ones the cached token actually holds. Comparing
against that value is comparing the request to itself, so a narrower cached
token looked sufficient and the code attempted a refresh that Google rejects
with ``invalid_scope``.
"""

import json

import pytest

from codenames.remote import drive

FULL = ["https://www.googleapis.com/auth/drive"]
READONLY = "https://www.googleapis.com/auth/drive.readonly"


FUTURE = "2099-01-01T00:00:00Z"
PAST = "2000-01-01T00:00:00Z"


def _write_token(path, scopes, expiry=FUTURE):
    # google-auth treats a token with no expiry as already expired, so every
    # fixture states one explicitly rather than relying on the default.
    path.write_text(
        json.dumps(
            {
                "token": "t",
                "refresh_token": "r",
                "client_id": "c",
                "client_secret": "s",
                "token_uri": "https://oauth2.googleapis.com/token",
                "scopes": scopes,
                "expiry": expiry,
            }
        ),
        encoding="utf-8",
    )


class _FakeCreds:
    valid = True
    expired = False
    refresh_token = "r"

    def __init__(self, scopes):
        self.scopes = scopes

    def to_json(self):
        return json.dumps({"token": "new", "scopes": self.scopes})


class _FakeFlow:
    created = []

    @classmethod
    def from_client_secrets_file(cls, secret_path, scopes):
        cls.created.append(scopes)
        return cls()

    def run_local_server(self, port=0):
        return _FakeCreds(FULL)


@pytest.fixture
def no_consent_needed(monkeypatch):
    """Make the consent flow observable without opening a browser."""
    _FakeFlow.created = []
    import google_auth_oauthlib.flow

    monkeypatch.setattr(google_auth_oauthlib.flow, "InstalledAppFlow", _FakeFlow)
    return _FakeFlow


def test_cached_scopes_reads_what_the_token_actually_holds(tmp_path):
    token = tmp_path / "token.json"
    _write_token(token, [READONLY])
    assert drive.cached_scopes(token) == {READONLY}


def test_cached_scopes_of_a_missing_file_is_empty(tmp_path):
    assert drive.cached_scopes(tmp_path / "absent.json") == set()


def test_cached_scopes_of_corrupt_json_is_empty(tmp_path):
    token = tmp_path / "token.json"
    token.write_text("{not json", encoding="utf-8")
    assert drive.cached_scopes(token) == set()


def test_narrower_cached_token_triggers_consent_not_refresh(tmp_path, no_consent_needed):
    token = tmp_path / "token.json"
    _write_token(token, [READONLY])
    creds = drive.load_credentials(token, tmp_path / "client_secret.json", scopes=FULL)
    assert no_consent_needed.created == [FULL]
    assert creds.scopes == FULL


def test_consent_rewrites_the_token_cache(tmp_path, no_consent_needed):
    token = tmp_path / "token.json"
    _write_token(token, [READONLY])
    drive.load_credentials(token, tmp_path / "client_secret.json", scopes=FULL)
    assert json.loads(token.read_text(encoding="utf-8"))["scopes"] == FULL


def test_absent_token_triggers_consent(tmp_path, no_consent_needed):
    drive.load_credentials(
        tmp_path / "absent.json", tmp_path / "client_secret.json", scopes=FULL
    )
    assert no_consent_needed.created == [FULL]


def test_sufficient_cached_token_is_reused_without_consent(tmp_path, no_consent_needed):
    token = tmp_path / "token.json"
    _write_token(token, FULL)
    creds = drive.load_credentials(token, tmp_path / "client_secret.json", scopes=FULL)
    assert no_consent_needed.created == []
    assert set(creds.scopes) == set(FULL)


def test_expired_token_with_sufficient_scope_is_refreshed_not_reconsented(
    tmp_path, no_consent_needed, monkeypatch
):
    import google.oauth2.credentials as gcreds

    refreshed = []

    def fake_refresh(self, request):
        refreshed.append(True)
        self.token = "refreshed"
        self.expiry = None  # google-auth treats None as "no expiry known"

    monkeypatch.setattr(gcreds.Credentials, "refresh", fake_refresh)
    token = tmp_path / "token.json"
    _write_token(token, FULL, expiry=PAST)

    drive.load_credentials(token, tmp_path / "client_secret.json", scopes=FULL)

    assert refreshed == [True]
    assert no_consent_needed.created == []


def test_refresh_failure_falls_back_to_consent(tmp_path, no_consent_needed, monkeypatch):
    import google.oauth2.credentials as gcreds
    from google.auth.exceptions import RefreshError

    def boom(self, request):
        raise RefreshError("revoked")

    monkeypatch.setattr(gcreds.Credentials, "refresh", boom)
    token = tmp_path / "token.json"
    _write_token(token, FULL, expiry=PAST)

    creds = drive.load_credentials(token, tmp_path / "client_secret.json", scopes=FULL)

    assert no_consent_needed.created == [FULL]
    assert creds.scopes == FULL
