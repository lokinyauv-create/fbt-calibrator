"""Updates from GitHub: versions are listed newest first, drafts/pre-releases skipped, newer is detected."""
import io
import json

from fbtcal import update


def test_versions_are_listed_newest_first_and_newer_is_detected(monkeypatch):
    data = [{"tag_name": "v1.2.1", "published_at": "2026-10-28T16:18:28Z", "body": "a"},
            {"tag_name": "v1.10.0", "published_at": "2026-12-01T00:00:00Z", "body": "b"},
            {"tag_name": "v2.0.0-rc1", "prerelease": True},
            {"tag_name": "v1.3.0", "published_at": "2026-11-14T00:00:00Z", "body": "c"}]
    monkeypatch.setattr(update, "_get", lambda url: io.BytesIO(json.dumps(data).encode()))
    rels = update.releases()
    assert [r["version"] for r in rels] == ["1.10.0", "1.3.0", "1.2.1"]      # 1.10 > 1.3 (not text order)
    monkeypatch.setattr(update, "VERSION", "1.3.0")
    assert update.newest_if_newer(rels)["tag"] == "v1.10.0"
    monkeypatch.setattr(update, "VERSION", "1.10.0")
    assert update.newest_if_newer(rels) is None
