"""Tests must never touch her real files: every path the app writes to points into a temp folder.
(2026-11-13: a test saved its fake calibration over the real one, and the app loaded it on restart.)"""
import pytest

from fbtcal import main as m


@pytest.fixture(autouse=True)
def private_files(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "CONFIG", tmp_path)
    monkeypatch.setattr(m, "CAL_FILE", tmp_path / "calibration.json")
    monkeypatch.setattr(m, "SETTINGS_FILE", tmp_path / "settings.json")
    for name in ("CACHE", "STATUS_FILE", "COMMANDS", "LOG_FILE"):
        if hasattr(m, name):
            monkeypatch.setattr(m, name, tmp_path / name.lower())
    monkeypatch.setattr(m, "play", lambda *_: None)
    yield
