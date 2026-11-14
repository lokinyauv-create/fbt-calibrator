"""Each VRChat avatar keeps its own feet distance, picked from VRChat's log."""
import json

from fbtcal import main
from fbtcal.main import Calibrator
from fbtcal.vrchat_log import VrchatLogWatcher

T = "2026.11.01 21:22:16 Debug      -  "
LOGIN = T + "[Authentication - ÎÎÍÏ] Preforming post login checks for Lokinya (usr_c1fee22a)\n"
LOCAL = T + '[Behaviour] Initialized PlayerAPI "Lokinya" is local\n'


def switch(who, avatar):
    return T + f"[Behaviour] Switching {who} to avatar {avatar}\n"


def watcher(tmp_path, text):
    path = tmp_path / "output_log_2026-11-01_19-02-02.txt"
    path.write_text(text)
    seen = []
    w = VrchatLogWatcher(lambda: None, lambda: None, lambda m: None, on_avatar=seen.append)
    w.log_dir = tmp_path
    w.poll()
    return w, path, seen


def append(w, path, text):
    with open(path, "a") as f:
        f.write(text)
    w.poll()


def test_picks_up_avatar_already_worn_at_start(tmp_path):
    text = LOGIN + switch("Lokinya", "Loksinya Succubus") + switch("Someone", "POI") \
        + switch("Lokinya", "Kittysan Void Hoppou") + switch("Someone", "OBAKO")
    w, path, seen = watcher(tmp_path, text)
    assert seen == ["Kittysan Void Hoppou"]  # only the one worn now, not the history


def test_other_players_do_not_change_ours(tmp_path):
    w, path, seen = watcher(tmp_path, LOGIN)
    append(w, path, switch("Someone", "POI") + switch("Lokinya", "Kihopki") + switch("Lokinya2", "OBAKO"))
    assert seen == ["Kihopki"]


def test_first_switch_logged_before_we_know_our_name(tmp_path):
    # In a fresh VRChat start our "Switching" line comes before the "is local" line.
    w, path, seen = watcher(tmp_path, "")
    append(w, path, switch("Lokinya", "Loksinya Succubus") + switch("Someone", "POI"))
    assert seen == []
    append(w, path, LOCAL)
    assert seen == ["Loksinya Succubus"]


def calibrator(tmp_path, monkeypatch, settings):
    monkeypatch.setattr(main, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(main, "CONFIG", tmp_path)
    monkeypatch.setattr(main, "log", lambda m: None)
    (tmp_path / "settings.json").write_text(json.dumps(settings))
    c = Calibrator.__new__(Calibrator)
    c.settings = dict(settings)
    c.avatar = None
    return c


def test_stance_follows_the_avatar(tmp_path, monkeypatch):
    c = calibrator(tmp_path, monkeypatch, {"stance_cm": 20})
    c.set_avatar("Kittysan Void Hoppou")
    assert c.settings["stance_cm"] == 20  # new avatar starts from the distance in use
    c.set_stance(14)
    c.set_avatar("Loksinya Succubus")
    assert c.settings["stance_cm"] == 14
    c.set_stance(24)
    c.set_avatar("Kittysan Void Hoppou")
    assert c.settings["stance_cm"] == 14
    c.set_avatar("Loksinya Succubus")
    assert c.settings["stance_cm"] == 24
    saved = json.loads((tmp_path / "settings.json").read_text())
    assert saved["avatar_stance"] == {"Kittysan Void Hoppou": 14, "Loksinya Succubus": 24}
    assert saved["stance_cm"] == 24


def test_same_avatar_again_keeps_manual_change(tmp_path, monkeypatch):
    # VRChat logs "Switching" again on every world join and FBT calibration reload.
    c = calibrator(tmp_path, monkeypatch, {"stance_cm": 12, "avatar_stance": {"Kihopki": 16}})
    c.set_avatar("Kihopki")
    c.set_stance(18)
    c.set_avatar("Kihopki")
    assert c.settings["stance_cm"] == 18
