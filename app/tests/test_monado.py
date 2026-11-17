"""Monado mode: the runtime's bridge publishes real poses, the app answers with offsets."""
import struct
import time

import numpy as np
from fbtcal import body as b
from fbtcal import main as m
from fbtcal import shm


class FakeShm:
    def heartbeat(self):
        pass

    def write_slots(self, slots):
        self.slots = slots


def write_devices(path, devs):
    n = shm.DEV_AT + shm.DEV_COUNT * shm.DEV_SIZE
    buf = bytearray(n)
    struct.pack_into("<IIQ", buf, 0, shm.DEV_MAGIC, 0, time.monotonic_ns())
    for i, (serial, kind, (pos, rot)) in enumerate(devs):
        struct.pack_into("<64sIIIIQ3d4d3d4d", buf, shm.DEV_AT + i * shm.DEV_SIZE, serial.encode(), 2, kind, 1, 0,
                         time.monotonic_ns(), *pos, *rot, *pos, *rot)
    path.write_bytes(bytes(buf))


def test_freeze_keeps_the_waist_where_the_standing_pelvis_is(tmp_path, monkeypatch):
    """At VRChat's calibration the frozen waist must be where it will be right after, or the hips jump."""
    monkeypatch.setattr(m, "SETTINGS_FILE", tmp_path / "settings.json")
    c = m.Calibrator.__new__(m.Calibrator)
    c.shm = FakeShm()
    q = np.array([1.0, 0, 0, 0])
    c.cal = {"W": {"role": "waist", "off": (np.array([0.0, 0.0, 0.1]), q)},
             "L": {"role": "left_foot", "off": (np.zeros(3), q)}, "R": {"role": "right_foot", "off": (np.zeros(3), q)}}
    c.eye, c.vrc_eye, c.world_fix, c.std_from_raw = 1.56, None, {}, (np.zeros(3), q)
    c.settings = {"stance_cm": 12}
    c.facing_in_head = (1.0, 0.0)
    head = (np.array([0.0, 1.56, 0.0]), q)
    c.last_real = {"W": (np.array([0.03, 0.88, -0.13]), q)}   # pelvis 3 cm right, 3 cm in front of the model
    c.follow_head(head)
    follow = b.compose(c.last_real["W"], c.cal["W"]["off"])[0]
    assert np.allclose(c.override["W"][0][[0, 2]], follow[[0, 2]])
    c.last_real = {"W": (np.array([0.6, 0.3, 0.4]), q)}        # lying: far away, the model is kept
    c.follow_head(head)
    assert np.linalg.norm(c.override["W"][0][[0, 2]] - follow[[0, 2]]) > 0.1


# ---- VRChat's own calibration measures her on the spot (nothing to press before) ----
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sim"))
from humanoid import Body, rot_y  # noqa: E402


def vrchat_press(body, state, stance=12, hands=True, saved=None, mount=None):
    c = m.Calibrator.__new__(m.Calibrator)
    c.shm = FakeShm()
    c.cal, c.eye, c.override, c.vrc_eye, c.world_fix = (saved or {}), (1.5 if saved else None), None, None, {}
    c.std_from_raw, c.vr, c.monado_on = (np.zeros(3), np.array([1.0, 0, 0, 0])), None, True
    c.waist_rec, c.note, c.last_hold_voice = None, ("", 0.0), 0.0
    c.settings = {"voice": False, "tpose": False, "stance_cm": stance, "waist_mount": mount or "auto"}
    if mount and mount in b.MOUNTS:
        c.settings["waist_angle"] = b.MOUNTS[mount]
    c.facing_in_head = (1.0, 0.0)
    head = body.head(state)
    c.last_head = head
    c.last_real = {s: p for s, p in zip(("W", "L", "R"), (body.trackers(state)[k] for k in ("waist", "left_foot", "right_foot")))}
    if hands:
        r = state["R_pelvis"]  # arms straight out to the sides, square to the body
        eye = state["eyes"]
        c.last_hands = ((eye + r @ np.array([-0.75, -0.15, 0])), (eye + r @ np.array([0.75, -0.15, 0])))
        c.last_hands = tuple((p, np.array([1.0, 0, 0, 0])) for p in c.last_hands)
    else:
        c.last_hands = (None, None)
    c.start_override("VRChat calibration")
    return c, head


def test_vrchat_calibration_works_with_nothing_calibrated_before_and_nothing_jumps():
    for eye, stance, head_yaw in ((1.45, 12, 0.0), (1.57, 20, 0.14), (1.85, 16, -0.12)):
        body = Body(eye=eye, seed=1)
        st = body.pose(head_yaw=head_yaw)             # her head is turned a little from her body
        c, head = vrchat_press(body, st, stance=stance)
        assert c.override and set(c.cal) == {"W", "L", "R"}
        assert abs(c.eye - eye) < 0.01                 # the height comes from the head
        right = st["R_pelvis"] @ np.array([1.0, 0, 0])   # her BODY's right (the head is turned a little)
        gap = np.hypot(*(c.last_real["R"][0] - c.last_real["L"][0])[[0, 2]])
        for s, role in (("W", "waist"), ("L", "left_foot"), ("R", "right_foot")):
            follow = b.compose(c.last_real[s], c.cal[s]["off"])
            frozen = c.override[s]
            d = frozen[0] - follow[0]
            assert abs(abs(float(np.dot(follow[1], frozen[1]))) - 1) < 1e-4, (eye, role)  # turn: no jump
            if role == "waist":
                assert np.linalg.norm(d) < 0.003, (eye, role)                       # no jump at all
            else:
                # the feet stay where she keeps them; only the freeze puts them at the avatar's stance,
                # so VRChat's avatar legs end up as far apart as hers (sideways only, nothing forward/up)
                assert abs(d[1]) < 0.003 and abs(float(d @ np.cross(right, [0, 1, 0]))) < 0.003, (eye, role)
                assert abs(abs(float(d @ right)) - (stance / 200 - gap / 2)) < 0.004, (eye, role)


def test_vrchat_calibration_while_lying_without_any_saved_calibration_is_refused():
    body = Body(seed=2)
    st = body.pose()
    st["eyes"] = st["eyes"] - np.array([0, 0.9, 0])   # head on the floor level
    c, _ = vrchat_press(body, st)
    assert c.override is None and not c.cal


def test_vrchat_calibration_when_not_standing_uses_the_saved_one():
    body = Body(seed=3)
    saved = {"W": {"role": "waist", "off": (np.array([0.1, 0.0, 0.15]), np.array([1.0, 0, 0, 0]))},
             "L": {"role": "left_foot", "off": (np.zeros(3), np.array([1.0, 0, 0, 0]))},
             "R": {"role": "right_foot", "off": (np.zeros(3), np.array([1.0, 0, 0, 0]))}}
    st = body.pose()
    st["eyes"] = st["eyes"] - np.array([0, 0.9, 0])
    c, _ = vrchat_press(body, st, saved=saved)
    assert c.override and np.allclose(c.cal["W"]["off"][0], [0.1, 0.0, 0.15])



# ---- she says where the waist tracker hangs ----

def test_mount_offset_points_from_the_belt_spot_to_the_body_centre():
    r, f = b.mount_offset(90, b.preset_distance(90, 1.56))
    assert abs(r + 0.16) < 1e-9 and abs(f) < 1e-9          # «right»: 16 cm to her left, same front-back
    r, f = b.mount_offset(0, b.preset_distance(0, 1.56))
    assert abs(r) < 1e-9 and abs(f + 0.11) < 1e-9          # «front»: 11 cm behind the belly
    assert abs(b.preset_distance(270, 1.80) - 0.16 * 1.80 / 1.56) < 1e-9   # bigger people, bigger belt
    r, f = b.mount_offset(100, 0.20)                       # any angle and distance
    assert abs(np.hypot(r, f) - 0.20) < 1e-9


def test_side_tracker_with_its_mount_puts_the_waist_at_the_body_centre_and_nothing_jumps():
    for mount, place in (("right", (0.16, 0.05, -0.02)), ("front", (0.0, 0.05, -0.11)), ("back_left", (-0.12, 0.05, 0.08))):
        body = Body(eye=1.56, seed=4, waist_mount=place)
        st = body.pose(head_yaw=0.1)
        c, head = vrchat_press(body, st, mount=mount)
        assert c.override
        final = b.compose(c.last_real["W"], c.cal["W"]["off"])
        tracker = c.last_real["W"][0]
        assert abs(final[0][1] - tracker[1]) < 0.01, mount                       # at the tracker's height
        assert np.hypot(*(final[0] - st["pelvis"])[[0, 2]]) < 0.035, mount       # near the real body centre
        assert np.linalg.norm(c.override["W"][0] - final[0]) < 0.003, mount     # freeze == follow


def test_mounted_lever_is_flat_so_a_tilting_side_tracker_does_not_drag_the_waist_sideways():
    body = Body(eye=1.56, seed=5, waist_mount=(0.16, 0.05, -0.02))
    st = body.pose()
    c, _ = vrchat_press(body, st, mount="right")
    off = c.cal["W"]["off"]
    real = c.last_real["W"]
    roll = np.radians(8)   # what the side tracker did when she sat (measured 2026-11-14)
    fwd = b.q_rot(b.compose(real, off)[1], [0, 0, -1.0])
    tilt = np.concatenate([[np.cos(roll / 2)], np.sin(roll / 2) * fwd])
    tilted = (real[0], b.q_norm(b.q_mul(tilt, real[1])))
    before, after = b.compose(real, off)[0], b.compose(tilted, off)[0]
    right = np.cross(fwd, [0, 1.0, 0])
    assert abs((after - before) @ right) < 0.005



def test_switching_our_calibration_off_passes_the_trackers_through():
    body = Body(seed=8)
    c, _ = vrchat_press(body, body.pose())
    assert c.override
    c.override = None
    c.settings["enabled"] = False
    c.push_slots()
    assert c.shm.slots == []                                  # no offsets at all
    c.start_override("VRChat calibration")
    assert c.override is None                                 # and no freeze


def test_the_devices_monado_publishes_are_read_back(tmp_path, monkeypatch):
    path = tmp_path / "devs"
    monkeypatch.setattr(shm, "DEV_NAME", str(path))
    q = np.array([1.0, 0, 0, 0])
    write_devices(path, [("HMD", shm.HMD, (np.array([0, 1.6, 0.0]), q)), ("W", shm.TRACKER, (np.array([0.1, 0.9, 0.0]), q))])
    got = {d["serial"]: d for d in shm.MonadoDevices().read()}
    assert set(got) == {"HMD", "W"} and got["W"]["kind"] == shm.TRACKER and np.allclose(got["W"]["pos"], [0.1, 0.9, 0])


def test_where_the_waist_tracker_hangs_is_found_by_itself():
    for mount, ang in b.MOUNTS.items():
        a = np.radians(ang)
        for eye in (1.50, 1.80):
            k = eye / 1.56
            body = Body(eye=eye, seed=13, waist_mount=(0.16 * k * np.sin(a), 0.05, -0.11 * k * np.cos(a)))
            st = body.pose(head_yaw=np.radians(7))
            c, _ = vrchat_press(body, st)
            found = c.waist_mount()[0]
            want = np.degrees(np.arctan2(0.16 * np.sin(a), 0.11 * np.cos(a))) % 360   # direction of that belt spot
            assert abs((found - want + 180) % 360 - 180) < 6, (mount, eye, found)
            assert b.place_name(found) == b.PLACE_NAMES[mount]
            final = b.compose(c.last_real["W"], c.cal["W"]["off"])
            assert np.hypot(*(final[0] - st["pelvis"])[[0, 2]]) < 0.035, (mount, eye)   # at the body centre
            assert np.linalg.norm(c.override["W"][0] - final[0]) < 0.003                 # nothing jumps


def test_a_tracker_switched_off_is_named(monkeypatch):
    body = Body(seed=14)
    c, _ = vrchat_press(body, body.pose())               # a good calibration first
    c.override = None
    said = []
    c.say = lambda text, secs=0: said.append(text)
    del c.last_real["L"]                                  # then the left foot tracker goes off
    c.start_override("VRChat calibration")
    assert c.override is None and said and "Ліва нога" in said[-1]
