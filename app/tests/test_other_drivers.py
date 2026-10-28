"""Another driver (Space Calibrator) moving trackers after ours must not break calibration."""
import numpy as np
from fbtcal import body as b
from fbtcal.main import Calibrator


class FakeShm:
    def write_slots(self, slots):
        self.slots = slots


def calibrator():
    c = Calibrator.__new__(Calibrator)
    c.shm = FakeShm()
    c.cal, c.eye, c.override, c.vrc_eye = {}, None, None, None
    c.world_fix = {}
    c.settings = {"stance_cm": 12}
    return c


def yaw_q(a):
    return np.array([np.cos(a / 2), 0, np.sin(a / 2), 0])


def test_calibrate_and_override_through_a_later_driver():
    rng = np.random.default_rng(3)
    head = (np.array([2.42, 1.62, 0.23]), yaw_q(0.7))
    std_from_raw = (np.array([0.1, 0.0, -0.2]), yaw_q(0.3))
    # what the later driver does to every Lighthouse pose (3.85 m / 153° seen 2026-10-28)
    later = (np.array([2.9, 1.9, -1.5]), b.q_norm(np.array([0.23, 0.1, 0.96, 0.05])))

    fwd, right = b.head_frame(head)
    base = b.neck(head); base[1] = 0
    body = {  # where the trackers really are, in the space apps see
        "W": (base + np.array([0, 0.92, 0]), b.q_norm(rng.normal(size=4))),
        "L": (base - 0.07 * right + np.array([0, 0.1, 0]), b.q_norm(rng.normal(size=4))),
        "R": (base + 0.07 * right + np.array([0, 0.1, 0]), b.q_norm(rng.normal(size=4))),
        "HMD-MOUNT": (head[0] + np.array([0, 0.05, 0.1]), b.q_norm(rng.normal(size=4))),
    }
    # the raw pose our driver reads is before `later`, in raw space
    raw = {s: b.compose(b.inverse(std_from_raw), b.compose(b.inverse(later), p)) for s, p in body.items()}

    def final(c, s):  # our driver, then the later one, as apps see it
        if c.override and s in c.override:
            out = c.override[s]
        else:
            out = b.compose(raw[s], c.cal[s]["off"]) if s in c.cal else raw[s]
        return b.compose(later, b.compose(std_from_raw, out))

    c = calibrator()
    c.std_from_raw = std_from_raw
    seen = {s: c.body_pose(s, final(c, s), raw[s]) for s in body}
    for s in body:
        assert np.allclose(seen[s][0], body[s][0], atol=1e-9)

    cal, eye = b.calibrate(head, seen)
    assert {s: x["role"] for s, x in cal.items()} == {"W": "waist", "L": "left_foot", "R": "right_foot"}
    c.cal, c.eye = cal, eye
    model = b.model_points(head, eye)
    for s, x in cal.items():
        assert np.allclose(final(c, s)[0], model[x["role"]][0], atol=1e-9)
        c.body_pose(s, final(c, s), raw[s])  # next tick, with the offset applied

    c.follow_head(head)
    model = b.model_points(head, eye, head_y=b.level_eye_y(head), ankle_side=0.06)
    for s, x in cal.items():
        assert np.allclose(final(c, s)[0], model[x["role"]][0], atol=1e-9)
        assert abs(abs(np.dot(final(c, s)[1], model[x["role"]][1])) - 1) < 1e-9


def test_headset_mounted_tracker_is_not_a_body_tracker():
    head = (np.array([0, 1.6, 0]), np.array([1.0, 0, 0, 0]))
    q = np.array([1.0, 0, 0, 0])
    trackers = {"on-head": (np.array([0, 1.7, 0.1]), q), "W": (np.array([0, 0.95, 0]), q),
                "L": (np.array([-0.07, 0.08, 0]), q), "R": (np.array([0.07, 0.08, 0]), q)}
    cal, _ = b.calibrate(head, trackers)
    assert set(cal) == {"W", "L", "R"} and cal["W"]["role"] == "waist"
