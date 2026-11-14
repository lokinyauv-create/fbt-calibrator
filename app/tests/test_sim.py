"""The virtual body (sim/) through the real calibration: nothing jumps and the avatar's hips follow the body."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sim"))
import run as sim  # noqa: E402
from humanoid import Body  # noqa: E402


def test_any_height_any_body_nothing_jumps_and_the_hips_follow():
    for eye, kw in ((1.45, {}), (1.85, {"pelvis_behind_neck": 0.06})):
        body = Body(eye=eye, seed=3, **kw)
        r = sim.run_auto("t", body, body.pose(head_yaw=np.radians(8)), verbose=False)
        assert r["jumps"]["waist"] < 0.2 and r["worst"] < 1.0      # cm


def test_every_belt_spot_from_the_settings_works():
    from fbtcal import body as B
    for mount in ("front", "right", "back_left"):
        a = np.radians(B.MOUNTS[mount])                       # the tracker really hangs there
        body = Body(eye=1.56, seed=4, waist_mount=(0.16 * np.sin(a), 0.05, -0.11 * np.cos(a)))
        r = sim.run_auto("t", body, body.pose(), verbose=False, mount=mount)
        assert r["jumps"]["waist"] < 0.2 and r["worst"] < 1.0
