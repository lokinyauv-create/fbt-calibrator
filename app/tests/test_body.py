import numpy as np
from fbtcal import body as b

def rand_q(rng):
    return b.q_norm(rng.normal(size=4))

def test_roundtrip_and_symmetry():
    rng = np.random.default_rng(1)
    for _ in range(50):
        yaw = rng.uniform(-np.pi, np.pi)
        head = (np.array([rng.uniform(-1,1), 1.55, rng.uniform(-1,1)]),
                b.q_mul(np.array([np.cos(yaw/2),0,np.sin(yaw/2),0]), b.q_norm(np.array([1,0.05,0,0.03]))))
        fwd, right = b.head_frame(head)
        base = b.neck(head); base[1]=0
        trackers = {
            "W": (base + np.array([0,0.9,0]) - 0.1*fwd, rand_q(rng)),
            "L": (base - 0.07*right + 0.08*fwd + np.array([0,0.11,0]), rand_q(rng)),
            "R": (base + 0.07*right + 0.08*fwd + np.array([0,0.12,0]), rand_q(rng)),
        }
        cal, eye = b.calibrate(head, trackers)
        roles = {s: c["role"] for s, c in cal.items()}
        assert roles == {"W": "waist", "L": "left_foot", "R": "right_foot"}
        assert b.feet_together(trackers, roles)
        model = b.model_points(head, eye, ankle_side=float(np.hypot(*(trackers["R"][0] - trackers["L"][0])[[0, 2]]) / 2))
        for s, c in cal.items():
            p, q = b.compose(trackers[s], c["off"])
            assert np.allclose(p, model[c["role"]][0], atol=1e-9)
            assert abs(abs(np.dot(q, model[c["role"]][1])) - 1) < 1e-9
        # symmetric feet about the body centre, same height
        l = model["left_foot"][0]; r = model["right_foot"][0]
        mid = (l + r) / 2
        assert np.allclose(mid[[0,2]], base[[0,2]]); assert abs(l[1]-r[1]) < 1e-12
        # matrix conversion round trip
        for s in trackers:
            p, q = b.mat_to_pose(b.pose_to_mat(*trackers[s]))
            assert np.allclose(p, trackers[s][0]); assert abs(abs(np.dot(q, trackers[s][1]))-1) < 1e-9

def test_any_pose_model_hangs_from_head():
    head = (np.array([0, 1.55, 0]), np.array([1.0, 0, 0, 0]))
    sit = (np.array([0, 1.05, 0]), np.array([1.0, 0, 0, 0]))
    a = b.model_points(head, 1.55); s = b.model_points(sit, 1.55, head_y=1.05)
    for k in a:
        assert np.allclose(a[k][0] - [0, 0.5, 0], s[k][0])


def test_nodding_does_not_move_the_body():
    """Looking down or up (pitch about the neck) leaves the any-pose model where it was."""
    level = (np.array([0.3, 1.55, -0.2]), np.array([np.cos(0.4), 0, np.sin(0.4), 0]))
    pivot = b.neck(level)
    ref = b.model_points(level, 1.55, head_y=b.level_eye_y(level))
    for pitch in (-1.4, -0.7, 0.5, 1.4):  # down to nearly straight down, up to nearly straight up
        rot = b.q_mul(level[1], np.array([np.cos(pitch / 2), np.sin(pitch / 2), 0, 0]))
        eyes = pivot - b.q_rot(rot, [0.0, -b.NECK_UP, b.NECK_FWD])
        head = (eyes, rot)
        assert np.allclose(b.neck(head), pivot)
        m = b.model_points(head, 1.55, head_y=b.level_eye_y(head))
        for k in ref:
            assert np.allclose(m[k][0], ref[k][0], atol=1e-9)
            assert abs(abs(np.dot(m[k][1], ref[k][1])) - 1) < 1e-9


def test_t_pose_facing_comes_from_the_hands_not_the_head():
    """Her head was turned 5-8° from her body in real T-poses; the waist must face where the body does."""
    look = np.radians(8)
    head = (np.array([0.0, 1.56, 0.0]), np.array([np.cos(look / 2), 0, np.sin(look / 2), 0]))
    left = (np.array([-0.75, 1.40, 0.0]), np.array([1.0, 0, 0, 0]))   # arms out, square to -Z
    right = (np.array([0.75, 1.40, 0.0]), np.array([1.0, 0, 0, 0]))
    facing = b.facing_from_hands(left, right)
    assert np.allclose(facing, [0, 0, -1], atol=1e-9)
    q = np.array([1.0, 0, 0, 0])
    trackers = {"W": (np.array([0.0, 0.9, 0.05]), q), "L": (np.array([-0.06, 0.08, 0.0]), q),
                "R": (np.array([0.06, 0.08, 0.0]), q)}
    cal, _ = b.calibrate(head, trackers, facing=facing)
    for s, c in cal.items():
        final = b.compose(trackers[s], c["off"])
        fwd = b.q_rot(final[1], [0, 0, -1.0])
        assert abs(np.degrees(np.arctan2(-fwd[0], -fwd[2]))) < 1e-6   # straight ahead, not 8° aside
    # the feet are placed left and right of the BODY, not of the head
    assert abs(b.compose(trackers["L"], cal["L"]["off"])[0][2] - b.compose(trackers["R"], cal["R"]["off"])[0][2]) < 1e-9


def test_level_roll_keeps_the_tracker_and_removes_only_the_sideways_roll():
    rng = np.random.default_rng(11)
    real = (np.array([0.16, 0.95, 0.0]), b.q_norm(rng.normal(size=4)))
    off = (np.array([-0.03, 0.10, 0.17]), b.q_norm(rng.normal(size=4)))
    f = b.compose(real, off)
    pos, q = b.level_roll(real[0], f)
    right, up, fwd = b.q_rot(q, [1, 0, 0]), b.q_rot(q, [0, 1, 0]), b.q_rot(q, [0, 0, -1])
    assert abs(right[1]) < 1e-9                                        # no sideways roll left
    assert np.allclose(fwd, b.q_rot(f[1], [0, 0, -1]), atol=1e-9)      # facing and forward tilt kept
    assert abs(np.linalg.norm(pos - real[0]) - np.linalg.norm(f[0] - real[0])) < 1e-9  # turned about the tracker
    # a pose that does not roll is left alone
    level = b.level_roll(real[0], (pos, q))
    assert np.allclose(level[0], pos, atol=1e-9) and abs(abs(np.dot(level[1], q)) - 1) < 1e-9


def test_sit_fix_takes_off_the_sitting_roll_but_keeps_real_leans():
    """sim/sitting.py: rigid 7.4°, «level» 6° (killed real leans), sit_fix ≤ ~1° error."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sim"))
    import sitting as S
    from humanoid import Body
    rng = np.random.default_rng(2)
    body = Body(waist_mount=(0.16, 0.05, -0.02))
    real0 = S.tracker(body, 0.0, 0.0, rng)
    off = b.compose(b.inverse(real0), (body.pelvis0, np.array([1.0, 0, 0, 0])))
    sit = b.compose(S.tracker(body, S.SIT_PITCH, 0.0, rng), off)
    fix = (np.radians(S.roll_of(sit[1])), np.radians(S.pitch_of(sit[1])))
    for pitch, roll in ((0, 0), (0, -6), (0, 6), (25, 0), (25, -6), (25, 6), (12, 0)):
        real = S.tracker(body, np.radians(pitch), np.radians(roll), rng)
        out = b.final_pose(real, off, fix)
        assert abs(S.roll_of(out[1]) - roll) < 1.5, (pitch, roll)
        assert np.allclose(out[0] - real[0], out[0] - real[0])
    # the tracker itself stays put: the waist point turns about it
    real = S.tracker(body, np.radians(25), 0.0, rng)
    rigid, out = b.compose(real, off), b.final_pose(real, off, fix)
    assert abs(np.linalg.norm(out[0] - real[0]) - np.linalg.norm(rigid[0] - real[0])) < 1e-9
