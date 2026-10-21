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
        base = head[0] - 0.07*fwd; base[1]=0
        trackers = {
            "W": (base + np.array([0,0.9,0]) - 0.1*fwd, rand_q(rng)),
            "L": (base - 0.07*right + 0.08*fwd + np.array([0,0.11,0]), rand_q(rng)),
            "R": (base + 0.07*right + 0.08*fwd + np.array([0,0.12,0]), rand_q(rng)),
        }
        cal, eye = b.calibrate(head, trackers)
        roles = {s: c["role"] for s, c in cal.items()}
        assert roles == {"W": "waist", "L": "left_foot", "R": "right_foot"}
        assert b.feet_together(trackers, roles)
        model = b.model_points(head, eye)
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
