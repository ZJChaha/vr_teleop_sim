#!/usr/bin/env python
"""离线探测带夹爪模型: 结构/尺寸/夹爪开度曲线/home 位姿 (不启动 viewer)。"""
import os
import numpy as np
import mujoco

M = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "assets", "kinova_gen3_robotiq", "scene.xml")
m = mujoco.MjModel.from_xml_path(M)
d = mujoco.MjData(m)

def nm(objtype, i):
    return mujoco.mj_id2name(m, objtype, i)

print("=" * 60)
print("nq =", m.nq, " nv =", m.nv, " nu =", m.nu, " njnt =", m.njnt)
print("actuators:", [nm(mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)])
print("joints   :", [nm(mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)])
print("jnt_limited:", m.jnt_limited.astype(int))
print("keyframes:", [nm(mujoco.mjtObj.mjOBJ_KEY, i) for i in range(m.nkey)])
print("=" * 60)

HOME = m.key("home").qpos.copy()
print("home qpos =", np.round(HOME, 5))

mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
mujoco.mj_forward(m, d)

for s in ["pinch", "pinch_site"]:
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s)
    print(f"site {s:10s} id={sid} pos={np.round(d.site_xpos[sid], 4)}")

for b in ["base_link", "bracelet_link", "base", "right_follower", "left_follower",
          "right_pad", "left_pad"]:
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
    if bid >= 0:
        print(f"body {b:16s} id={bid} pos={np.round(d.xpos[bid], 4)}")

print("-" * 60)
print("抓取目标点 (pinch) 相对基座:", np.round(d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "pinch")], 4))

# ---- 夹爪开度 vs ctrl (0~255) ----
print("-" * 60)
print("ctrl -> 两指间距 (m) 与 driver 关节角")
rf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_follower")
lf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_follower")
tip_r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_silicone_pad")
tip_l = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_silicone_pad")
for c in [0, 16, 32, 64, 96, 128, 160, 192, 224, 255]:
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    d.ctrl[:7] = HOME[:7]
    d.ctrl[7] = c
    for _ in range(4000):
        mujoco.mj_step(m, d)
    gap_f = np.linalg.norm(d.xpos[rf] - d.xpos[lf])
    gap_s = np.linalg.norm(d.xpos[tip_r] - d.xpos[tip_l])
    print(f"  ctrl={c:3d}  follower间距={gap_f*1000:6.1f}mm  pad间距={gap_s*1000:6.1f}mm"
          f"  driver={d.qpos[7]:.4f}rad  pinch_z={d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE,'pinch')][2]:.4f}")

# ---- 可达性: 从 home 做 IK 到一些下探目标 ----
print("-" * 60)
sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "pinch")
p0 = d.site_xpos[sid].copy()
print("home pinch pos:", np.round(p0, 4))

def ik(target, iters=400, lam=0.01):
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    mujoco.mj_forward(m, d)
    for _ in range(iters):
        jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, sid)
        J = np.vstack([jacp, jacr])[:, :7]
        ep = target - d.site_xpos[sid]
        U, S, Vt = np.linalg.svd(J, full_matrices=False)
        Jp = Vt.T @ np.diag(S / (S**2 + lam**2)) @ U.T
        dq = Jp @ np.concatenate([ep, np.zeros(3)])
        dq = np.clip(dq, -0.05, 0.05)
        q = d.qpos.copy()
        q[:7] = np.where(m.jnt_limited[:7], np.clip(q[:7] + dq, m.jnt_range[:7, 0], m.jnt_range[:7, 1]), q[:7] + dq)
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
    return d.site_xpos[sid].copy(), q[:7]

for dz in [0.30, 0.35, 0.40, 0.45]:
    for dy in [-0.20, -0.10, 0.0, 0.10, 0.20]:
        tgt = np.array([p0[0], p0[1] + dy, dz])
        got, q = ik(tgt)
        err = np.linalg.norm(got - tgt)
        print(f"  target(y={p0[1]+dy:+.2f}, z={dz:.2f}) -> err={err*1000:6.1f}mm  q={np.round(q,3)}")
