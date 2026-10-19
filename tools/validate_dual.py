#!/usr/bin/env python
"""双臂遥操作的无头验证 —— 不连头显，用假手柄消息驱动两条臂。

用法:
    VR_TELEOP_PROFILE=robot_profile_mydual python tools/validate_dual.py

（仓库不自带双臂模型：这份工具验的是**你自己**的 arms=[...] 配置与 MJCF 接线。）

检查五件事（任一条失败就说明双臂接线有问题）:
    1) 索引解析：每条臂的 site / qpos / ctrl 都落在模型内，且互不重叠
    2) 跟随正确性：手柄移动 → 该臂末端到「配置规定的映射」算出的目标点
    3) 独立性：只动左手柄时，右臂关节的变化不超过「静止基线漂移」
    4) 夹爪独立：左手扣扳机只影响左臂夹爪执行器
    5) 双臂同时：两手各自移动，两臂各自跟随、互不串扰

不需要显示器、不写任何文件、不碰数据集。
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import mujoco                                                    # noqa: E402

import server as S                                               # noqa: E402

OK, BAD = "✅", "❌"
fails = []


def chk(name, cond, detail=""):
    print(f" {OK if cond else BAD} {name}" + (f"  —  {detail}" if detail else ""))
    if not cond:
        fails.append(name)
    return cond


def hand(pos, squeeze=0.0, trigger=0.0, quat=(0, 0, 0, 1.0)):
    return {"pos": [float(v) for v in pos], "quat": [float(v) for v in quat],
            "squeeze": float(squeeze), "trigger": float(trigger)}


def main():
    P = S.PROFILE
    print("=" * 78)
    print(f"profile : {P.name}")
    print(f"模型    : {P.model_path}")
    print("=" * 78)

    if not S.DUAL_ARM:
        print(f" {BAD} 这不是双臂配置（n_arms=1）。")
        print("       用 VR_TELEOP_PROFILE=<你的双臂 profile> 指定一份 arms=[...] 配置。")
        print("       怎么写见 docs/08_DUAL_ARM.md 第 2 节 / 第 9 节。")
        return 1

    rig = S.Rig(reset=False)
    model, data = rig.model, rig.data
    arms = S.build_arms(model, data)
    for a in arms:
        a.apply_home()
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep

    # ---------------- [1] 索引 ----------------
    print(f"\n[1] 索引解析（nq={model.nq} nu={model.nu} njnt={model.njnt}）")
    spans_q, spans_a = [], []
    for a in arms:
        chk(f"{a.name} site/qpos/ctrl 均在模型内",
            a.q_end <= model.nq and a.a_off + a.nu <= model.nu and a.site_id >= 0,
            f"qpos[{a.q_off}:{a.q_end}) ctrl[{a.a_off}:{a.a_off + a.nu}) site={a.site_id}")
        spans_q.append((a.q_off, a.q_end, a.name))
        spans_a.append((a.a_off, a.a_off + a.nu, a.name))
    for spans, label in ((spans_q, "qpos"), (spans_a, "ctrl")):
        ok = True
        for i in range(len(spans)):
            for j in range(i + 1, len(spans)):
                if max(spans[i][0], spans[j][0]) < min(spans[i][1], spans[j][1]):
                    ok = False
        chk(f"各臂 {label} 区间不重叠", ok,
            " ".join(f"{n}[{a}:{b})" for a, b, n in spans))

    def run(n, msgs=None):
        """跑 n 帧。msgs = {臂名: 手柄消息}（None = 全部松手）。"""
        for _ in range(n):
            for a in arms:
                a.on_hand((msgs or {}).get(a.name))
            S.step_arms(arms, model, data, dt)
            mujoco.mj_step(model, data)

    def release(n=30):
        """全部松手，让锁定/锚点/blend 复位。"""
        run(n, {})

    # 静止基线：不碰手柄时，位置伺服在重力下的自然漂移
    q_ref = {a.name: data.qpos[a.q_sl].copy() for a in arms}
    release(120)
    drift = {a.name: float(np.abs(data.qpos[a.q_sl] - q_ref[a.name]).max())
             for a in arms}
    print("\n  静止基线漂移 (120 帧, 无手柄输入): "
          + "  ".join(f"{k}={v:.2e} rad" for k, v in drift.items()))

    left = next(a for a in arms if a.hand_name == "left")
    right = next(a for a in arms if a.hand_name == "right")
    tcp0 = {a.name: a.tcp().copy() for a in arms}
    print("  初始 TCP: " + "  ".join(
        f"{a.name}=({tcp0[a.name][0]:+.3f},{tcp0[a.name][1]:+.3f},{tcp0[a.name][2]:+.3f})"
        for a in arms))

    # ---------------- [2] 跟随正确性 ----------------
    print("\n[2] 跟随正确性：左手柄沿自身 x 轴移 5cm，末端应到配置算出的目标点")
    p0 = np.zeros(3)
    p1 = np.array([0.05, 0.0, 0.0])
    q_ik = left.ee_quat()
    want, _ = S.map_hand_to_target(p1, [0, 0, 0, 1.0], p0, [0, 0, 0, 1.0],
                                   tcp0[left.name], q_ik)
    # 先锁定（第一帧只建立锚点），再用同一个手位置跑够帧数让伺服跟上
    run(2, {left.name: hand(p0, squeeze=1.0)})
    run(400, {left.name: hand(p1, squeeze=1.0)})
    got = left.tcp()
    err = float(np.linalg.norm(got - want))
    chk("末端到达映射目标点", err < 0.03,
        f"目标={np.round(want, 3).tolist()} 实到={np.round(got, 3).tolist()} 误差={err*1000:.1f}mm")
    chk("末端确实离开了初始位置",
        float(np.linalg.norm(got - tcp0[left.name])) > 0.01,
        f"{np.linalg.norm(got - tcp0[left.name])*1000:.1f} mm")

    # ---------------- [3] 独立性 ----------------
    print("\n[3] 独立性：只动左手柄，右臂只能有基线级漂移")
    qR = data.qpos[right.q_sl].copy()
    run(200, {left.name: hand(p1 + np.array([0.02, 0.02, 0.0]), squeeze=1.0)})
    dR = float(np.abs(data.qpos[right.q_sl] - qR).max())
    limit = max(3.0 * drift[right.name], 2e-4)
    chk("右臂未被带动", dR < limit, f"Δq={dR:.2e} rad（阈值 {limit:.1e}）")

    # ---------------- [4] 夹爪独立 ----------------
    print("\n[4] 夹爪独立：左手扣扳机 → 只有左臂夹爪执行器变")
    if not (left.has_gripper and right.has_gripper):
        print(f" {OK} 跳过（{'左' if not left.has_gripper else ''}"
              f"{'右' if not right.has_gripper else ''}臂 n_gripper_qpos=0，没有夹爪）")
        left.on_hand(hand(p0, squeeze=1.0, trigger=1.0))
    else:
      release(30)
      gL0, gR0 = data.ctrl[left.idx_grip], data.ctrl[right.idx_grip]
      run(400, {left.name: hand(p0, squeeze=1.0, trigger=1.0)})
      gL, gR = data.ctrl[left.idx_grip], data.ctrl[right.idx_grip]
      span = abs(left.spec.gripper.ctrl_close - left.spec.gripper.ctrl_open)
      chk("左臂夹爪收到接近闭合的指令",
          abs(gL - left.spec.gripper.ctrl_close) < 0.05 * span + 1e-9,
          f"ctrl[{left.idx_grip}]={gL:.4f}（开 {left.spec.gripper.ctrl_open} "
          f"→ 闭 {left.spec.gripper.ctrl_close}）")
      chk("右臂夹爪保持不动", abs(gR - gR0) < 1e-9,
          f"ctrl[{right.idx_grip}]={gR:.4f}（初值 {gR0:.4f}）")

    # ---------------- [5] 双臂同时 ----------------
    print("\n[5] 双臂同时：两手各移 4cm（左手 +x、右手 -y），两臂各自跟随")
    release(30)
    tA = {a.name: a.tcp().copy() for a in arms}
    pa = np.zeros(3)
    pL = np.array([0.04, 0.0, 0.0])
    pR = np.array([0.0, -0.04, 0.0])
    run(2, {left.name: hand(pa, squeeze=1.0), right.name: hand(pa, squeeze=1.0)})
    run(400, {left.name: hand(pL, squeeze=1.0), right.name: hand(pR, squeeze=1.0)})
    dL = left.tcp() - tA[left.name]
    dR = right.tcp() - tA[right.name]
    chk("两臂都动了", float(np.linalg.norm(dL)) > 0.01 and float(np.linalg.norm(dR)) > 0.01,
        f"左 {np.linalg.norm(dL)*1000:.0f}mm  右 {np.linalg.norm(dR)*1000:.0f}mm")
    # 两手的位移方向不同 → 两臂位移向量不该几乎平行
    cos = float(dL @ dR / (np.linalg.norm(dL) * np.linalg.norm(dR) + 1e-12))
    chk("两臂位移方向不同（没有串成一件事）", cos < 0.5,
        f"cos={cos:+.2f}  左Δ={np.round(dL*1000,1).tolist()}mm  "
        f"右Δ={np.round(dR*1000,1).tolist()}mm")

    print("\n" + "=" * 78)
    if fails:
        print(f"{BAD} {len(fails)} 项失败: " + "; ".join(fails))
        return 1
    print(f"{OK} 双臂验证全部通过（{len(arms)} 条臂）")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
