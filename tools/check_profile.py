#!/usr/bin/env python
"""检查 robot_profile.py 与你的 MuJoCo 模型是否对得上。

用法:
    python tools/check_profile.py                     # 检查默认配置
    VR_TELEOP_PROFILE=robot_profile_myarm python tools/check_profile.py

它会逐项核对并给出 ✅/⚠️/❌：
    · 模型能不能加载
    · 关节/执行器/qpos 的数量够不够
    · ee_site / 夹爪 body / 方块 body / 相机 是否真的存在
    · home_qpos 是否在关节限位内
    · 夹爪的 ctrl 区间与 qpos 行程是否自洽（最容易填错的一项）

不改任何文件，纯只读检查。
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import mujoco                                                   # noqa: E402
from robot_profile import load_profile                          # noqa: E402

OK, WARN, BAD = "✅", "⚠️ ", "❌"
rows = []
bad = 0


def chk(name, cond, detail="", soft=False):
    global bad
    mark = OK if cond else (WARN if soft else BAD)
    if not cond and not soft:
        bad += 1
    rows.append((mark, name, detail))
    return cond


def check_arms(P, m, d):
    """逐臂检查（双臂配置走这里；单臂仍走原来的标量路径）。

    每条臂要能独立回答三个问题：
      1) 索引能不能解析出模型里的真实关节/执行器？（joint_offset / actuator_offset）
      2) 两条臂的 qpos 区间、执行器区间有没有重叠？（重叠 = 两条臂抢同一个关节）
      3) 这条臂的 home 位姿在不在限位内、夹爪方向对不对？
    """
    specs = P.arm_specs
    chk("臂数量 ≥ 1", len(specs) >= 1, f"{len(specs)} 条")

    seen_q, seen_a = [], []
    for i, s in enumerate(specs):
        tag = f"[臂{i}:{s.name}]"
        j0, nq = int(s.joint_offset), int(s.n_arm_qpos)
        a0, nu = int(s.actuator_offset), int(s.n_arm_actuators)

        ok_j = chk(f"{tag} 关节区间在模型内", 0 <= j0 and j0 + nq <= m.njnt,
                   f"joint[{j0}:{j0 + nq}) of njnt={m.njnt}")
        ok_a = chk(f"{tag} 执行器区间在模型内", 0 <= a0 and a0 + nu <= m.nu,
                   f"ctrl[{a0}:{a0 + nu}) of nu={m.nu}")
        chk(f"{tag} drive_hand 合法", str(s.drive_hand).lower() in ("left", "right"),
            f"{s.drive_hand!r}")

        if not ok_j:
            continue
        q_off = int(m.jnt_qposadr[j0])
        q_end = q_off + nq + int(s.n_gripper_qpos)
        chk(f"{tag} qpos 区间在模型内", q_end <= m.nq,
            f"qpos[{q_off}:{q_end}) of nq={m.nq}")
        seen_q.append((s.name, q_off, q_end))
        if ok_a:
            seen_a.append((s.name, a0, a0 + nu))

        chk(f"{tag} 末端 site '{s.ee_site}' 存在",
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s.ee_site) >= 0)
        chk(f"{tag} 夹爪 body '{s.gripper_body}' 存在",
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, s.gripper_body) >= 0,
            soft=True)

        # 夹爪执行器（相对本臂）；n_gripper_qpos=0 表示没有夹爪，跳过
        if int(s.n_gripper_qpos) > 0:
            gi = a0 + int(s.gripper.ctrl_index)
            chk(f"{tag} 夹爪执行器下标有效", gi < m.nu, f"ctrl[{gi}] of nu={m.nu}")
        else:
            rows.append((OK, f"{tag} 无夹爪", "n_gripper_qpos=0，夹爪相关跳过"))

        # home 位姿
        chk(f"{tag} home_qpos 长度 = n_arm_qpos", len(s.home_qpos) == nq,
            f"{len(s.home_qpos)} vs {nq}")
        if len(s.home_qpos) == nq and nq > 0:
            q = np.asarray(s.home_qpos, float)
            js = list(range(j0, j0 + nq))
            over = []
            for k, j in enumerate(js):
                if bool(m.jnt_limited[j]):
                    lo, hi = m.jnt_range[j]
                    if q[k] < lo - 1e-6 or q[k] > hi + 1e-6:
                        over.append(f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)}"
                                    f"={q[k]:+.3f}∉[{lo:+.3f},{hi:+.3f}]")
            chk(f"{tag} home 位姿在限位内", not over, "; ".join(over))
            # 这个位姿下末端在哪（写进 qpos 再前向一次）
            d.qpos[:] = 0.0
            d.qpos[q_off:q_off + nq] = q
            mujoco.mj_forward(m, d)
            sidx = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s.ee_site)
            if sidx >= 0:
                p = d.site_xpos[sidx]
                tcp = p - float(s.tcp_offset) * d.site_xmat[sidx].reshape(3, 3)[:, 2]
                rows.append((OK, f"{tag} home 下 TCP 位置",
                             f"site={np.round(p, 3).tolist()} TCP={np.round(tcp, 3).tolist()}"))

    # 重叠检查：两条臂抢同一个 qpos / 执行器 → 一定有一条不受控
    def overlap(pairs):
        out = []
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                n1, a1, b1 = pairs[i]
                n2, a2, b2 = pairs[j]
                if max(a1, a2) < min(b1, b2):
                    out.append(f"{n1}[{a1}:{b1}) ∩ {n2}[{a2}:{b2})")
        return out

    oq, oa = overlap(seen_q), overlap(seen_a)
    chk("各臂 qpos 区间互不重叠", not oq, "; ".join(oq))
    chk("各臂执行器区间互不重叠", not oa, "; ".join(oa))

    hands = [str(s.drive_hand).lower() for s in specs]
    chk("每条臂绑定了不同的手柄", len(set(hands)) == len(hands),
        f"{hands}" + ("（两条臂抢同一个手柄 → 只能一起动）"
                      if len(set(hands)) != len(hands) else ""))

    rows.append((OK, "双臂布局",
                 "｜".join(f"{s.name}:手={s.drive_hand} 关节@{s.joint_offset} "
                           f"ctrl@{s.actuator_offset}" for s in specs)))


def main():
    P = load_profile()
    print("=" * 78)
    print(f"profile : {P.name}")
    print(f"model   : {P.model_path}")
    print("=" * 78)

    if not os.path.exists(P.model_path):
        print(f"{BAD} 模型文件不存在: {P.model_path}")
        return 1
    m = mujoco.MjModel.from_xml_path(P.model_path)
    d = mujoco.MjData(m)

    # ---------- 1. 维度 ----------
    chk("模型可加载", True, f"nq={m.nq} nv={m.nv} nu={m.nu} ncam={m.ncam}")
    if not P.arms:      # 双臂时这些标量字段不参与控制，改由 check_arms 逐条核对
        chk("n_arm_qpos + n_gripper_qpos ≤ nq",
            P.n_arm_qpos + P.n_gripper_qpos <= m.nq,
            f"{P.n_arm_qpos}+{P.n_gripper_qpos} vs nq={m.nq}")
        chk("n_arm_actuators ≤ nu", P.n_arm_actuators <= m.nu,
            f"{P.n_arm_actuators} vs nu={m.nu}")
        chk("夹爪执行器下标在范围内", P.gripper.ctrl_index < m.nu,
            f"ctrl[{P.gripper.ctrl_index}] of {m.nu}")
        if P.gripper.boost_ctrl_index is not None:
            chk("增压执行器下标在范围内（或设为 None）",
                P.gripper.boost_ctrl_index < m.nu,
                f"ctrl[{P.gripper.boost_ctrl_index}] of {m.nu}", soft=True)

    # ---------- 2. 名字 ----------
    def sid(objtype, name):
        return mujoco.mj_name2id(m, objtype, name)

    if not P.arms:      # 双臂时末端由每条 ArmSpec 各自声明，下面 check_arms 逐条检查
        chk(f"末端 site '{P.ee_site}' 存在",
            sid(mujoco.mjtObj.mjOBJ_SITE, P.ee_site) >= 0)
        chk(f"夹爪 body '{P.gripper_body}' 存在",
            sid(mujoco.mjtObj.mjOBJ_BODY, P.gripper_body) >= 0, soft=True)
    for b in P.object_bodies:
        chk(f"被操作 body '{b}' 存在", sid(mujoco.mjtObj.mjOBJ_BODY, b) >= 0)
    for c in P.cameras:
        if not (c.preview or c.record):
            continue        # 两个开关都关 = 这路完全不使用，不必检查
        exists = sid(mujoco.mjtObj.mjOBJ_CAMERA, c.model_name) >= 0
        injected = c.model_name in P.extra_cameras
        tags = ("预览" if c.preview else "") + ("+录制" if c.record else "")
        chk(f"相机 '{c.model_name}' 可用", exists or injected,
            ("模型里已有" if exists else ("内存注入" if injected else "缺失"))
            + (f"（{tags.lstrip('+')}）" if tags else ""))

    # ---------- 3/4. 逐臂（双臂）或标量（单臂）----------
    if P.arms:
        check_arms(P, m, d)
    else:
        # ---------- 3. 初始位姿 ----------
        chk("home_qpos 长度正确", len(P.home_qpos) == P.n_arm_qpos,
            f"{len(P.home_qpos)} vs {P.n_arm_qpos}")
        if len(P.home_qpos) == P.n_arm_qpos:
            q = np.asarray(P.home_qpos, float)
            over = []
            for j in range(P.n_arm_qpos):
                lo, hi = m.jnt_range[j]
                limited = bool(m.jnt_limited[j]) if j < len(m.jnt_limited) else True
                if limited and (q[j] < lo - 1e-6 or q[j] > hi + 1e-6):
                    over.append(f"j{j + 1}={q[j]:+.3f}∉[{lo:+.3f},{hi:+.3f}]")
            chk("home_qpos 在关节限位内", not over, "; ".join(over))
            # 直接量一下这个位姿下末端在哪
            d.qpos[:P.n_arm_qpos] = q
            mujoco.mj_forward(m, d)
            sidx = sid(mujoco.mjtObj.mjOBJ_SITE, P.ee_site)
            if sidx >= 0:
                p = d.site_xpos[sidx]
                tcp = p - P.tcp_offset * d.site_xmat[sidx].reshape(3, 3)[:, 2]
                rows.append((OK, "home 位姿下的 TCP 位置",
                             f"site={np.round(p, 3).tolist()}  TCP={np.round(tcp, 3).tolist()}"))

        # ---------- 4. 夹爪刻度（最容易填错） ----------
        g = P.gripper
        lo_q, hi_q = float(m.jnt_range[g.qpos_index][0]), float(m.jnt_range[g.qpos_index][1])
        chk("qpos_at_open/close 在该关节量程内",
            lo_q - 1e-6 <= min(g.qpos_at_open, g.qpos_at_close)
            and max(g.qpos_at_open, g.qpos_at_close) <= hi_q + 1e-6,
            f"量程 [{lo_q:.3f}, {hi_q:.3f}]，配置 {g.qpos_at_open}→{g.qpos_at_close}")
        chk("夹爪 qpos 行程不为零",
            abs(g.qpos_at_close - g.qpos_at_open) > 1e-3,
            f"行程 {abs(g.qpos_at_close - g.qpos_at_open):.4f}")

        # 真跑一次：下发"开"和"闭"，看 qpos 到底往哪边走
        def drive(ctrl_val, steps=600):
            d.qpos[:] = 0
            d.qvel[:] = 0
            for _ in range(steps):
                d.ctrl[:] = 0
                d.ctrl[g.ctrl_index] = ctrl_val
                mujoco.mj_step(m, d)
            return float(d.qpos[g.qpos_index])

        q_open = drive(g.ctrl_open)
        q_close = drive(g.ctrl_close)
        want_open, want_close = g.qpos_at_open, g.qpos_at_close
        ok_dir = (q_close - q_open) * (want_close - want_open) > 0
        chk("夹爪方向一致（下发开/闭 → qpos 变化方向与配置相同）", ok_dir,
            f"实测 {q_open:+.3f}→{q_close:+.3f}，配置 {want_open}→{want_close}")
        err = max(abs(q_open - want_open), abs(q_close - want_close))
        chk("夹爪刻度数值接近实测", err < 0.15,
            f"最大差 {err:.3f}（差太多说明 qpos_at_open/close 需要重标）", soft=True)

    # ---------- 4.5 遥操作映射 ----------
    t = P.teleop
    chk("drive_hand 是 right/left", t.drive_hand in ("right", "left"),
        f"当前 = {t.drive_hand!r}")
    chk("scale > 0", t.scale > 0, f"当前 = {t.scale}")
    rm = np.asarray(t.rot_map, float)
    chk("rot_map 是 3x3", rm.shape == (3, 3), f"shape={rm.shape}")
    if rm.shape == (3, 3):
        chk("rot_map 是正交矩阵（能当旋转用）",
            np.allclose(rm @ rm.T, np.eye(3), atol=1e-6),
            "R·Rᵀ 应等于 I")
        chk("rot_map 行列式为 +1（不是镜像）",
            abs(np.linalg.det(rm) - 1.0) < 1e-6, f"det={np.linalg.det(rm):+.4f}")
    chk("axis_signs / rot_signs 都是 3 个 ±1",
        len(t.axis_signs) == 3 and len(t.rot_signs) == 3
        and all(abs(float(v)) == 1.0 for v in list(t.axis_signs) + list(t.rot_signs)),
        f"axis={t.axis_signs} rot={t.rot_signs}")
    rows.append((OK, "遥操作映射",
                 f"驱动手={t.drive_hand}  跟姿态={'是' if t.follow_orientation else '否'}  "
                 f"scale={t.scale}"))

    # ---------- 4.6 相机开关 ----------
    prev = [c.model_name for c in P.cameras if c.preview]
    rec = [c.model_name for c in P.cameras if c.record]
    # 双臂 / 纯遥操作配置可以不录数据 → 只提醒，不算错
    chk("至少一路相机在录制", len(rec) > 0, f"录制 {rec or '（无）'}",
        soft=(P.is_dual_arm or len(P.cameras) == 0))
    rows.append((OK, "相机开关",
                 f"预览 {len(prev)} 路 {prev}｜录制 {len(rec)} 路 {rec}"
                 f"（录制路数 = 数据集体积与渲染开销的倍数）"))

    # ---------- 5. 打印 ----------
    print()
    for mark, name, detail in rows:
        print(f" {mark} {name}" + (f"  —  {detail}" if detail else ""))
    print()
    print("=" * 78)
    if bad:
        print(f"{BAD} {bad} 项必须修（其余 ⚠️ 只是提醒）")
    else:
        print("✅ 全部必检项通过，可以 ./run.sh 试着连头显了")
    print("=" * 78)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
