#!/usr/bin/env python
"""打印一个 MJCF 的完整索引表 —— 填 ArmSpec 的 joint_offset / actuator_offset 用。

用法:
    python tools/list_model.py                      # 用 profile 里的模型
    python tools/list_model.py path/to/your.xml     # 指定模型
    VR_TELEOP_PROFILE=robot_profile_mydual python tools/list_model.py

输出四张表（下标 = 你在 profile 里要填的数字）:

    joints     下标 / 名字 / 类型 / qpos 地址 / dof 地址 / 限位
    actuators  下标 / 名字 / 驱动的关节
    sites      下标 / 名字（填 ee_site）
    cameras    下标 / 名字（填 cameras / extra_cameras）
    bodies     名字（填 gripper_body / object_bodies）

纯只读，不改任何文件。
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import mujoco                                                  # noqa: E402

JTYPES = {
    int(mujoco.mjtJoint.mjJNT_FREE): "free",
    int(mujoco.mjtJoint.mjJNT_BALL): "ball",
    int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
    int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
}


def nm(m, objtype, i):
    return mujoco.mj_id2name(m, objtype, i) or f"<unnamed {i}>"


def main():
    if len(sys.argv) > 1:
        path = os.path.abspath(sys.argv[1])
    else:
        from robot_profile import load_profile
        path = load_profile().model_path

    print("=" * 78)
    print(f"model : {path}")
    if not os.path.exists(path):
        print("✗ 文件不存在")
        return 1
    m = mujoco.MjModel.from_xml_path(path)
    print(f"nq={m.nq}  nv={m.nv}  nu={m.nu}  njnt={m.njnt}  "
          f"nsite={m.nsite}  ncam={m.ncam}  nbody={m.nbody}")
    print("=" * 78)

    print("\n--- joints（下标就是 joint_offset 的基准）" + "-" * 26)
    print(f"{'id':>3}  {'name':<26} {'type':<6} {'qpos':>5} {'dof':>4}  range")
    for j in range(m.njnt):
        lim = (f"[{m.jnt_range[j, 0]:+.3f}, {m.jnt_range[j, 1]:+.3f}]"
               if m.jnt_limited[j] else "无限位")
        print(f"{j:>3}  {nm(m, mujoco.mjtObj.mjOBJ_JOINT, j):<26} "
              f"{JTYPES.get(int(m.jnt_type[j]), '?'):<6} {m.jnt_qposadr[j]:>5} "
              f"{m.jnt_dofadr[j]:>4}  {lim}")

    print("\n--- actuators（下标就是 actuator_offset 的基准）" + "-" * 18)
    print(f"{'id':>3}  {'name':<26} {'传动类型':<10} {'驱动对象':<24} ctrlrange")
    TRN = {int(mujoco.mjtTrn.mjTRN_JOINT): "joint",
           int(mujoco.mjtTrn.mjTRN_TENDON): "tendon",
           int(mujoco.mjtTrn.mjTRN_SITE): "site",
           int(mujoco.mjtTrn.mjTRN_BODY): "body"}
    for a in range(m.nu):
        tt = int(m.actuator_trntype[a])
        tid = int(m.actuator_trnid[a, 0])
        kind = TRN.get(tt, f"?({tt})")
        if tt == int(mujoco.mjtTrn.mjTRN_JOINT):
            target = nm(m, mujoco.mjtObj.mjOBJ_JOINT, tid)
        elif tt == int(mujoco.mjtTrn.mjTRN_TENDON):
            target = nm(m, mujoco.mjtObj.mjOBJ_TENDON, tid)
        elif tt == int(mujoco.mjtTrn.mjTRN_SITE):
            target = nm(m, mujoco.mjtObj.mjOBJ_SITE, tid)
        else:
            target = str(tid)
        cr = m.actuator_ctrlrange[a]
        crs = f"[{cr[0]:+.3f}, {cr[1]:+.3f}]" if m.actuator_ctrllimited[a] else "无限"
        print(f"{a:>3}  {nm(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a):<26} "
              f"{kind:<10} {target:<24} {crs}")

    print("\n--- sites（填 ee_site）" + "-" * 45)
    for s in range(m.nsite):
        print(f"{s:>3}  {nm(m, mujoco.mjtObj.mjOBJ_SITE, s)}")

    print("\n--- cameras（填 cameras / extra_cameras）" + "-" * 27)
    for c in range(m.ncam):
        print(f"{c:>3}  {nm(m, mujoco.mjtObj.mjOBJ_CAMERA, c)}")

    print("\n--- bodies（填 gripper_body / object_bodies）" + "-" * 22)
    for b in range(m.nbody):
        print(f"{b:>3}  {nm(m, mujoco.mjtObj.mjOBJ_BODY, b)}")

    print("\n" + "=" * 78)
    print("填 ArmSpec 时：")
    print("  joint_offset     = 这条臂第一个关节的 id（见 joints 表）")
    print("  actuator_offset  = 这条臂第一个执行器的 id（见 actuators 表）")
    print("  n_arm_qpos       = 这条臂的关节数（不含夹爪）")
    print("  n_gripper_qpos   = 夹爪占几个 qpos（2F-85 = 8；单自由度夹爪 = 1）")
    print("  gripper.ctrl_index / qpos_index = **相对本臂** 的下标（单臂时就是绝对值）")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
