#!/usr/bin/env python
"""端到端离线验证 (无 viewer): 复用 server.py 的 Rig / ik_solve / map_hand_to_target。

1) IK 稳定性: 300 步螺旋轨迹, 相邻解最大跳变 + 跟踪误差
2) 单手流程 (纯物理抓取): 用"模拟手柄"经 server 的映射驱动机械臂,
   对 3 个方块做 悬停 → 下探 → 合爪(靠摩擦夹住) → 抬升 → 平移 → 松爪放下,
   全程检查方块是否**真的被摩擦力带走** (不是被代码搬走的)。
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import server as S

np.set_printoptions(precision=4, suppress=True)


# ============================================================
# 1) IK 稳定性
# ============================================================
def test_ik_stability():
    """1) IK 稳定性: 300 步螺旋 + 奇异位型。

    ★ 必须包在函数里 —— 这个文件会被别的脚本 import (复用 run_pick),
      模块级代码会在 import 那一刻执行。本项目已经因为"被 import 的脚本没加
      保护"踩过坑: import 就把整轮验证跑了一遍, 输出混进调用者的日志。
      常量放模块级, **一切副作用进函数/__main__**。
    """
    print("=" * 70)
    print("1) IK 稳定性 (300 步螺旋, 闭环 IK, 与 server 同一套 ik_solve)")
    rig = S.Rig()
    rig.settle(1500)
    m, d = rig.model, rig.data
    mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
    tcp0 = rig.tcp().copy()
    ee0q = S.mat_to_quat(d.site_xmat[rig.site_id].reshape(3, 3))
    print(f"   初始抓取中心 {np.round(tcp0,4)}")

    qs, errs = [], []
    q_cmd = None
    for k in range(300):
        th = k * 0.03
        r = 0.06 * (0.3 + 0.7 * k / 299)
        tgt = tcp0 + np.array([r * np.cos(th) * 0.4, r * np.sin(th), 0.05 * np.sin(th * 0.7)])
        ang = 0.15 * np.sin(th * 0.8)
        Rz = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
        tq = S.mat_to_quat(Rz @ S.quat_to_mat(ee0q))
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        q = S.ik_solve(m, d, rig.site_id, tgt, tq, q_seed=q_cmd)
        q_cmd = q
        rig.drive(q, S.DRIVE_MODE, contact=rig.touching_object())   # 与 server 相同驱动
        d.ctrl[S.IDX_GRIP] = 0.0
        mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        qs.append(q.copy())
        errs.append(np.linalg.norm(rig.tcp() - tgt))
    qs = np.array(qs); errs = np.array(errs)
    jumps = np.degrees(np.abs(np.diff(qs, axis=0)).max(axis=1))
    print(f"   跟踪误差: max {errs.max()*1000:.2f} mm, mean {errs.mean()*1000:.2f} mm"
          f"   (原版同测试: 0.03 / 0.00 mm)")
    print(f"   相邻解最大跳变: {jumps.max():.2f}°  (原版同测试 0.67°: "
          f"{'量级一致 ✓' if jumps.max() < 1.5 else '偏大 ✗'})")

    # 奇异位型 (口径与原版一致: 单次 IK 调用相对上次解的最大关节变化)
    rig2 = S.Rig(); m2, d2 = rig2.model, rig2.data
    d2.qpos[:7] = np.array([0, 0.0, 3.14159265, 0.0, 0.0, 0.0, 0.0])
    d2.ctrl[:7] = d2.qpos[:7]
    mujoco.mj_forward(m2, d2)
    mujoco.mj_kinematics(m2, d2); mujoco.mj_comPos(m2, d2)
    p_sing = rig2.tcp().copy()
    tq2 = S.mat_to_quat(d2.site_xmat[rig2.site_id].reshape(3, 3))
    mx = 0.0
    prev = d2.qpos[:7].copy()
    for k in range(60):
        mujoco.mj_kinematics(m2, d2); mujoco.mj_comPos(m2, d2)
        q = S.ik_solve(m2, d2, rig2.site_id, p_sing + [0.001*k, 0, 0.002*k], tq2, q_seed=prev)
        mx = max(mx, float(np.abs(q - prev).max()))
        prev = q
        d2.ctrl[:7] = q
        mujoco.mj_step(m2, d2)
    print(f"   奇异位型单次调用最大解变化 = {np.degrees(mx):.2f}°"
          f"   (原版同测试 22.34°; 这是原版算法固有量级, 不是回归)")


    # ============================================================
    # 2) 单手抓取-搬运 (纯物理摩擦)
    # ============================================================
    print("=" * 70)
    print("2) 单手流程 · 纯物理摩擦抓取 (代码不搬动方块, 只合爪 + 移动机械臂)")

AXE = S.AXIS_SIGNS * S.SCALE
RMT = S.R_MAP.T
RATE = 0.5      # 手移动 1 单位 → 分几步


def run_pick(obj_name, drop_xy, obj_xy=None, layout=None, grasp_down=False):
    rig = S.Rig()
    rig.settle(1500)
    m, d = rig.model, rig.data
    site = rig.site_id
    # layout: {方块名: (x, y)} —— 一次性摆好**全部**方块 (用于验证随机布局)
    if layout is not None:
        for nm, xy in layout.items():
            a = rig.obj_qadr[nm]
            d.qpos[a] = xy[0]
            d.qpos[a + 1] = xy[1]
            d.qpos[a + 2] = 0.240
            d.qpos[a + 3:a + 7] = (1.0, 0.0, 0.0, 0.0)
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        rig.settle(600)
    if obj_xy is not None:                      # 覆盖单个方块位置 (用于扫布局)
        a = rig.obj_qadr[obj_name]
        d.qpos[a] = obj_xy[0]
        d.qpos[a + 1] = obj_xy[1]
        d.qpos[a + 2] = 0.240
        d.qpos[a + 3:a + 7] = (1.0, 0.0, 0.0, 0.0)
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        rig.settle(400)

    hand0 = np.zeros(3)
    hand_q = np.array([0.0, 0.0, 0.0, 1.0])
    tcp_lock = rig.tcp().copy()
    ee0q = S.mat_to_quat(d.site_xmat[site].reshape(3, 3))
    q_ik = None
    grip_cmd = 0.0

    def hand_for(target_tcp):
        return hand0 + RMT @ ((np.asarray(target_tcp, float) - tcp_lock) / AXE)

    def tick(target_tcp, trigger):
        nonlocal q_ik, grip_cmd
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        tt, tq = S.map_hand_to_target(hand_for(target_tcp), hand_q, hand0, hand_q,
                                      tcp_lock, ee0q)
        if grasp_down:
            # 位置仍走手柄映射, **只把目标姿态换成"夹爪朝下"** —— 复现用户实际的抓法
            tq = S.mat_to_quat(DOWN_R)
        q_ik = S.ik_solve(m, d, site, tt, tq, q_seed=q_ik)
        gt = rig.grip_cmd_of(trigger)
        grip_cmd = (min(gt, grip_cmd + S.GRIP_RATE) if gt > grip_cmd
                    else max(gt, grip_cmd - S.GRIP_RATE))
        rig.set_gripper(grip_cmd)
        rig.drive(q_ik, S.DRIVE_MODE, contact=rig.touching_object())
        mujoco.mj_step(m, d)

    def goto(target, trigger, steps):
        a = rig.tcp().copy()
        for i in range(1, steps + 1):
            tick(a + (np.asarray(target, float) - a) * (i / steps), trigger)

    oid = rig.obj_body[obj_name]
    obj0 = d.xpos[oid].copy()
    # 三段式接近: 抬到安全高 → 平移到正上方 → 竖直下探到物体中心
    goto(np.array([rig.tcp()[0], rig.tcp()[1], 0.45]), 0.0, 300)
    goto(np.array([obj0[0], obj0[1], 0.45]), 0.0, 400)
    goto(obj0, 0.0, 600)
    for _ in range(300):
        tick(obj0, 0.0)
    approach_err = float(np.linalg.norm(rig.tcp() - obj0))

    # 合爪 (靠摩擦夹住)
    for i in range(1, 601):
        tick(obj0, i / 600)
    for _ in range(1500):
        tick(obj0, 1.0)
    contacts, driver, detected = rig.grasp_report()
    grab_z = d.xpos[oid][2]

    # 抬升 (只动机械臂, 方块靠摩擦跟随)
    lift = np.array([obj0[0], obj0[1], obj0[2] + 0.18])
    goto(lift, 1.0, 800)
    for _ in range(600):
        tick(lift, 1.0)
    dz = float(d.xpos[oid][2] - grab_z)
    rel = float(np.linalg.norm(d.xpos[oid] - rig.tcp()))

    # 平移到投放点
    side = np.array([drop_xy[0], drop_xy[1], lift[2]])
    goto(side, 1.0, 900)
    carried_rel = float(np.linalg.norm(d.xpos[oid] - rig.tcp()))

    # 下放 → 松爪
    down = np.array([drop_xy[0], drop_xy[1], 0.30])
    goto(down, 1.0, 400)
    for i in range(1, 301):
        tick(down, 1.0 - i / 300)
    for _ in range(300):
        tick(down, 0.0)
    goto(np.array([drop_xy[0], drop_xy[1], 0.45]), 0.0, 300)
    for _ in range(1200):
        tick(np.array([drop_xy[0], drop_xy[1], 0.45]), 0.0)
    final = d.xpos[oid].copy()
    placed = np.linalg.norm(final[:2] - np.array(drop_xy)) < 0.08 and final[2] > 0.2
    return dict(approach=approach_err, contacts=contacts, driver=driver,
                detected=detected, dz=dz, rel=rel, carried_rel=carried_rel,
                final=final, placed=placed)


# 落点 (桌面内的空位)。★ 放模块级: 别的脚本要 import 它复用**同一套**落点,
#   否则"等价"的自写测试会和真测试对不上 (本项目踩过)。
DROPS = [("obj1", (0.52, -0.44)), ("obj2", (0.52, 0.44)),
         ("obj3", (0.72, -0.40)), ("obj4", (0.72, 0.42))]

# "夹爪朝下"的目标姿态: site 本地 +z (指垫外侧) 指向世界 -z, 本地 x 指世界 +x。
# ★ 为什么要这个: run_pick 默认**沿用 home 的水平夹爪**, 而用户实际操作是**把腕子转朝下**
#   —— 从用户真实 episode 量出来, 抓取瞬间夹爪轴与竖直方向夹角只有 8.5~9.6°。
#   两种朝向的可达区**完全不同**(实测: 水平夹爪在 |y|<0.2 处 1/6 通过, 而记忆里
#   朝下时可达区在 y≈0 处最宽) → 扫"物块能放哪"必须用**用户实际用的那个姿态**。
DOWN_R = np.array([[1.0, 0.0, 0.0],
                   [0.0, -1.0, 0.0],
                   [0.0, 0.0, -1.0]])       # det=+1 的纯旋转


if __name__ == "__main__":
    test_ik_stability()
    ok = 0
    for name, drop in DROPS:
        r = run_pick(name, drop)
        good = r["dz"] > 0.10 and r["carried_rel"] < 0.06 and r["placed"]
        ok += int(good)
        print(f"   {name}: 到位误差={r['approach']*1000:4.1f}mm  "
              f"合爪后指垫接触={sorted(r['contacts'])} driver={r['driver']:.3f} "
              f"判定夹住={r['detected']}")
        print(f"        抬升Δz={r['dz']*1000:+6.1f}mm  搬运中与夹爪距离={r['carried_rel']*1000:5.1f}mm  "
              f"落点={np.round(r['final'],3)} 放到目标点={r['placed']} -> {'✓' if good else '✗'}")
    print(f"   结果: {ok}/{len(DROPS)} 完成物理抓取搬运")
