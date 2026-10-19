#!/usr/bin/env python
"""VR 遥操作 (Kinova Gen3 + Robotiq 2F-85 夹爪 / 桌面抓取版)

单手方案 (只用右手柄):
  · 侧键 squeeze (grip):  按住 = 锁定跟随 + 用头显朝向校准参考系; 松开 = 冻结
  · 扳机 trigger:         夹爪开合, 松开全开 → 扣到底闭合 (任何时刻都有效)
  · A / B 键:             机械臂回任务初始位 + 夹爪张开 + 方块归位
  · 左手柄:               不使用

架构 (与原版一致):
  主线程  : aiohttp HTTPS 静态页 + /ws  (端口 8444)
  子线程  : MuJoCo viewer 主循环 → 读共享状态 → 侧键/扳机 → IK → 驱动

IK: SVD 阻尼最小二乘 + 热启动 + 关节速度钳制 (同已验证的原版).
工具点: 不是 pinch site, 而是 TCP = pinch - TCP_OFF*(site z) —— 即机器人两指垫的
        中间, 也就是真正的"抓取中心". 直接拿 pinch site 当工具点的话, 它比指垫还
        靠外 3~4cm, 物体对准它时两指会停在物体上方抓空 (实测).

抓取: **纯物理摩擦** —— 扳机把夹爪合上, 靠接触摩擦夹住物体; 代码不搬动方块,
      只用"左右指垫是否都摸到同一个物体"来判定是否夹住 (对应真机 2F-85 的
      物体检测)。前提是把下面三件事都标定对, 见 calibrate() 的注释。

运行:
  ./run.sh          (推荐, 会做依赖自检)
  或: python -u server.py
Quest 访问:
  https://<本机IP>:8444   (自签证书, 需在头显里点"继续访问")

离线验证: tools/validate_e2e.py (复用本文件的 Rig / ik_solve, 无 viewer 跑完整流程)
"""
import asyncio
import atexit
import collections
import json
import os
import signal
import ssl
import sys
import threading
import time

import numpy as np
import mujoco
import mujoco.viewer  # 必须显式导入
from aiohttp import web, WSMsgType

# 保证从任意工作目录启动都能 import 到同目录的 runlog
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from runlog import RunLogger, file_fingerprint        # noqa: E402
from camview import CameraViewer                      # noqa: E402
from dataset_recorder import DatasetRecorder, GRIP_QMAX   # noqa: E402
from robot_profile import load_profile                # noqa: E402

# ============================================================================
# 机器人配置 —— 想换机械臂只改 robot_profile.py，本文件不用动
# ============================================================================
PROFILE = load_profile()

MODEL_PATH = PROFILE.model_path
HOST = "0.0.0.0"
PORT = int(os.environ.get("VR_TELEOP_PORT", "8444"))     # 可用环境变量覆盖, 便于跑第二个实例
CERT = os.environ.get("VR_TELEOP_CERT", os.path.join(ROOT, "cert.pem"))
KEY = os.environ.get("VR_TELEOP_KEY", os.path.join(ROOT, "key.pem"))

NU_ARM = PROFILE.n_arm_actuators        # 前这么多执行器 = 机械臂关节
IDX_GRIP = PROFILE.gripper.ctrl_index   # 夹爪执行器下标
IDX_BOOST = PROFILE.gripper.boost_ctrl_index   # 可能是 None（没有增压执行器）
NQ_ARM = PROFILE.n_arm_qpos
NQ_GRIP_END = PROFILE.nq_grip_end       # qpos[NQ_ARM:NQ_GRIP_END] = 夹爪关节

# 多臂（单臂时 N_ARMS=1，下面这些不影响原路径）
ARM_SPECS = PROFILE.arm_specs
N_ARMS = len(ARM_SPECS)
DUAL_ARM = N_ARMS > 1
DRIVE_HANDS = tuple(dict.fromkeys(str(s.drive_hand).lower() for s in ARM_SPECS))

# 任务初始位姿 (tools/sweep_gripper_orientation.py + finalize_task_home.py 搜索得到):
#   夹爪"斜朝前 45°"(在竖直朝下与水平朝前之间), TCP 在 (0.48, 0, 0.54)。
# 为什么不是竖直朝下: 夹爪朝下时 j6 ≈ -1.45~-1.9 (近乎 90° 腕弯) 是几何固有的, 而 j6
#   量程只有 ±2.09 → 余量薄, 手一拧就顶死 (实测限位帧 30~125/s, IK 残差峰 169mm,
#   解跳变 15° → 肉眼"一顿一顿")。换朝向实测对比 (邻域 ±0.12m 最差限位余量):
#     竖直朝下 0.334rad | 朝前17° 0.474 | 朝前30° 0.411 | **斜朝前45° 0.533** |
#     较平60° 0.287 | 水平朝前90° 0.050
#   → 45° 最好; 做成完全水平反而最差 (需要另一套腕部构型, 顶上另一个限位)。
# 本姿态下各限位余量: j2=2.085, j4=1.074, j6=1.141 rad; 100/100 邻域可达。
# 注意: 这是配合"位移球 0.22m + 姿态跟随 15°"选出来的 (tools/tune_home_ball.py):
#   球心尽量贴近工作区(三个方块最远 0.198m, 落在球内), 且球内 100% 可达、余量 0.161rad。
# 初始位姿 = **原版(无夹爪)的 keyframe "home"**, 一字不差:
#   夹爪本地 z 轴 = (+1, 0, 0) → 正前方水平朝前, pinch site 在 (0.457, 0, 0.434)。
#   关节余量: j2 1.99 / j4 0.31 / j6 1.14 rad。
# ★ 2026-10-16 用户要求: 初始位置 = run_20261016_122111 日志**最后时刻**的位姿。
#   来源: logs/run_20261016_122111.bin 最后一行 q0..q6 (t=155.55s, 该次运行被 Ctrl-C 结束)。
#   ⚠ 注意: 原 TASK_HOME 是配合"位移球 0.22m + 姿态跟随 15°"专门搜出来的
#     (tools/tune_home_ball.py, 球心贴工作区且球内 100% 可达)。换成下面这个位姿后,
#     **那套标定不再自动成立** —— 若手柄操控感到范围/余量不合适, 需重跑那个工具重定球半径。
#   (j4/j6 是无限旋转关节, 数值 >2π / <0 都合法, 保持原样以保证与日志逐位一致)
# ★ 开源版: 已移到 robot_profile.py 的 home_qpos（换机器人只改那里）
TASK_HOME = np.array(PROFILE.home_qpos, float)

# 抓取中心(两指垫组中心)相对 pinch site 沿本地 -z 的偏移 (m)。
#   实测(厂商版与 menagerie 版结果一致): 两指碰撞垫组中心在 pinch 本地 z = -14.2mm
#   => TCP_OFF = 0.0142。TCP 才是真正的"抓取中心", IK 与遥操作都以它为目标。
#   注意 pinch site 本身在指垫之外 (本地 z=0.145), 直接用它会让指垫停在物体上方。
# ★ 开源版: 已在 robot_profile.py 里（tcp_offset）
TCP_OFF = PROFILE.tcp_offset
GRIP_CENTER_LOCAL = np.array([0.0, 0.0, 0.145 - TCP_OFF])   # 抓取中心在夹爪 base 系

# ---- 执行器/接触标定 (四-2): 让仿真层的驱动方式与真机一致 ----
#   1) 驱动方式: 见下方 DRIVE_MODE (默认 auto: 自由空间瞬移 = 原版手感, 接触切伺服)
#   2) 关节 armature/damping: 模型为 0, 真机有反射惯量与摩擦; 加小值也让接触更稳
#      (armature=0 时接触冲击会出现数百牛的瞬时力)
#   3) 夹爪力: 用"世界体静态方块"当力探针标定 (把自由物体钉住是错的方法, 会产生假力):
#      tendon forcerange 1N → 指垫约 21~28N (四连杆机械增益 ~21~28x, 见
#      tools/calibrate_2f85_force.py)。真机 2F-85 的 rFR 0~255 对应指垫 20~235N,
#      故 forcerange 取 1.0~11.2N 即可覆盖真机全程; 默认 5N ≈ 指垫 128N ≈ 中档 rFR。
ARM_ARMATURE = 0.05
ARM_DAMPING = 0.5
# 注意: 这里**故意不再对 IK 解做单帧限幅**。原版 vr_teleop/server.py 没有这一层,
# IK 解直接写 ctrl; 我早些时候加的 "相对实际位置限 0.05rad/帧" 等于给机械臂加了
# 速度上限, 指令永远只能领先实际 0.05rad → 手一动机械臂就在后面爬
# (实测跟踪误差从 0.01mm 恶化到 51mm)。真机的速度上限本来由执行器力矩天然限制,
# 仿真里同理, 不需要在指令侧再限一次。
# 同理**不要对目标加任何"墙"**: 位移球、姿态角上限、抗积分饱和都试过并已全部撤销 ——
# 手一旦越过墙, 目标就停住、反向一动又跳回来, 这正是"一顿一顿 + 手腕跟随失灵"的
# 直接来源 (见 README "一顿一顿" 一节)。原版没有墙, 手到哪末端到哪。
# ---- 驱动方式 ----
# 这个 MJCF 的臂体执行器**没法调成能精确跟踪快速指令**: 实测快速运动时 j4/j6 力矩
# 打满 (跟随误差 7.1°/9.5° 正是"出够力矩所必需"的误差), 而提 kp 只会让它更早饱和
# (kp×2 立刻发散, j3 跟随误差 370°), 改 dt 完全无效 → 不是数值问题, 是力矩不足。
# 原版 vr_teleop/server.py 用"瞬移"(每帧把 qpos 直接设成 IK 解) 绕过了执行器,
# 所以跟踪恒为 IK 残差 (1.4mm)、手感干脆; 但瞬移会往接触里注入能量, 抓不住东西。
#
# 因此:
#   "teleport" 原版方案: 跟踪最好 (1.4mm), 但物理抓取不成立 (实测夹不住)
#   "servo"    只写 ctrl: 接触力物理真实、能靠摩擦夹住 (实测 3/3), 但快速运动有
#              速度跟随误差 (0.19/0.38/0.75 m/s → 9/15/25 mm)
#   "auto"     无损切换 (默认): 没碰到物体时用瞬移 (手感同原版), 一接触物体就切伺服
#              (接触力物理真实、抓取成立); 松手离开物体后再切回瞬移
# ---- 抓取方案 (两种, 一行切换) ----
#   "physics": 纯物理摩擦。抓取靠接触摩擦, 接触时必须切伺服(否则动力学不成立),
#              于是抓取/抬升阶段带着伺服的速度跟随误差(快速运动 ~25mm) → 手感发钝。
#              数据有效(接触力真实), 供 VLA 微调用。
#   "attach" : 物块挂载(最初的方案)。扳机过阈值且方块在抓取中心附近 → 吸附;
#              搬运期间每帧把方块位姿设成 (夹爪位姿 × 吸附瞬间的相对位姿), 并关掉
#              方块碰撞; 松开扳机恢复物理。
#              **好处: 抓取不再依赖接触物理 → 全程可用瞬移(原版手感), 没有伺服滞后。**
#              代价: 被夹方块相对夹爪位姿恒定(冗余)、接触力恒为零 → 这种数据不适合训 VLA。
GRASP_MODE = "physics"
ATTACH_TRIG = 0.40      # 扳机 > 此值 且 方块在抓取中心附近 → 吸附
DETACH_TRIG = 0.25      # 扳机 < 此值 → 松开 (带迟滞, 防抖)
ATTACH_DIST = 0.060     # 方块中心到抓取中心的距离阈值 (m)

DRIVE_MODE = "auto"
CONTACT_HYST = 20        # (已弃用, 保留兼容) 离开接触后需连续这么多帧无接触才切回瞬移
# ---- 瞬移↔物理 的**连续过渡** (取代按接触硬切换) ----
# 用户实测: "夹爪快碰到物块时会卡住, 卡一会然后下一帧直接瞬移"。
# 根因: 一碰到物块 (touching_object) 就从 teleport 硬切到 servo → 伺服的速度跟随误差
# 瞬间出现并累积 → 看着"卡住"; 离开接触又硬切回 teleport → 突然跳回 = "下一帧直接瞬移"。
# 修法两条:
#   1) 判据从"接触"改成**抓取意图** —— 只有真要夹时 (扣了扳机 / 已经夹住) 才交给物理。
#      单纯靠近/蹭到不切 → 靠近阶段全程瞬移。
#   2) 交接用**连续坡道** blend 0→1 (TELEPORT_RAMP 秒内渐变), 不再有突变点。
PHYS_TRIGGER_MIN = float(PROFILE.teleop.trigger_min)  # 扳机超过此值 → 认为要抓了, 控制权交给物理
TELEPORT_RAMP = 0.12     # 瞬移↔物理 的过渡时长 (秒); 0 = 退化成硬切换
# viewer 同步频率。**必须与原版一致: 每 5 步 (100Hz)**。
# 这里踩过一个坑: 曾经改成"每 8 步 + 墙钟节流 max(1/60, 3*渲染耗时)", 想降到 62.5Hz,
# 结果实测只有 **31.3Hz** —— 第 8 步时距上次同步才 16ms < 节流要求的 16.67ms, 被跳过,
# 只能等到第 16 步, 于是同步间隔变成 32ms。31Hz 刷新率肉眼就是"一卡一卡"
# (原版 100Hz, 间隔标准差 0.21ms; 我那时 31Hz / 1.77ms)。
# 结论: 不要用墙钟节流去"降频", 它会把周期性的节拍变成相位漂移。
# 实测 (tools/bench_sync_jitter.py, 真实 viewer):
#   每5步无节流 → 99.8Hz, 间隔 10.00ms ± 0.09ms ✓
#   每8步+节流   → 31.3Hz, 间隔 31.7ms ± 1.77ms ✗
VIEWER_SYNC_EVERY = 5

# ============================================================================
# 相机视图 (VLA 数据采集看画面用)
# ----------------------------------------------------------------------------
# 把腕部相机(wrist)和全局相机(cam_scene)的视野**各自独立开一个窗口**显示。
# 跑在独立线程 + 独立 MjData: 控制循环只发布 qpos 快照(50Hz), 不阻塞仿真。
# 关掉窗口不影响遥操作; 任何一个环节起不来也只是打印一行, 仿真照常。
#   CAMVIEW_SHOW   总开关
#   CAMVIEW_FPS    刷新率 (20 够看; 调高更占 GPU/CPU)
#   CAMVIEW_SIZE   每个窗口分辨率 (只影响显示, 与模型里相机的 resolution 无关)
#   CAMVIEW_CAMS   要显示的 (窗口标题, 模型里的相机名)
CAMVIEW_SHOW = os.environ.get("VR_TELEOP_CAMVIEW", "1").lower() not in (
    "0", "false", "no", "off")     # 用 VR_TELEOP_CAMVIEW=0 可临时关掉
CAMVIEW_FPS = 20.0
CAMVIEW_SIZE = (640, 480)
CAMVIEW_CAMS = [(c.label, c.model_name) for c in PROFILE.cameras if c.preview]
# 窗口摆放 (x,y)。屏幕 2560x1440。窗口 640x480(+标题栏≈520 高)。
# 原来在 +1700 → MuJoCo 主画面被挡; 用户要求"往右边移动一点" → 挪到最右侧。
# ★ 2026-10-16 加第三个窗口: 三窗竖排要 3x520=1560 > 1440 屏幕高，放不下 ⇒ 改成 2x2，
#   原有两个窗口的位置**不动**，新的"全局俯视"放在它们左边那一格。
CAMVIEW_POS = ["+1904+80", "+1904+620", "+1256+80", "+1256+620"]

# ★★ 2026-10-16: 要关掉的渲染通道（**预览窗口与录数据同时生效**，保证两边画面一致）
#   为什么: 加了第三路相机(cam_top)后，数采从 50Hz 掉到 **37.4Hz** ——
#           MuJoCo 每帧默认要为**阴影**和**反射**各多渲一趟整个场景:
#           640x480 单帧 6.12ms（原样）→ 关阴影 4.06 → 再关反射 **1.81**（三路就 18.4 → 5.4ms）。
#           实测 3 路相机: 无 37.4Hz / 只关反射 45.7 / 只关阴影 49.7 / **两个都关 50.0Hz 慢拍 0**。
#   代价: 画面少一层地面反光 + 少了机械臂/物块投在桌面上的阴影 —— 实测三个视角上差别都很小
#         （对比图 /tmp/camchk2/flags.png）。用户 2026-10-16 选定"两个都关"。
#   注意: 这只影响**画面**（渲染通道），不改相机位姿、不改模型、不改任何 XML。
RENDER_OFF = tuple(PROFILE.render_off)
# 相机偏转: {相机名: 绕该相机**自身 x 轴**的偏转角(度)}。只改内存里的局部四元数,
# **不动任何 XML 文件** (共享的厂商模型也不动)。
# 对腕部相机: + 角度 = 光轴朝夹爪那一侧偏 (夹爪会进入画面); - 角度 = 背向夹爪。
# 效果对照图见 tools/render_wrist_tilt.py (0/+20/-20/+40°)。改 0 即恢复原状。
# ★ 2026-10-14: 腕部/全局相机改用下面的 CAMVIEW_ABS（绝对参数），这里留空。
#    原始值（恢复用）: {"wrist": 45.0}
CAMVIEW_TILT = {}
# 相机位移: {相机名: (dx,dy,dz)} —— 在父体坐标系里的平移(米)。
# 腕部相机的"远离臂"方向 —— 按**垂直于它所在那根臂**来算:
#   相机挂在 bracelet_link, 该连杆的**臂轴 = 本地 z 轴**(实测: 本地 -z = 世界 +x = 末段连杆方向);
#   相机相对连杆原点偏移 (0, -0.0564, -0.0585), 其中**垂直于臂**的分量 = (0, -0.0564, 0)
#   → "垂直于臂且远离臂" 的方向 = **-y**。
#   (试错过: 沿 -z 往前 → 相机跑到两指中间; 沿 +z 往回 → 钻进腕壳画面被堵死; 沿 -x 侧向 → 也不对。
#    对照图见 tools/render_wrist_perp.py / render_wrist_dirs.py)
# ★ 2026-10-14: 改用 CAMVIEW_ABS；原始值（恢复用）: {"wrist": (0.0, -0.12, 0.0)}
CAMVIEW_OFFSET = {}

# ★★ 2026-10-16 新增"全局俯视相机" cam_top（用户要求：在机械臂基座正上方、倾斜俯视）
#   模型里**本来没有**这个相机 —— 用 MjSpec 在**编译前内存注入**（见 camview.load_model_with_cams），
#   一个 XML 文件都不动。这里同时登记进 CAMVIEW_ABS，好让预览窗口和数采走同一条通路。
#   参数怎么来的（不是拍的）:
#     ① 先拿到一张"想要的视角"的截图，
#        其中大窗口（MuJoCo viewer 的自由相机）就是目标。
#     ② 从截图里分割出**桌面**（静态，世界坐标已知: x∈[0.16,1.26] y∈±0.55 台面 z=0.22，
#        见 scene_grasp.xml:144-146）→ 得远边两角 (689,234)/(1390,237) 和左右边缘方向。
#     ③ ★ 坑: 只用桌面（**平面**）特征时，焦距与位姿是**耦合**的 —— 每个 fovy 都能拟合，
#        残差都一样（先试了 25°~77° 全都能拟合到几像素）。必须靠**非平面**特征破歧义。
#     ④ 用**机械臂轮廓**: 截图那一刻机械臂停在 TASK_HOME
#        （日志 run_20261016_170243 全程 tcp=(0.502,-0.180,0.328) 未动），
#        于是拿"渲染出来的臂轮廓"和截图里的臂轮廓比 IoU，扫描 fovy：
#          fovy 25→75 IoU 单调升到 0.35（质心差 4.3px），**77.5° 时 IoU 0.387 / 质心差 0.7px** 后回落
#        → 再用 pattern-search 以 IoU 为目标精修 → **IoU 0.654**。
#     ⑤ 结果: 相机在 (0.062, 0.093, 1.255) —— 离基座水平仅 **11 cm**，即"基座正上方"✓，
#        高 1.26 m、光轴俯角 **72.5°**（很陡的俯视），fovy 79.5°。
#
#   ★★ 2026-10-16 下午用户追加两条要求，改了两处：
#      ① "把 A(现在) 视角整体放大，让桌子的边跟画面边框平行，不要歪"
#         → 用户的真实意图是**桌子的顶边(远边)与画面边框平行**（不是四条边都平行）。
#           几何: 顶边沿世界 y 方向 ⇒ 它要投影成水平线，只需相机光轴**没有左右偏**
#           （相机与瞄准点在同一个 y、且无 roll）。当时相机 y=0.093、瞄准点 y=0 →
#           光轴偏 -y → 顶边歪 **1.1°**。
#           实测量化(投影桌子四角、量各边与画面水平轴的夹角):
#             原参数: 顶边 1.1°、底边 1.1°、左边 8.4°、右边 8.4°（四条都歪）
#             改 y=0 后: 顶边 **0.00°**、底边 0.00°、左右边 1.1°（只有左右有极轻微梯形）
#           ⇒ **y 归零**（相机 y 与瞄准点 y 都取 0，台面中线）即可。
#           注: 想四条边全 0° 必须让光轴垂直向下，但那要相机离开基座上方（挪到台心上方），
#              与"在基座正上方"冲突 —— 用户明确只要顶边平行，所以保持倾斜俯视。
#          ② "相机再靠近点" → 旋钮是**收窄 fovy**，不是放低相机。
#          放低反而更差: 相机越低压得越斜 ⇒ 要装下整桌就得用更广的 fovy ⇒ 方块更小
#          （实测高度 1.255→0.85 时方块 812→564px）。
#          变焦时**远边(顶边)被挤出画面**的界限也量了（远边 y 必须 >0 才看得见平行）:
#             fovy 48.8° 远边y=29 近边y=454 方块 812px  ← 整桌四角都在画面内
#             fovy 46.5° 远边y=18 近边y=466 方块 899px  ← **采用**（最紧且顶边仍可见）
#             fovy 44.5° 远边y= 6 左边缘出画      方块 967px
#             fovy 42.8° 远边y=-4 顶边已挤出画面   方块1057px  ✘ 平行就看不见了
#          选定 46.5°：方块比原来大 3.7 倍，顶边完全平行且可见，机械臂与两套方块布局都在画面内。
# ★ 开源版: 这些值都放在 robot_profile.py（extra_cameras / camera_overrides / render_off），
#   换机器人只需改那里；本文件的注释保留，用作"参考实现为什么这么定"的说明。
CAMVIEW_NEW = dict(PROFILE.extra_cameras)

# ★★ 2026-10-14 相机视角调整（用户要求；**只改内存，一个 XML 都不动**）
#   为什么需要它: 有些调整（改目标点 + 俯角 + fovy）不是"绕自身 x 轴转"能表达的
#   —— 实测 cam_scene 的残差 0.177 ≫ 1e-6，只能直接给绝对位姿。
#   应用位置: camview.CameraViewer 与 dataset_recorder.DatasetRecorder 都会调
#             camview.apply_abs_cams()，所以**预览窗口与录进数据集的画面完全一致**。
CAMVIEW_ABS = dict(PROFILE.camera_overrides)
# ↑ 参考实现的三路相机绝对位姿（wrist / cam_scene / cam_top）在 robot_profile.py 的
#   camera_overrides；下面注释是当初调这些值的过程记录，改机位前值得读一遍。
# 原始值（恢复用，来自 XML 原值）:
#   wrist     pos=(0, -0.05639, -0.058475)  quat=(0,0,0,1)  fovy=41.83792730009236
#   cam_scene pos=(-0.000470, 0.530470, 1.044489)
#             quat=(0.35605539, 0.14025395, -0.33860298, -0.85959375)  fovy=55
# 上一版（第一轮调整，若要回退到那一版）:
#   wrist     pos=(0, -0.17639, -0.058475)  quat=(0,0,-0.42261826,0.90630779)  fovy=75
#   cam_scene pos=(-0.031354, 0.658882, 0.914927)
#             quat=(0.27253201, 0.12708376, -0.4030582, -0.8643611)  fovy=52
#   恢复办法: 把 CAMVIEW_ABS 置 {} 并恢复上面的 CAMVIEW_TILT/CAMVIEW_OFFSET（原值见注释）。

# ============================================================================
# 数据集采集 (LeRobot v2.1 + π0 关节空间 + 50Hz + 双相机) —— 见 dataset_recorder.py
# ----------------------------------------------------------------------------
# 启动时会在终端里**问你本次的任务描述**(就是训练用的语言指令, 写进 tasks.jsonl)。
# 之后: 右手柄侧键 → 开始录一段 → 左手柄**扳机** → 切片 → 左手柄 **X=成功 / Y=失败**
#       → 右手柄侧键 → 开始下一轮, 循环直到关闭程序。
# 数据集**跨次启动累积**(episode 编号接着上次), 全部写进 DATASET_ROOT。
# 关掉: VR_TELEOP_DS=0   换目录: VR_TELEOP_DS_ROOT=...
# 跳过输入(测试用): VR_TELEOP_DS_TASK="把红方块放进盘子"
DATASET_ENABLE = os.environ.get("VR_TELEOP_DS", "1").lower() not in (
    "0", "false", "no", "off")
# ★ 开源版默认落点 = 仓库内 data/lerobot（相对路径，clone 下来就能跑）。
#   可用 VR_TELEOP_DS_ROOT 覆盖（例如指到外置盘）。
DATASET_ROOT = os.environ.get("VR_TELEOP_DS_ROOT",
                              os.path.join(ROOT, "data", "lerobot"))
DATASET_FPS = 50.0              # π0 关节空间的频率 (ARX/Trossen 也是 50Hz)
DATASET_IMAGE = (480, 640)      # (高, 宽)
# 相机列 ← robot_profile.py 的 cameras（只取 record=True 的那几路）
DATASET_CAMS = [(c.lerobot_key, c.model_name) for c in PROFILE.cameras if c.record]
DATASET_MAX_EP_S = 300.0        # 单段上限(秒), 防止忘按切片
DATASET_MIN_EP_S = 0.2          # 短于此的段丢弃(误触)
# 场景几何是静态的 (只有位姿在变), 所以 viewer.sync(state_only=True) 就够:
# 只同步位姿, 跳过几何/材质更新 —— 开销更小。
VIEWER_STATE_ONLY = True

# viewer **主画面**的默认视角: "机械臂和桌子的左上后方" (用户要求, 免得每次用鼠标调)。
# 方位角依据: 臂朝 +x、up=+z → 机器人左手侧 = +y; 相机在 (-x,+y) = 左后方。
# 实测 MjvCamera 映射: az=315 → pos=(-0.46,+1.01,0.82); elev<0 = 相机在高处朝下。
# 数值由 tools/design_viewer_cam.py **算出来**并出图确认 (不是拍的):
#   投影包围盒已居中(偏移<2px)、四边余量 124/120/28/22 px、机械臂+桌子占屏 20.7%。
# 想自己调: 跑那个脚本改 AZIMUTHS/ELEVATIONS 重出对照图。
VIEWER_CAM = dict(azimuth=315.0, elevation=-22.0, distance=1.60,
                  lookat=(0.6034, 0.1090, 0.2024))

# ============================================================================
# 物块随机摆放 (启动时随机一次 + 每次 A/B 回位时重新随机)
# ----------------------------------------------------------------------------
# ★ 区域是**量出来的**, 不是拍的, 而且必须用**用户实际的抓取姿态**去量 ——
#   用户是"夹爪朝下"抓的 (从真实 episode 量: 抓取瞬间夹爪轴与竖直方向只差 8.5~9.6°),
#   而 run_pick 默认沿用 home 的**水平**夹爪。两种朝向的可达区完全不同:
#     水平夹爪: |y|<0.2 处几乎全失败 (1/6 通过), |y|>=0.24 才行;
#     夹爪朝下: |y|<=0.36 基本全通过 (38/42)。
#   我一开始用水平夹爪扫, 得出了"y≈0 不可用"的错误结论 —— 差点按错的区域去设随机。
#
# 实测网格 (tools/sweep_reach_grid.py ... down, 逐点真实 e2e, 夹爪朝下) —— **旧区域**:
#        y:  -0.36 -0.24 -0.12  0.00  0.12  0.24  0.36
#   x=0.38      ✓     ✓     ✓     ✓     ✓     ✓     ✓
#   x=0.42      ✓     ✓     ✓     ✓     ✓     ✓     ✓
#   x=0.46      ✓     ✓     ✓     ✓     ✓     ✓     ✓
#   x=0.50      ✗5    ✓     ✓     ✓     ✓     ✓     ✓      ← (0.50,-0.36) 抬不起来
#   x=0.34      ...                                       ← (0.34,+0.36) 抬不起来
#   x=0.54      ✗0    ✓     ✓     ✓     ✓     ✓     ✗0     ← 外侧极限
# → 当时取 x∈[0.38,0.48] y∈[-0.32,0.32]。
#
# ★ 2026-10-16 用户要求「方块集中一点, 别摆成一条线, 在中间一小片里完全随机」。
#   旧区域是 0.10 x 0.64 m —— **又窄又长**, 4 个方块必然沿 y 排成一条线, 就是这个毛病。
#   新区域改成 **26 x 30 cm 的方形块**, 且中心往臂根方向挪 (x 中心 0.43→0.37)。
#
#   方块能往内(近臂根)放到多小? 旧网格只测到 x=0.38, 没测过更内侧, 所以重新量:
#     (a) 先用 IK 探针 (夹爪朝下, z=0.45→0.24 整链) 粗扫: 中央 x∈[-0.10,0.50]
#         y∈±0.40 几乎全可达, x 小到 0.18 仍有余量 —— 说明**瓶颈不是内侧**;
#     (b) 再用**真实 e2e** 验: tools/validate_random_layout.py 6 0.20 0.52 -0.16 0.16 0.12
#         → **6/6 组布局全部通过, 24/24 次抓取成功**(含 x=0.227 / y=±0.159 这些边界点,
#           到位残差一律 0.0mm, 抬升 Δz 160~165mm)。
#   ⇒ 新区域四周各留 0.01~0.03m 余量, 落在上面验证过的范围内。
#
#   四个方块**同时**随机的布局仍由 tools/validate_random_layout.py 直接验 (见 README)。
OBJ_RANDOM = os.environ.get("VR_TELEOP_OBJ_RANDOM", "1").lower() not in (
    "0", "false", "no", "off")
OBJ_RANDOM_X = PROFILE.object_random_x   # 世界 x 范围 (米)；换机器人时按自己可达区改 robot_profile.py
OBJ_RANDOM_Y = PROFILE.object_random_y   # 世界 y 范围 (米)
#   ★ 这是**实测可达边界内能取到的、最靠桌面中心**的矩形（中心 x=0.54）。
#     桌面几何中心在 x=0.71，**够不到**（见下）。
#   ★ 边界是怎么测出来的（都用 tools/validate_random_layout.py 真实 e2e，夹爪朝下）:
#       x(0.42,0.60) y(±0.16)   → 4/4 组、16/16 抓取 全通过
#       x(0.46,0.62) y(±0.155)  → 6/6 组、24/24 抓取 全通过  ★采用
#       x(0.58,0.64) y(±0.20)   → 开始出现在 x≥0.61 且 |y|≥0.18 的点抬不起来(Δz<100mm)
#       x(0.62,0.70)            → 全挂(Δz 12~80mm，"夹得住抬不起来")
#   ★★ 我自己之前两次把这里判错，记下来:
#      (1) 先用"只从 TASK_HOME 单点迭代 150 次"的探针，得出"x 只能到 0.54~0.56" ——
#          **错的**。e2e 的 `goto` 是**沿轨迹插值 + 每步热启动**，能一路挪到 x=0.62。
#          单点迭代不收敛不代表整条轨迹到不了。
#      (2) 又把 16x26cm + 间距 0.12m 当成可行区域 —— 那个形状下 4 个方块两两≥0.12m
#          在某个随机种子下**300 万次都采不出来**，会把 validate_random_layout.py 卡死
#          （已修该文件的 sample()：每点限次、超了整组重来）。
#          教训: 区域太扁时"能采出 4 个点"本身就是一个约束，不只是可达性。
OBJ_MIN_SEP = PROFILE.object_min_sep   # 方块中心最小间距 (给夹爪接近留余地)
OBJ_Z = PROFILE.object_z               # 台面 z + 方块半高
OBJ_AVOID_TCP = 0.13             # 生成点要离**当前** TCP 至少这么远 (防生成在夹爪里)

GRIP_FORCE_RANGE = 5.0   # 肌腱力上限; 5N ≈ 指垫 128N ≈ 真机 rFR 中档 (标定见上)

# ============================================================================
# 运行日志 (每次运行一个文件, 事后把文件发来即可定位问题)
# ----------------------------------------------------------------------------
# 采样 250Hz (每 2 步一行): 500Hz 全量太占空间, 100Hz 又看不出单帧跳变, 250Hz 折中。
# 约 0.6 MB/分钟; 10 分钟 ≈ 6 MB, 压缩后更小。
# 一共生成三个文件 (同名不同后缀):
#   run_<时间>.bin   二进制, 每行 ncol 个 float32, 无表头 (列名见 .json 的 columns)
#   run_<时间>.json  列定义 + 配置快照 + 源码指纹 + 结束摘要  ← 先看这个
#   run_<时间>.log   文本事件流 (启动/回位/抓取/异常), 读起来最快
# 目录: logs/
LOG_ENABLE = True
LOG_DIR = os.environ.get("VR_TELEOP_LOGDIR", os.path.join(ROOT, "logs"))
LOG_EVERY_STEPS = 2          # 每几帧采一行 (2 → 250Hz)
LOG_GZIP = False             # True 则关闭时把 .bin 压成 .bin.gz

# 要记录的列 (顺序即 .bin 里的列顺序)
LOG_COLUMNS = (
    ["t", "step_ms"]
    + [f"q{i}" for i in range(7)]        # 实际关节角 (rad)
    + [f"qd{i}" for i in range(7)]       # 关节速度 (rad/s) —— 看抽风最直接
    + [f"ctrl{i}" for i in range(7)]     # 下发给执行器的位置指令 (rad)
    + [f"qik{i}" for i in range(7)]      # IK 原始解 (未做任何限幅)
    + ["tgt_x", "tgt_y", "tgt_z"]        # 指令目标 (TCP 位置, m)
    + ["tgt_qx", "tgt_qy", "tgt_qz", "tgt_qw"]
    + ["tcp_x", "tcp_y", "tcp_z"]        # 实际末端 (TCP 位置, m)
    + ["tcp_qx", "tcp_qy", "tcp_qz", "tcp_qw"]
    + ["ik_pos_err", "ik_rot_err", "ik_manip"]   # IK 残差(m)/(rad) 与可操作度
    + ["drive_code", "blend"]            # 驱动方式(0瞬移/1伺服/2物理) 与交接权重
    + ["grip_cmd", "grip_driver", "grip_held"]   # 夹爪指令 / 夹爪关节角 / 是否夹住(0/1)
    + ["trigger", "squeeze", "locked"]   # 手柄输入
    + ["hand_x", "hand_y", "hand_z",     # 手柄原始位姿 (位置+四元数,
       "hand_qx", "hand_qy", "hand_qz", "hand_qw",   # 四元数是回放必需的
       "hand_age_ms"]        # ★ 距最近一次**收到**新手柄消息的毫秒数
    + ["loop_ms", "sync_ms", "reset_anim"]       # 耗时与状态
)


def log_meta():
    """日志文件开头要写进去的"这是哪一版/什么配置"。"""
    return {
        "variant": "单手抓取版",
        "model": MODEL_PATH,
        "fingerprint": {
            "server.py": file_fingerprint(os.path.abspath(__file__)),
            "scene_grasp.xml": file_fingerprint(MODEL_PATH),
            "runlog.py": file_fingerprint(os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "runlog.py")),
        },
        "config": {
            "GRASP_MODE": GRASP_MODE, "DRIVE_MODE": DRIVE_MODE,
            "SCALE": SCALE, "TCP_OFF": TCP_OFF, "GRIP_FORCE_RANGE": GRIP_FORCE_RANGE,
            "AXIS_SIGNS": list(AXIS_SIGNS), "ROT_SIGNS": list(ROT_SIGNS),
            "VIEWER_SYNC_EVERY": VIEWER_SYNC_EVERY, "FILTER_ENABLE": FILTER_ENABLE,
            "PHYS_TRIGGER_MIN": PHYS_TRIGGER_MIN, "TELEPORT_RAMP": TELEPORT_RAMP,
            "IK_LAMBDA": IK_LAMBDA, "IK_MAX_DQ": IK_MAX_DQ, "IK_MAX_ITERS": IK_MAX_ITERS,
            "ARM_ARMATURE": ARM_ARMATURE, "ARM_DAMPING": ARM_DAMPING,
            "TASK_HOME": [float(x) for x in TASK_HOME],
        },
    }

# ---- 手抖滤波 (一欧元滤波 / 1€ filter) —— **默认关闭** ----
# 2026-10-08: 实现了并实测有效 (抖动 5.75mm → 0.70mm, 滤除率 84~88%, 频域上有意运动
# 几乎无损), 但**用户实测后认为不如不加, 已默认关闭**。
#
# 为什么"有效"却"手感更差" —— 教训记录在这里, 免得以后又绕回来:
#   实测快速运动时的跟随误差:  无滤波 5.75mm  →  一欧元 22.52mm。
#   即**为了滤掉 5.75mm 的抖动, 代价是快速移动时多出 17mm 的滞后**。
#   遥操作要精确定位, 滞后(尤其是"到位时机"被推迟)比抖动更难受, 而且操作者
#   自己会在需要精细对准时自然稳手。所以这类任务上**不滤更划算**。
#   (一欧元已经是同类里延迟最低的了 —— 同样 0.66mm 抖动时固定低通跟随误差 24/92mm,
#    一欧元只有 8.6/30mm。即便如此仍然不划算。)
#
# 想启用: 把 FILTER_ENABLE 改 True, 并按需调 min_cutoff/beta (见 README)。
FILTER_ENABLE = False
FILTER_MIN_CUTOFF = 1.0    # 手"静止/慢动"时的截止频率 (Hz): 越小滤得越狠
FILTER_BETA = 1.0          # 速度自适应强度: 越大 → 快速运动越不延迟
FILTER_D_CUTOFF = 1.0      # 速度估计自身的低通截止 (Hz)
FILTER_ROT_MIN_CUTOFF = 1.0
FILTER_ROT_BETA = 1.0
HAND_LOG = ""              # 填路径(如 /tmp/hand.csv)则记录 原始/滤波后 手柄位姿, 用于按真实手抖调参


def calibrate(model):
    """把模型改造成"伺服可跟、接触量级合理"的标定态 (不改动厂商 XML, 运行时生效)。"""
    if DUAL_ARM:
        # 每条臂按自己的 dof / 执行器区间标定
        for s in ARM_SPECS:
            j0 = int(s.joint_offset)
            d0 = int(model.jnt_dofadr[j0])
            nq = int(s.n_arm_qpos)
            model.dof_armature[d0:d0 + nq] = ARM_ARMATURE
            model.dof_damping[d0:d0 + nq] = ARM_DAMPING
            a0 = int(s.actuator_offset)
            if int(s.n_gripper_qpos) > 0:      # 没有夹爪的臂跳过
                gi = a0 + int(s.gripper.ctrl_index)
                model.actuator_forcerange[gi] = [-GRIP_FORCE_RANGE, GRIP_FORCE_RANGE]
                bi = s.gripper.boost_ctrl_index
                if bi is not None:
                    model.actuator_forcerange[a0 + int(bi)] = [0.0, 0.0]
        return
    model.dof_armature[:NQ_ARM] = ARM_ARMATURE
    model.dof_damping[:NQ_ARM] = ARM_DAMPING
    model.actuator_forcerange[IDX_GRIP] = [-GRIP_FORCE_RANGE, GRIP_FORCE_RANGE]
    if IDX_BOOST is not None:
        model.actuator_forcerange[IDX_BOOST] = [0.0, 0.0]

# 手柄(VR)方向 → 机械臂世界系 —— ★ 全部来自 robot_profile.py 的 teleop（可完全自定义）
R_MAP = np.asarray(PROFILE.teleop.rot_map, float)
AXIS_SIGNS = np.asarray(PROFILE.teleop.axis_signs, float)  # 平移 [前后, 左右, 上下]
ROT_SIGNS = np.asarray(PROFILE.teleop.rot_signs, float)    # 旋转 [pitch, yaw, roll]

SCALE = float(PROFILE.teleop.scale)                        # 手柄移 1m → 末端移 SCALE 米
GRIP_THRESH = float(PROFILE.teleop.squeeze_thresh)         # 侧键按下阈值 (锁定跟随)
GRIP_RATE = float(PROFILE.teleop.gripper_rate) * 255.0     # 夹爪命令每步最大变化 (0~255 刻度)
FOLLOW_ORI = bool(PROFILE.teleop.follow_orientation)        # False = 只跟位置
DRIVE_HAND = str(PROFILE.teleop.drive_hand).lower()        # 哪只手驱动机械臂
LABEL_HAND = "left" if DRIVE_HAND == "right" else "right"  # 另一只手负责采集按键
BOOST_CMD = 0.0         # 备用并联夹持力 (0~1); 标定后主执行器已够, 默认不出力


def set_boost(ctrl):
    """写"增压执行器"的指令；没有这个执行器（None）时什么也不做。

    ★ 不能写成 ctrl[IDX_BOOST]，IDX_BOOST 为 None 时那会变成 ctrl[-1]
      —— 对别的机器人就是"把最后一根关节改成 0"，非常隐蔽的 bug。
    """
    if IDX_BOOST is not None and 0 <= IDX_BOOST < ctrl.shape[0]:
        ctrl[IDX_BOOST] = BOOST_CMD

IK_LAMBDA = PROFILE.ik_damping
IK_MAX_DQ = PROFILE.ik_max_dq
IK_MAX_ITERS = PROFILE.ik_iterations
IK_TOL = PROFILE.ik_pos_tol
IK_LAST_ERR = 0.0       # 上一次 ik_solve 的综合残差
IK_LAST_POS = 0.0       # 位置残差 (m)
IK_LAST_ROT = 0.0       # 姿态残差 (rad)
# ---- 抗积分饱和 (anti-windup) ----
# 手伸到够不到的地方时, 目标是"够不到就越来越偏"的: IK 残差持续存在、臂卡住不动,
# 手一收回来目标又瞬间跳回可达区 → 解突变 (肉眼就是"顿一下再窜一下")。


# ---------------- 四元数工具 (xyzw) ----------------
def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw*bx + ax*bw + ay*bz - az*by,
                     aw*by - ax*bz + ay*bw + az*bx,
                     aw*bz + ax*by - ay*bx + az*bw,
                     aw*bw - ax*bx - ay*by - az*bz])


def quat_inv(q):
    q = np.asarray(q, float)
    return np.array([-q[0], -q[1], -q[2], q[3]]) / (q @ q)


def quat_to_mat(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
                     [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])


def mat_to_quat(R):
    R = np.asarray(R)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        q = np.array([(R[2,1]-R[1,2])/S, (R[0,2]-R[2,0])/S, (R[1,0]-R[0,1])/S, 0.25*S])
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            S = np.sqrt(1 + R[0,0] - R[1,1] - R[2,2]) * 2
            q = np.array([0.25*S, (R[0,1]+R[1,0])/S, (R[0,2]+R[2,0])/S, (R[2,1]-R[1,2])/S])
        elif i == 1:
            S = np.sqrt(1 + R[1,1] - R[0,0] - R[2,2]) * 2
            q = np.array([(R[0,1]+R[1,0])/S, 0.25*S, (R[1,2]+R[2,1])/S, (R[0,2]-R[2,0])/S])
        else:
            S = np.sqrt(1 + R[2,2] - R[0,0] - R[1,1]) * 2
            q = np.array([(R[0,2]+R[2,0])/S, (R[1,2]+R[2,1])/S, 0.25*S, (R[1,0]-R[0,1])/S])
    q = np.array(q, float)
    return q / np.linalg.norm(q)


def quat_log(q):
    """四元数对数映射 (短路径) → 轴角向量; 180° 处无除零。"""
    x, y, z, w = q
    v = np.array([x, y, z])
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return np.zeros(3)
    if w < 0:
        v = -v
        w = -w
    return 2.0 * np.arctan2(nv, w) * v / nv


# ---------------- 一欧元滤波 (1€ filter) ----------------
def _lp_alpha(cutoff, dt):
    """一阶低通的平滑系数。cutoff 可为标量或数组 (逐分量不同截止)。"""
    tau = 1.0 / (2.0 * np.pi * np.maximum(cutoff, 1e-6))
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """标量/向量通用的一欧元滤波。

    x_hat_prev = 上一时刻滤波值, dx_hat = 低通后的"速度"。
    截止频率 = min_cutoff + beta*|dx_hat|: 手抖(往复, 速度均值≈0)→ 强滤;
    有意运动(持续速度)→ 截止抬高 → 低延迟。
    """

    def __init__(self, min_cutoff=1.0, beta=1.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = beta
        self.d_cutoff = float(d_cutoff)
        self.x_hat = None
        self.dx_hat = None

    def reset(self):
        self.x_hat = None
        self.dx_hat = None

    def __call__(self, x, dt):
        x = np.asarray(x, float)
        # 防 NaN/Inf 污染滤波器状态 (一旦污染就再也恢复不了)
        if not np.all(np.isfinite(x)):
            return (np.zeros_like(x) if self.x_hat is None else self.x_hat.copy())
        if self.x_hat is None:
            self.x_hat = x.copy()
            self.dx_hat = np.zeros_like(x)
            return x.copy()
        dx = (x - self.x_hat) / dt
        a_d = _lp_alpha(self.d_cutoff, dt)
        self.dx_hat = a_d * dx + (1.0 - a_d) * self.dx_hat
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx_hat)
        a = _lp_alpha(cutoff, dt)
        self.x_hat = a * x + (1.0 - a) * self.x_hat
        return self.x_hat.copy()


class OneEuroQuat:
    """四元数版一欧元滤波 (绕"到原始值的旋转轴"平滑, 不做逐分量插值)。

    每帧算出 滤波值→原始值 的增量旋转 dq(角度 ang), 取自适应系数 a,
    就只施加 a 比例的那个旋转。角度小时与线性滤波等价, 且自动走短路径。
    """

    def __init__(self, min_cutoff=1.0, beta=1.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.q_hat = None
        self.spd_hat = 0.0

    def reset(self):
        self.q_hat = None
        self.spd_hat = 0.0

    def __call__(self, q, dt):
        q = np.asarray(q, float)
        # 防 NaN/Inf / 零四元数污染状态
        nrm = float(np.linalg.norm(q)) if np.all(np.isfinite(q)) else 0.0
        if nrm < 1e-9:
            return (np.array([0.0, 0.0, 0.0, 1.0]) if self.q_hat is None
                    else self.q_hat.copy())
        q = q / nrm
        if self.q_hat is None:
            self.q_hat = q.copy()
            return q.copy()
        dq = quat_mul(q, quat_inv(self.q_hat))
        if dq[3] < 0.0:                      # 短路径
            dq = -dq
        ang = 2.0 * np.arccos(np.clip(dq[3], -1.0, 1.0))
        spd = ang / dt
        a_d = _lp_alpha(self.d_cutoff, dt)
        self.spd_hat = a_d * spd + (1.0 - a_d) * self.spd_hat
        cutoff = self.min_cutoff + self.beta * self.spd_hat
        a = float(_lp_alpha(cutoff, dt))
        axis = dq[:3]
        n = float(np.linalg.norm(axis))
        if n > 1e-12 and a > 1e-9:
            half = 0.5 * a * ang
            step = np.concatenate([axis / n * np.sin(half), [np.cos(half)]])
            self.q_hat = quat_mul(step, self.q_hat)
            self.q_hat /= np.linalg.norm(self.q_hat)
        return self.q_hat.copy()


def S_quat_wxyz(q_xyzw):
    """xyzw → wxyz (MuJoCo qpos 里自由关节四元数是 wxyz 顺序)。"""
    x, y, z, w = q_xyzw
    return np.array([w, x, y, z])


def angle_wrap(d):
    return np.arctan2(np.sin(d), np.cos(d))


# ---------------- 运行诊断 (逐秒汇总, 开销可忽略) ----------------
class Diag:
    """定位"一顿一顿/奇异点"这类问题的仪器。

    统计口径:
      hz        实际仿真步频 (需 ≥ 500 才算跟得上实时)
      late      本秒有多少帧没睡够 (即跟不上、在追赶) —— 直接对应"卡/顿"
      sync      同步频率 Hz(±间隔标准差ms)。**标准差就是肉眼的一卡一卡**;
                原版节奏 = 100Hz ± 0.2ms。曾经因墙钟节流掉到 31Hz ± 1.8ms = 明显顿
      ikerr     IK 解与目标的残差峰值 (mm); 大 = 目标到不了(可能撞限位/奇异)
      dq        相邻两帧 IK 解的关节最大变化 (°); 大 = 解在跳(分支切换/奇异)
      lim       本秒有多少帧有受限期关节贴到限位
    """
    def __init__(self):
        self.sync_last = None   # 上一次 viewer.sync 的时刻 (算间隔用), reset 不清
        self.reset()

    def reset(self):
        self.t0 = time.time(); self.steps = 0; self.late = 0
        self.sync_sum = 0.0; self.sync_max = 0.0; self.sync_n = 0
        self.pos_max = 0.0; self.rot_max = 0.0
        self.dq_max = 0.0; self.lim = 0; self.lim_j = []
        # 只清空间隔样本, 保留 sync_last —— 否则每秒的第一个间隔算不出来
        self.sync_iv = []
        # 手抖滤波: 每秒统计"从原始手柄输入里滤掉的高频量"(mm)
        self.hand_n = 0; self.hand_sum = 0.0; self.hand_max = 0.0
        # 网络质量: 头显发来的位姿消息的到达间隔
        self.net_n = 0; self.net_sum = 0.0; self.net_max = 0.0; self.net_late = 0

    def net_seen(self, gap):
        """记一次"头显消息到达间隔"(秒)。gap>1s 视为断流/空闲, 不计入抖动统计。"""
        if gap <= 0.0 or gap > 1.0:
            return
        self.net_n += 1
        self.net_sum += gap
        if gap > self.net_max:
            self.net_max = gap
        if gap > 0.05:          # 50ms = 3 个 72Hz 周期, 明显的一次卡顿
            self.net_late += 1

    def hand_seen(self, mm):
        """记录一次 |滤波前 - 滤波后| (mm)。手抖越厉害/滤波越强, 这个数越大。"""
        self.hand_n += 1
        self.hand_sum += mm
        if mm > self.hand_max:
            self.hand_max = mm

    def tick(self, sync_cost, pos_err, rot_err, dq, at_limit):
        """pos_err 单位 m, rot_err 单位 rad, dq 单位 rad。"""
        self.steps += 1
        if sync_cost is not None:
            self.sync_sum += sync_cost; self.sync_max = max(self.sync_max, sync_cost)
            self.sync_n += 1
        if pos_err is not None:
            self.pos_max = max(self.pos_max, pos_err)
        if rot_err is not None:
            self.rot_max = max(self.rot_max, rot_err)
        if dq is not None:
            self.dq_max = max(self.dq_max, dq)
        if at_limit:
            self.lim += 1
            self.lim_j = at_limit

    def sync_seen(self):
        """每次调用 viewer.sync 后调一次, 记录与上次的间隔 (秒)。

        **间隔的标准差就是肉眼看到的'一卡一卡'**: 均值再高, 只要间隔抖得厉害
        (比如 10/32ms 交替), 看起来就是顿。原版 100Hz 的间隔标准差只有 0.2ms。
        """
        t = time.perf_counter()
        if self.sync_last is not None:
            self.sync_iv.append(t - self.sync_last)
        self.sync_last = t

    def maybe_report(self, locked):
        el = time.time() - self.t0
        if el < 1.0:
            return None
        self.t0 = time.time()
        if self.sync_iv:
            _n = len(self.sync_iv)
            _hz = _n / sum(self.sync_iv)
            _std = float(np.std(self.sync_iv)) * 1000
            _sync = f"同步={_hz:5.1f}Hz(±{_std:.2f}ms) "
        else:
            _sync = "同步=  -- "
        # 网络质量: 头显来包频率 + 到达间隔 (均值/峰) + 明显卡顿次数
        #   Quest 1 刷新率 72Hz → 正常 30~40Hz(单手柄消息) 间隔约 13~33ms;
        #   峰间隔上百 ms = 丢包/省电休眠, 手上会被当成"一顿一顿"。
        if self.net_n:
            _net = (f"网络={self.net_n/el:5.1f}Hz "
                    f"间隔={self.net_sum/self.net_n*1000:4.1f}/{self.net_max*1000:5.1f}ms")
            if self.net_late:
                _net += f"(>50ms×{self.net_late})"
            _net += " "
        else:
            _net = "网络=   -- (无数据) "
        s = (
            # ---- 恢复被注释掉的关键字段 (2026-10-10) ----
            # 之前有人把这几行注释掉了(只留 解跳变/限位帧/手抖滤除/数采), 但按本项目自记的
            # 教训, 这里有 4 项是**硬指标**, 注释掉会让现场排查失去入口:
            #   ① 步频/跟不上 —— 判断"卡没卡"的唯一硬指标 (正常 500.0/s、跟不上=0)
            #   ② 同步Hz±ms   —— 远端实锤过的坑: 31Hz/σ1.77ms 就是肉眼"一卡一卡"
            #   ③ 网络Hz/间隔 —— 用户当初**专门要求**加的 (不戴头显看链路质量)
            #   ④ 位置/姿态残差 —— 分开报(曾经混量纲导致误读 "2944mm")
            # 只把 渲染ms 省了 (实测恒 0.03ms, 从来没成为瓶颈; 要时再打开下面那行)。
            f"[DIAG] 步频={self.steps/el:6.1f}/s late={self.late:3d} "
             f"{_sync}{_net}"
            # f"渲染{'--' if not self.sync_n else f'{self.sync_sum/self.sync_n*1000:.2f}'}"
            #  f"/{'--' if not self.sync_n else f'{self.sync_max*1000:.1f}'}ms "
             f"位置残差={self.pos_max*1000:6.1f}mm 姿态残差={np.degrees(self.rot_max):5.1f}° "
             f"解跳变={np.degrees(self.dq_max):5.1f}° "
             f"限位帧={self.lim:3d}"
             + (f" {self.lim_j}" if self.lim_j else "")
             + (f" 手抖滤除={self.hand_sum/self.hand_n:4.1f}/{self.hand_max:4.1f}mm"
                if self.hand_n else "")
             + (f" 数采={self.ds}" if getattr(self, "ds", "") else ""))
        self.reset()
        return s


# ---------------- 共享状态 ----------------
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.right = None
        self.left = None
        self.locked = False
        self.hand0_pos = None
        self.hand0_quat = None
        self.tcp0_pos = None
        self.ee0_quat = None
        self.target_tcp = None
        self.target_quat = None
        self.reset_requested = False
        self.grip_held = None
        self.hand_history = collections.deque(maxlen=300)
        self.hand_seq = 0        # 每收到一帧新手柄位姿 +1 (用于判断"是否新样本")
        self.hand_t = 0.0        # 最近一次收到新手柄位姿的时刻 (测输入延迟)

    def update_hands(self, msg):
        with self.lock:
            if 'right' in msg:
                self.right = msg['right']
            if 'left' in msg:
                self.left = msg['left']
            # 延迟/新鲜度只看"驱动手"（另一只手只发按键，没有连续位姿）
            # 双臂时两只手都是驱动手（DRIVE_HANDS 有两项），每条消息只计一次。
            for _h in DRIVE_HANDS:
                drv = msg.get(_h)
                if drv is None:
                    continue
                p = drv.get('pos')
                q = drv.get('quat')
                if p is not None and q is not None:
                    self.hand_history.append((list(p), list(q)))
                    self.hand_seq += 1
                    self.hand_t = time.time()
                    break

    def request_reset(self):
        with self.lock:
            self.reset_requested = True


# ---------------- IK ----------------
def clip_arm(q, model, jnt_ids=None):
    """把关节角裁到限位内（无限旋转关节不裁）。jnt_ids=None = 第 0 条臂的关节。"""
    js = list(range(NQ_ARM)) if jnt_ids is None else list(jnt_ids)
    return np.where(model.jnt_limited[js],
                    np.clip(q, model.jnt_range[js, 0], model.jnt_range[js, 1]), q)


def ik_solve(model, data, site_id, tcp_target, target_quat, q_seed=None,
             q_slice=None, d_slice=None, jnt_ids=None, tcp_off=None):
    """7 关节 DLS 阻尼最小二乘 IK (与原版 vr_teleop/server.py 完全同一套算法)。

    与原版唯一的有意差异: 计算过程中会**临时**写 data.qpos 求 Jacobian,
    返回前把状态还原, 只把关节解交给调用方写 ctrl —— 由位置执行器伺服过去,
    而不是把 qpos 瞬移过去 (瞬移会让接触求解出现非物理的巨大力)。
    除此之外 IK 数学、迭代次数、逐次限幅、关节限位处理都与原版一致。

    q_seed: 热启动种子。主循环应传**上一帧的解** (与原版"求解后不还原 qpos,
    下一帧自然从上一帧解起算"等价)。传 None 则从当前物理状态起算。

    多臂参数（不传 = 第 0 条臂，与单臂行为逐字相同）:
      q_slice  这条臂在 data.qpos 里的切片
      d_slice  这条臂的关节在 Jacobian 列上的切片（dof 下标）
      jnt_ids  这条臂的关节号（用于限位裁剪）
      tcp_off  这条臂的抓取中心偏移
    """
    q_sl = slice(0, NQ_ARM) if q_slice is None else q_slice
    d_sl = slice(0, NQ_ARM) if d_slice is None else d_slice
    js = list(range(NQ_ARM)) if jnt_ids is None else list(jnt_ids)
    off = TCP_OFF if tcp_off is None else float(tcp_off)

    q_save = data.qpos.copy()
    q = (np.asarray(q_seed, float).copy() if q_seed is not None
         else data.qpos[q_sl].copy())
    # 种子也要先落到模型里, 否则第一次迭代的 Jacobian/误差是用没同步的状态算的
    data.qpos[q_sl] = q
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    z = data.site_xmat[site_id].reshape(3, 3)[:, 2]
    e = np.zeros(6)
    for _ in range(IK_MAX_ITERS):
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J = np.vstack([jacp, jacr])[:, d_sl]

        ep = (tcp_target + off * z) - data.site_xpos[site_id]
        q_cur = mat_to_quat(data.site_xmat[site_id].reshape(3, 3))
        eo = quat_log(quat_mul(target_quat, quat_inv(q_cur)))
        e = np.concatenate([ep, eo])
        if np.linalg.norm(e) < IK_TOL:
            break

        U, S, Vt = np.linalg.svd(J, full_matrices=False)
        J_pinv = Vt.T @ np.diag(S / (S**2 + IK_LAMBDA**2)) @ U.T
        dq = np.clip(J_pinv @ e, -IK_MAX_DQ, IK_MAX_DQ)

        q = clip_arm(q + dq, model, js)
        data.qpos[q_sl] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        z = data.site_xmat[site_id].reshape(3, 3)[:, 2]
    # 还原物理状态: 只把 q 交给调用方写 ctrl (伺服驱动, 不瞬移)
    data.qpos[:] = q_save
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    # 记录本次解的残差 (诊断用)。
    # 注意: 位置是米、姿态是弧度, 必须分开报 —— 之前把两者范数合成一个数当"mm"看,
    # 1 rad 的姿态误差会被显示成 1000mm, 极易误判。
    global IK_LAST_ERR, IK_LAST_POS, IK_LAST_ROT
    IK_LAST_POS = float(np.linalg.norm(e[:3]))
    IK_LAST_ROT = float(np.linalg.norm(e[3:]))
    IK_LAST_ERR = float(np.hypot(IK_LAST_POS, IK_LAST_ROT))
    return q


# ---------------- 手柄 → 末端目标 映射 ----------------
# ---------------- 手柄 → 末端目标 映射 ----------------
def map_hand_to_target(p, q, hand0_pos, hand0_quat, tcp0, ee0_quat):
    """全量相对位姿映射 —— **与原版 vr_teleop/server.py 逐字一致, 不加任何限幅**。

    之前我在这里加了"位移不超过 0.22m / 转角不超过 15°"两道限幅, 那是错的:
    手一旦越过限幅边界, 目标就停住不动, 反向一动又瞬间跳回来 —— 这正是
    "一顿一顿"和"腕部姿势跟随不对"的直接来源。原版没有限幅, 手到哪末端到哪。
    """
    dp_vr = np.asarray(p, float) - np.asarray(hand0_pos, float)
    dp_arm = AXIS_SIGNS * (R_MAP @ dp_vr) * SCALE
    target_tcp = np.asarray(tcp0, float) + dp_arm

    dq_vr = quat_mul(np.asarray(q, float), quat_inv(np.asarray(hand0_quat, float)))
    dq_vr = np.array([ROT_SIGNS[0]*dq_vr[0], ROT_SIGNS[1]*dq_vr[1],
                      ROT_SIGNS[2]*dq_vr[2], dq_vr[3]])
    if not FOLLOW_ORI:
        # 只跟位置：姿态保持基准（抓取瞬间的朝向）。
        # 腕部自由度不够 / 不需要翻转手腕的任务用这个，IK 会轻松很多。
        return target_tcp, mat_to_quat(quat_to_mat(np.asarray(ee0_quat, float)))
    R_rel = R_MAP @ quat_to_mat(dq_vr) @ R_MAP.T
    target_quat = mat_to_quat(R_rel @ quat_to_mat(np.asarray(ee0_quat, float)))
    return target_tcp, target_quat


# ---------------- 多臂运行时 ----------------
class ArmRuntime:
    """一条臂的全部运行时状态：索引、跟随锁定、IK 热启动、夹爪。

    单臂时这个类只被间接使用（单臂主循环仍走原来的全局变量路径，行为不变）；
    双臂时每条臂一个实例，主循环对它们逐个跑同一套"读手柄 → 映射 → IK → 下发"。

    索引来源：`ArmSpec.joint_offset` / `actuator_offset`，qpos 与 dof 地址从
    `model.jnt_qposadr / jnt_dofadr` 推出 —— 不假设关节在模型里连续。
    """

    def __init__(self, spec, model, data):
        self.spec = spec
        self.model = model
        self.data = data
        self.name = spec.name
        self.hand_name = str(spec.drive_hand).lower()
        if self.hand_name not in ("left", "right"):
            raise ValueError(f"臂 '{spec.name}': drive_hand 必须是 'left' 或 'right'，"
                             f"当前 = {spec.drive_hand!r}")

        self.nq = int(spec.n_arm_qpos)
        self.nu = int(spec.n_arm_actuators)
        j0 = int(spec.joint_offset)
        if j0 < 0 or j0 + self.nq > model.njnt:
            raise ValueError(f"臂 '{spec.name}': joint_offset={j0} + n_arm_qpos={self.nq} "
                             f"超出模型关节数 {model.njnt}")
        self.jnt_ids = list(range(j0, j0 + self.nq))
        self.q_off = int(model.jnt_qposadr[j0])
        self.d_off = int(model.jnt_dofadr[j0])
        self.q_sl = slice(self.q_off, self.q_off + self.nq)
        self.d_sl = slice(self.d_off, self.d_off + self.nq)
        # 夹爪关节区（qpos 上紧接臂关节之后）
        self.q_grip = self.q_off + self.nq
        self.q_end = self.q_grip + int(spec.n_gripper_qpos)
        if self.q_end > model.nq:
            raise ValueError(f"臂 '{spec.name}': qpos 区间 [{self.q_off},{self.q_end}) "
                             f"超出模型 nq={model.nq}")

        # 执行器区
        a0 = int(spec.actuator_offset)
        if a0 < 0 or a0 + self.nu > model.nu:
            raise ValueError(f"臂 '{spec.name}': actuator_offset={a0} + n_arm_actuators="
                             f"{self.nu} 超出模型 nu={model.nu}")
        self.a_off = a0
        self.a_sl = slice(a0, a0 + self.nu)
        # n_gripper_qpos = 0 表示这条臂没有夹爪（纯定位任务），夹爪相关全部跳过
        self.has_gripper = int(spec.n_gripper_qpos) > 0
        self.idx_grip = (a0 + int(spec.gripper.ctrl_index)) if self.has_gripper else None
        self.idx_boost = (None if (not self.has_gripper
                                   or spec.gripper.boost_ctrl_index is None)
                          else a0 + int(spec.gripper.boost_ctrl_index))
        for label, idx in (("夹爪", self.idx_grip), ("增压", self.idx_boost)):
            if idx is not None and not (0 <= idx < model.nu):
                raise ValueError(f"臂 '{spec.name}': {label}执行器下标 {idx} "
                                 f"超出模型 nu={model.nu}")

        # 末端
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, spec.ee_site)
        if self.site_id < 0:
            raise ValueError(f"臂 '{spec.name}': 模型里没有 site '{spec.ee_site}'")
        self.grip_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                           spec.gripper_body)
        self.tcp_off = float(spec.tcp_offset)

        home = np.asarray(spec.home_qpos, float) if len(spec.home_qpos) else np.zeros(self.nq)
        if home.shape != (self.nq,):
            raise ValueError(f"臂 '{spec.name}': home_qpos 长度 {home.shape[0]} "
                             f"≠ n_arm_qpos {self.nq}")
        self.home = home

        # ---- 运行时状态 ----
        self.q_ik = None
        self.prev_q_ik = None
        self.grip_cmd = 0.0        # 0..255（历史刻度）
        self.blend = 0.0           # 0=瞬移 1=物理
        self.locked = False
        self.hand = None           # 最近一帧手柄消息
        self.trigger = 0.0
        self.squeeze = 0.0
        self.hand0_pos = None
        self.hand0_quat = None
        self.tcp0 = None
        self.ee0_quat = None
        self.target_tcp = None
        self.target_quat = None
        self.reset_anim = False
        self.reset_start = None
        self.reset_from = None
        self.reset_step0 = 0

    # ---- 状态 ----
    def site_z(self):
        return self.data.site_xmat[self.site_id].reshape(3, 3)[:, 2]

    def tcp(self):
        return self.data.site_xpos[self.site_id] - self.tcp_off * self.site_z()

    def ee_quat(self):
        return mat_to_quat(self.data.site_xmat[self.site_id].reshape(3, 3))

    def apply_home(self, data=None):
        """这条臂回到自己的初始位姿，夹爪张开。"""
        d = self.data if data is None else data
        d.qpos[self.q_sl] = self.home
        d.qvel[self.d_sl] = 0.0
        d.qpos[self.q_grip:self.q_end] = 0.0
        d.qvel[self.q_grip:self.q_end] = 0.0
        d.ctrl[self.a_sl] = self.home
        if self.idx_grip is not None:
            d.ctrl[self.idx_grip] = self.spec.gripper.ctrl_open
        if self.idx_boost is not None:
            d.ctrl[self.idx_boost] = self.spec.gripper.boost_value
        self.q_ik = None
        self.prev_q_ik = None
        self.grip_cmd = 0.0
        self.blend = 0.0
        self.locked = False

    # ---- 驱动 ----
    def drive(self, q_sol, blend=0.0):
        """写 qpos（按 blend 混入）与 ctrl。blend=0 纯瞬移，1 纯伺服。"""
        d = self.data
        if blend <= 0.0:
            d.qpos[self.q_sl] = q_sol
        elif blend < 1.0:
            d.qpos[self.q_sl] = (1.0 - blend) * q_sol + blend * d.qpos[self.q_sl]
        if blend < 1.0:
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)
        d.ctrl[self.a_sl] = q_sol
        return "物理" if blend >= 1.0 else ("过渡" if blend > 0.0 else "瞬移")

    def set_gripper(self, cmd):
        """cmd: 0..255 的历史刻度。没有夹爪的臂是空操作。"""
        if self.idx_grip is None:
            return
        g01 = float(np.clip(cmd, 0.0, 255.0)) / 255.0
        self.data.ctrl[self.idx_grip] = self.spec.gripper.cmd_to_ctrl(g01)
        if self.idx_boost is not None:
            self.data.ctrl[self.idx_boost] = self.spec.gripper.boost_value

    def grip_cmd_of(self, trigger):
        return float(np.clip(trigger, 0.0, 1.0)) * 255.0

    # ---- IK ----
    def ik(self, target_tcp, target_quat):
        self.q_ik = ik_solve(self.model, self.data, self.site_id,
                             target_tcp, target_quat, q_seed=self.q_ik,
                             q_slice=self.q_sl, d_slice=self.d_sl,
                             jnt_ids=self.jnt_ids, tcp_off=self.tcp_off)
        return self.q_ik

    # ---- 手柄 → 目标 ----
    def on_hand(self, msg):
        """更新这条臂的手柄消息并维护锁定状态。返回 (target_tcp, target_quat)。"""
        self.hand = msg
        if msg is None:
            self.trigger = self.squeeze = 0.0
            self.locked = False
            return self.target_tcp, self.target_quat

        self.trigger = float(msg.get("trigger", 0.0) or 0.0)
        self.squeeze = float(msg.get("squeeze", 0.0) or 0.0)
        p = np.asarray(msg.get("pos", (0.0, 0.0, 0.0)), float)
        q = np.asarray(msg.get("quat", (0.0, 0.0, 0.0, 1.0)), float)
        if q.shape != (4,):
            q = np.array([0.0, 0.0, 0.0, 1.0])

        if self.squeeze > GRIP_THRESH:
            if not self.locked:
                # 锁定瞬间: 记下手柄/末端基准位姿 (相对映射的锚点)
                self.locked = True
                self.hand0_pos = p.copy()
                self.hand0_quat = q.copy()
                self.tcp0 = self.tcp().copy()
                self.ee0_quat = self.ee_quat()
                self.q_ik = None
            else:
                self.target_tcp, self.target_quat = map_hand_to_target(
                    p, q, self.hand0_pos, self.hand0_quat, self.tcp0, self.ee0_quat)
        else:
            if self.locked:
                self.locked = False
                self.q_ik = None
        return self.target_tcp, self.target_quat


def build_arms(model, data):
    """按 profile 建出全部臂的运行时。任何索引/名字错误都在**启动时**报出来。"""
    specs = PROFILE.arm_specs
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise SystemExit(f"[双臂] ✗ ArmSpec.name 有重名: {names}\n"
                         "       日志/界面/状态字典都用它做键，必须唯一。")
    return [ArmRuntime(s, model, data) for s in specs]


# ---------------- 装置 (模型 + 索引 + 抓取约束) ----------------
class Rig:
    """把场景加载/索引/初始状态/抓取约束收在一起, 方便离线验证复用。"""

    def __init__(self, model_path=MODEL_PATH, calibrated=True, reset=True):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        if calibrated:
            calibrate(self.model)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, PROFILE.ee_site)
        self.grip_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PROFILE.gripper_body)
        self.obj_names = list(PROFILE.object_bodies)
        self.obj_body = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
                         for n in self.obj_names}
        self.obj_qadr = {n: m.jnt_qposadr[m.body_jntadr[self.obj_body[n]]]
                         for n in self.obj_names}
        # 挂载方案需要: 方块的速度地址 (搬运时清零)、几何范围 (搬运时关碰撞)、
        # 以及碰撞标志的原始值 (恢复时要还原)
        self.obj_dadr = {n: m.jnt_dofadr[m.body_jntadr[self.obj_body[n]]]
                         for n in self.obj_names}
        self.obj_geom = {n: (m.body_geomadr[self.obj_body[n]],
                             m.body_geomnum[self.obj_body[n]])
                         for n in self.obj_names}
        self.geom_contype0 = m.geom_contype.copy()
        self.geom_conaffinity0 = m.geom_conaffinity.copy()
        mujoco.mj_resetData(m, self.data)
        self.obj_qpos0 = {n: self.data.qpos[self.obj_qadr[n]:self.obj_qadr[n]+7].copy()
                          for n in self.obj_names}
        self.obj_layout = None          # 最近一次随机摆放 [(名,x,y)...] (None=XML 原布局)
        self.obj_layout_relax = None    # 生成时用的 (避让半径, 最小间距), 用于自检
        self.held = None        # 挂载方案: 当前被搬运的方块名
        self.held_rel = None    # (rel_pos, rel_R): 方块在夹爪 base 系下的位姿
        if reset:
            self.reset_scene()
        else:
            mujoco.mj_forward(m, self.data)

    # --- 状态 ---
    def site_z(self):
        return self.data.site_xmat[self.site_id].reshape(3, 3)[:, 2]

    def tcp(self):
        return self.data.site_xpos[self.site_id] - TCP_OFF * self.site_z()

    def reset_scene(self):
        """机械臂回任务初始位, 夹爪张开, 方块归位。"""
        d, m = self.data, self.model
        self.detach()               # 挂载方案: 先解除搬运并恢复方块碰撞
        d.ctrl[:] = 0.0
        d.qvel[:] = 0.0
        d.qpos[:NQ_ARM] = TASK_HOME
        d.qpos[NQ_ARM:NQ_GRIP_END] = 0.0
        for n in self.obj_names:
            a = self.obj_qadr[n]
            d.qpos[a:a+7] = self.obj_qpos0[n]
        d.ctrl[:NQ_ARM] = TASK_HOME
        d.ctrl[IDX_GRIP] = 0.0
        set_boost(d.ctrl)
        mujoco.mj_forward(m, d)

    # --- 物块随机摆放 (用户要求: 在臂可达区内随机, 且留冗余) ---
    def randomize_objects(self, rng=None, avoid_xy=None):
        """把 4 个方块随机摆到可达区内的位置。

        约束 (按重要性):
          1. 落在 OBJ_RANDOM_X × OBJ_RANDOM_Y 矩形内 —— 这个矩形必须整体在臂可达区内;
          2. 方块中心两两间距 >= OBJ_MIN_SEP (给夹爪接近留余地);
          3. 离**当前** TCP 的 (x,y) 距离 >= OBJ_AVOID_TCP —— 回位时机械臂可能正好悬在
             某处, 若方块直接生成在夹爪里, 求解器会把它弹飞。

        约束太紧放不下时**逐步放宽** (先缩避让半径, 再缩间距), 并打印用了哪一档 ——
        宁可稍微挤一点, 也不要生成失败留下旧的固定布局。

        同时把新位置写进 `obj_qpos0` —— 这样 `reset_scene()` / "方块归位"的目标就是
        **本次布局**, 不会把方块弹回 XML 里的老坐标。

        返回 [(名字, x, y), ...]; 失败返回 None (调用方保持原布局)。
        """
        rng = rng or np.random.default_rng()
        n = len(self.obj_names)
        x0, x1 = OBJ_RANDOM_X
        y0, y1 = OBJ_RANDOM_Y
        # (避让半径, 最小间距) —— 从最严到最松
        relax = [(OBJ_AVOID_TCP, OBJ_MIN_SEP),
                 (OBJ_AVOID_TCP * 0.6, OBJ_MIN_SEP),
                 (OBJ_AVOID_TCP * 0.3, OBJ_MIN_SEP * 0.9),
                 (0.0, OBJ_MIN_SEP * 0.8)]
        pts, used = None, None
        for ar, sep in relax:
            cand = []
            for _ in range(6000):
                if len(cand) == n:
                    break
                p = (float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)))
                if any((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < sep ** 2 for q in cand):
                    continue
                if avoid_xy is not None and ar > 0 and \
                        np.hypot(p[0] - avoid_xy[0], p[1] - avoid_xy[1]) < ar:
                    continue
                cand.append(p)
            if len(cand) == n:
                pts, used = cand, (ar, sep)
                break
        if pts is None:
            return None
        d = self.data
        for name, (x, y) in zip(self.obj_names, pts):
            a = self.obj_qadr[name]
            d.qpos[a] = x
            d.qpos[a + 1] = y
            d.qpos[a + 2] = OBJ_Z
            d.qpos[a + 3:a + 7] = (1.0, 0.0, 0.0, 0.0)   # 不随机朝向 (姿态不影响抓取判据)
            self.obj_qpos0[name][:3] = (x, y, OBJ_Z)      # 归位目标 = 本次布局
            self.obj_qpos0[name][3:7] = (1.0, 0.0, 0.0, 0.0)
            v = self.obj_dadr[name]
            d.qvel[v:v + 6] = 0.0
        mujoco.mj_forward(self.model, d)
        self.obj_layout = [(nm, p[0], p[1]) for nm, p in zip(self.obj_names, pts)]
        self.obj_layout_relax = used
        return self.obj_layout

    # --- 物块挂载 (GRASP_MODE="attach") ---
    def set_obj_collide(self, name, on):
        """开关某个方块的碰撞 (搬运期间关掉, 免得和夹爪/桌面打架)。"""
        a, num = self.obj_geom[name]
        if on:
            self.model.geom_contype[a:a+num] = self.geom_contype0[a:a+num]
            self.model.geom_conaffinity[a:a+num] = self.geom_conaffinity0[a:a+num]
        else:
            self.model.geom_contype[a:a+num] = 0
            self.model.geom_conaffinity[a:a+num] = 0

    def detach(self):
        """解除挂载: 恢复方块碰撞, 让它重新受物理支配。"""
        if self.held is not None:
            self.set_obj_collide(self.held, True)
            self.held = None
            self.held_rel = None

    def update_grasp(self, trigger, held):
        """扳机过阈值 + 方块在抓取中心附近 → 挂上 (返回新的 held)。

        为什么用"跟随式挂载"而不是 weld 等式约束: 机械臂是"每帧把 qpos 写到 IK 解"
        这种瞬移式驱动, 再叠一个刚性约束时求解器每帧都要纠正位置突变, 会把方块弹飞
        (实测被甩到几米外)。关掉碰撞的跟随式搬运稳定可控, 松开后恢复物理。
        """
        d = self.data
        if held is not None and trigger < DETACH_TRIG:
            self.detach()
            print(f"[抓取] 松开 {held}")
            return None
        if held is None and trigger > ATTACH_TRIG:
            Rg = d.xmat[self.grip_body].reshape(3, 3)
            pg = d.xpos[self.grip_body]
            best = None
            for n in self.obj_names:
                b = self.obj_body[n]
                rel = Rg.T @ (d.xpos[b] - pg)
                dist = float(np.linalg.norm(rel - GRIP_CENTER_LOCAL))
                if dist < ATTACH_DIST and (best is None or dist < best[1]):
                    best = (n, dist, rel, Rg.T @ d.xmat[b].reshape(3, 3))
            if best is not None:
                n, dist, rel, relR = best
                self.held = n
                self.held_rel = (rel, relR)
                self.set_obj_collide(n, False)
                print(f"[抓取] 挂上 {n} (离抓取中心 {dist*1000:.1f}mm)")
                return n
        return held

    def carry(self):
        """把被挂载方块刚性跟随到夹爪上 (搬运期间每帧调用, 在 mj_step 之前)。"""
        if self.held is None:
            return
        d = self.data
        n = self.held
        rel, relR = self.held_rel
        Rg = d.xmat[self.grip_body].reshape(3, 3)
        pg = d.xpos[self.grip_body]
        a = self.obj_qadr[n]
        d.qpos[a:a+3] = pg + Rg @ rel
        d.qpos[a+3:a+7] = S_quat_wxyz(mat_to_quat(Rg @ relR))
        v = self.obj_dadr[n]
        d.qvel[v:v+6] = 0.0

    def settle(self, steps=1500):
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

    def pad_geoms(self):
        """左右两侧指垫的 geom id 集合。"""
        out = {"r": set(), "l": set()}
        for n in ("right_pad1", "right_pad2", "left_pad1", "left_pad2"):
            g = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
            if g >= 0:
                out["r" if n.startswith("right") else "l"].add(g)
        return out

    # --- 驱动 ---
    def touching_object(self):
        """机械臂/夹爪是否正碰到某个可抓物体 (与桌子的接触不算)。"""
        m, d = self.model, self.data
        objs = set(self.obj_body.values())
        for c in range(d.ncon):
            con = d.contact[c]
            b1 = m.geom_bodyid[con.geom1]
            b2 = m.geom_bodyid[con.geom2]
            if (b1 in objs) ^ (b2 in objs):
                other = b2 if b1 in objs else b1
                if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, other) != "table":
                    return True
        return False

    def drive(self, q_sol, mode, contact=False, blend=0.0):
        """按模式下发 IK 解。

        teleport: 每帧把 qpos 设成 IK 解 (原版做法, 跟踪 1.4mm, 但接触非物理)
        servo   : 只写 ctrl, 由位置伺服过去 (接触物理真实, 但快速运动有速度跟随误差)
        auto    : 按 contact 在两者间切 (已弃用: 硬切换会在接触瞬间造成"卡住再瞬移")

        blend ∈ [0,1] —— **连续过渡**, 取代硬切换:
            blend=0 → qpos 完全设成 IK 解 (等价 teleport)
            blend=1 → 完全不碰 qpos      (等价 servo)
            0<blend<1 → 只把 qpos 朝解挪 (1-blend) 的比例 → 平滑交接给物理
        """
        d = self.data
        if mode == "auto":
            mode = "servo" if contact else "teleport"
        if mode == "teleport":
            if blend <= 0.0:
                d.qpos[:NQ_ARM] = q_sol
            elif blend < 1.0:
                d.qpos[:NQ_ARM] = (1.0 - blend) * q_sol + blend * d.qpos[:NQ_ARM]
            if blend < 1.0:
                mujoco.mj_kinematics(self.model, d)
                mujoco.mj_comPos(self.model, d)
        d.ctrl[:NQ_ARM] = q_sol
        return mode

    # --- 夹爪 ---
    def set_gripper(self, cmd):
        """cmd: 0..255 的归一化行程（历史接口）。按 profile 映射成真实 ctrl。"""
        g01 = float(np.clip(cmd, 0.0, 255.0)) / 255.0
        self.data.ctrl[IDX_GRIP] = PROFILE.gripper.cmd_to_ctrl(g01)
        set_boost(self.data.ctrl)

    def grip_cmd_of(self, trigger):
        return float(np.clip(trigger, 0.0, 1.0)) * 255.0

    # --- 抓取 (纯物理) ---
    def detect_grasp(self):
        """物理判定"是否夹住": 某个方块同时与左右两指的指垫接触。

        对应真机 2F-85 自带的物体检测 (电流超限 → 手指停住并上报),
        仿真里最接近的等价量就是"两指都摸到同一个物体"。
        注意: 这里只做**判定**, 不改变物理 —— 抓持完全靠接触摩擦。
        """
        d, m = self.data, self.model
        pads = self.pad_geoms()
        touched = {"r": set(), "l": set()}
        for c in range(d.ncon):
            c1, c2 = d.contact[c].geom1, d.contact[c].geom2
            for a, b in ((c1, c2), (c2, c1)):
                side = "r" if a in pads["r"] else ("l" if a in pads["l"] else None)
                if side is None:
                    continue
                bb = m.geom_bodyid[b]
                for n in self.obj_names:
                    if bb == self.obj_body[n]:
                        touched[side].add(n)
        both = touched["r"] & touched["l"]
        return sorted(both)[0] if both else None

    def grasp_report(self):
        """诊断用: 当前与指垫接触的物体 + 两指间距 + driver 角。"""
        d, m = self.data, self.model
        pads = self.pad_geoms()
        names = set()
        for c in range(d.ncon):
            for a, b in ((d.contact[c].geom1, d.contact[c].geom2),
                         (d.contact[c].geom2, d.contact[c].geom1)):
                if a in pads["r"] or a in pads["l"]:
                    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[b])
                    names.add(nm)
        return names, float(d.qpos[NQ_ARM]), self.detect_grasp()


# ---------------- MuJoCo 主循环 ----------------
# 停止信号: Ctrl-C / kill / 关 viewer 都汇到这一条**干净收尾**路径。
#
# ★ 为什么必须有 (实测): 不装信号处理器时, 主线程在 asyncio.run 里吃到
#   KeyboardInterrupt → 解释器立刻开始拆解, 而仿真线程可能正停在 MuJoCo 的 **C 代码**里
#   → 段错误。实测 Ctrl-C 的退出码是 **-11 (SIGSEGV)**, 而且
#   数采/相机窗口/日志摘要**全都没收尾** (录制中还会留下半截 mp4)。
#   现在的做法: 信号 → 置 STOP → 主循环跳出 → 关 viewer → 停数采(会自动丢弃
#   未完成的那一集并删掉半截 mp4) → 停相机窗口 → 写日志摘要 → 主线程 join 后再退出。
STOP = threading.Event()
_stop_why = {"why": ""}


def request_stop(why):
    """请求干净退出 (幂等)。"""
    if not STOP.is_set():
        _stop_why["why"] = why
        print(f"\n[退出] {why} → 开始干净收尾 "
              f"(关 viewer / 停数采 / 关相机窗口 / 写日志摘要)...", flush=True)
    STOP.set()


def _ask_yes(prompt):
    """问一个是/否。认 y/yes/是/要/1 为"是"; 空串或其它都是"否"。
    EOF/中断(非交互环境)也算"否" —— 绝不因为读不到输入就默默开始采集。"""
    try:
        a = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt, OSError):
        print()
        return False
    return a in ("y", "yes", "是", "要", "1", "true", "对", "好")


def ask_startup_task():
    """启动时问「要不要数采」+「任务描述」, 返回任务串; 不采集则返回 None。

    ★★★ 必须在**主线程**、且在任何线程/GL/tkinter 窗口/事件循环**之前**调用。

    踩过的坑 (2026-10-10, 用户报「epoll_ctl: Invalid argument / 非法指令(核心已转储)」):
    我一开始把这段提问放在 `run_mujoco` 里 —— 那是个**子线程**, 而 camview 的
    tkinter 窗口同时已经起来了。结果一回答就 SIGILL 崩溃。gdb 抓到的原生堆栈是:

        #0  Tcl_Panic ()                      libtcl9.0.so
        #2  EventHook ()                      _tkinter.cpython-311-*.so
        #3  my_fgets
        #4  PyOS_StdioReadline
        #5  PyOS_Readline
        #6  builtin_input_impl ()              ← input()

    机制: tkinter 初始化时会把 CPython 的 `PyOS_ReadlineFunctionPointer` 换成它自己的
    `EventHook` (目的是「等用户输入时还能继续处理 Tk 事件」)。于是 `input()` 不是走
    普通的 stdio 读, 而是进了 **Tcl 的事件循环**; 在非 Tk 主线程上调用 → `Tcl_Panic`
    → 库自己 `abort`/`ud2` → **SIGILL (core dumped)**。

    为什么之前测试没发现: 那个钩子**只在 stdin 是终端(tty)时才装** —— 我一直用
    `< 文件` 管道喂 stdin, 走的是另一条路径, 全绿; 用户一在真终端里跑就崩。
    → **凡是交互式提问, 要么在主线程问, 要么别用 `input()`。**
      (复现/回归工具: `tools/repro_tty_input.py` 用 pty 模拟真终端。)
    """
    if not DATASET_ENABLE:
        return None
    if DUAL_ARM:
        print("[数采] 双臂模式：本版**不做数据采集**（只遥操作）。"
              "见 docs/08_DUAL_ARM.md 的'为什么双臂不采数'。")
        return None
    env_task = (os.environ.get("VR_TELEOP_DS_TASK") or "").strip()
    if env_task:
        print(f"[数采] 环境变量指定任务『{env_task}』, 跳过提问, 直接开始采集")
        return env_task
    if threading.current_thread() is not threading.main_thread():
        # 保险丝: 万一以后又有人从子线程调它, 别让用户吃 SIGILL
        print("[数采] ⚠ 不是主线程, 跳过交互提问 (避免 tkinter 的 readline 钩子崩溃); "
              "本次不采集。要采集请用环境变量 VR_TELEOP_DS_TASK")
        return None

    print("\n" + "=" * 70)
    print("[数采] 本次运行**要做数据采集**吗?")
    print("       要   → 输入 y 回车, 然后填写本次的任务描述")
    print("       不要 → 直接回车 (只做遥操作练习, 不写数据集)")
    print("=" * 70, flush=True)
    if not _ask_yes("要做数据采集吗? [y/N] > "):
        print("[数采] 好, 本次**不采集数据**, 只做遥操作。"
              "(想采集就重启并回答 y)")
        return None
    print("[数采] 请输入**本次的任务描述** —— 它会作为训练用的语言指令写进数据集。")
    print("       例: 把红色方块放到黄色方块上")
    try:
        task = input("任务> ").strip()
    except (EOFError, KeyboardInterrupt, OSError):
        task = ""
    if not task:
        task = "pick up the block and place it on the table"
        print(f"[数采] 没有输入任务, 使用默认『{task}』")
    return task


# ---------------- 双臂主循环 ----------------
def step_arms(arms, model, data, dt):
    """推进一帧：每条臂各自 IK + 下发 + 夹爪。返回 {臂名: 驱动模式}。

    单臂/双臂共用同一套逻辑（`run_mujoco_dual` 与 tools/validate_dual.py 都调它），
    保证"验证过的行为"和"跑起来的行为"是同一份代码。
    """
    modes = {}
    for a in arms:
        if a.locked and a.target_tcp is not None and a.target_quat is not None:
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            q = a.ik(a.target_tcp, a.target_quat)
            # 判据 = 抓取意图（扣扳机），与单臂 physics 模式一致
            want_physics = a.trigger > PHYS_TRIGGER_MIN
            _sw = dt / TELEPORT_RAMP if TELEPORT_RAMP > 1e-6 else 1.0
            a.blend = (min(1.0, a.blend + _sw) if want_physics
                       else max(0.0, a.blend - _sw))
            modes[a.name] = a.drive(q, a.blend)
        else:
            modes[a.name] = "求解中" if a.locked else "冻结"

        gt = a.grip_cmd_of(a.trigger)
        a.grip_cmd = (min(gt, a.grip_cmd + GRIP_RATE) if gt > a.grip_cmd
                      else max(gt, a.grip_cmd - GRIP_RATE))
        a.set_gripper(a.grip_cmd)
    return modes


def run_mujoco_dual(state):
    """双臂遥操作：左手柄驱动一条臂、右手柄驱动另一条，互不干扰。

    与单臂路径的**有意差异**（都是"这一版没做"，不是遗漏）：
      * 不做数据采集 —— LeRobot 特征是按"单臂关节空间 + 一个夹爪"设计的，
        双臂要么拼成 2N+2 维、要么拆成两组，需要重新设计；见 docs/08。
      * 不做物块挂载（attach 方案）—— 挂载状态是单臂实现，双臂只用物理接触。
      * 不做运行日志 —— LOG_COLUMNS 是按单臂 7 关节写死的。
      * 不做方块随机摆放 —— 纯遥操作练习，不涉及任务。

    两条臂共用同一套映射参数（R_MAP / 符号 / scale / 是否跟姿态）：
    同一个人的两只手，坐标系约定本来就一致。
    """
    rig = Rig(reset=False)
    model, data = rig.model, rig.data
    arms = build_arms(model, data)

    hands = [a.hand_name for a in arms]
    if len(set(hands)) != len(hands):
        raise SystemExit(
            "[双臂] ✗ 两条臂绑定了同一个手柄，无法各自独立控制。\n"
            f"       当前: {[(a.name, a.hand_name) for a in arms]}\n"
            "       改 robot_profile.py 里每个 ArmSpec 的 drive_hand（一左一右）。")

    print(f"[双臂] 模型 {MODEL_PATH}")
    print(f"[双臂] nq={model.nq} nv={model.nv} nu={model.nu} njnt={model.njnt}")
    for a in arms:
        print(f"[双臂]   {a.name:<8} 手={a.hand_name:<5} "
              f"关节[{a.q_off}:{a.q_off + a.nq}] ctrl[{a.a_off}:{a.a_off + a.nu}] "
              f"site={a.spec.ee_site}(id={a.site_id}) 夹爪ctrl={a.idx_grip}")
    print(f"[双臂] 驱动: teleport↔物理 渐变 (TELEPORT_RAMP={TELEPORT_RAMP}s)｜"
          f"跟姿态={'是' if FOLLOW_ORI else '否'} scale={SCALE}")
    if DATASET_ENABLE:
        print("[双臂] ⚠ 数据采集已自动关闭（双臂采数尚未实现，见 docs/08）")
    if GRASP_MODE == "attach":
        print("[双臂] ⚠ GRASP_MODE=attach 在双臂下不支持，按 physics 运行")

    # ---- 相机视图（与单臂共用同一套配置）----
    camview = CameraViewer(MODEL_PATH, CAMVIEW_CAMS, size=CAMVIEW_SIZE,
                           fps=CAMVIEW_FPS, enabled=CAMVIEW_SHOW,
                           dump_dir=os.environ.get("VR_TELEOP_CAMDUMP") or None,
                           dump_every=int(os.environ.get("VR_TELEOP_CAMDUMP_EVERY", "0")),
                           positions=CAMVIEW_POS,
                           tilt_deg=CAMVIEW_TILT,
                           offset=CAMVIEW_OFFSET,
                           abs_cams=CAMVIEW_ABS,
                           extra_cams=CAMVIEW_NEW,
                           render_off=RENDER_OFF).start()

    for a in arms:
        a.apply_home()
    mujoco.mj_forward(model, data)

    from glutil import gl_guard
    with gl_guard():
        viewer_cm = mujoco.viewer.launch_passive(model, data,
                                                 show_left_ui=False,
                                                 show_right_ui=False)
    with viewer_cm as viewer:
        viewer.cam.distance = VIEWER_CAM["distance"]
        viewer.cam.elevation = VIEWER_CAM["elevation"]
        viewer.cam.azimuth = VIEWER_CAM["azimuth"]
        viewer.cam.lookat[:] = VIEWER_CAM["lookat"]

        step = 0
        dt = model.opt.timestep
        wall_start = time.time()
        sync_cost = 0.0
        modes = {a.name: "冻结" for a in arms}

        while viewer.is_running() and not STOP.is_set():
            sim_time = step * dt
            wall_elapsed = time.time() - wall_start
            sleep_left = sim_time - wall_elapsed
            if sleep_left > 0:
                time.sleep(sleep_left)

            # ---- 回位 ----
            with state.lock:
                need_reset = state.reset_requested
                state.reset_requested = False
            if need_reset:
                for a in arms:
                    a.apply_home()
                for n in rig.obj_names:
                    adr = rig.obj_qadr[n]
                    data.qpos[adr:adr + 7] = rig.obj_qpos0[n]
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                print(f"[双臂] 回位: {len(arms)} 条臂 → 各自 home, 夹爪张开, 方块归位")

            # ---- 读手柄 → 每条臂各自维护锁定/目标 ----
            for a in arms:
                msg = state.right if a.hand_name == "right" else state.left
                a.on_hand(msg)

            # ---- 每条臂各自 IK + 下发（顺序无关: IK 内部会保存/还原整份 qpos）----
            modes = step_arms(arms, model, data, dt)

            mujoco.mj_step(model, data)
            camview.publish(data.qpos)

            if step % VIEWER_SYNC_EVERY == 0:
                _t0 = time.perf_counter()
                viewer.sync(state_only=VIEWER_STATE_ONLY)
                sync_cost = time.perf_counter() - _t0
                lines = []
                for a in arms:
                    t = a.tcp()
                    lines.append(
                        f"{a.name}({a.hand_name[0].upper()}手) {modes[a.name]:<3} "
                        f"x={t[0]:+.3f} y={t[1]:+.3f} z={t[2]:+.3f} "
                        f"夹爪{a.grip_cmd / 255 * 100:3.0f}%")
                viewer.set_texts(
                    (mujoco.mjtFontScale.mjFONTSCALE_150,
                     mujoco.mjtGridPos.mjGRID_TOPLEFT,
                     f"VR Teleop - 双臂 {len(arms)} 条臂",
                     "\n".join(lines)
                     + "\n侧键=各自跟随(A/B=回位)  扳机=各自夹爪"
                     + f"  渲染 {sync_cost * 1000:.1f}ms"))
            step += 1

            if step % 500 == 0:
                s = "  ".join(
                    f"{a.name}:{'锁' if a.locked else '松'} "
                    f"tcp=({a.tcp()[0]:+.2f},{a.tcp()[1]:+.2f},{a.tcp()[2]:+.2f})"
                    for a in arms)
                print(f"[双臂] t={sim_time:6.2f}s {s}", flush=True)

    camview.stop()
    print("[双臂] 收尾完成 (相机窗口已关)")


def run_mujoco(state):
    if DUAL_ARM:
        return run_mujoco_dual(state)
    rig = Rig()
    model, data = rig.model, rig.data
    site_id = rig.site_id

    print(f"[MuJoCo] 模型 {MODEL_PATH}")
    print(f"[MuJoCo] nq={model.nq} nv={model.nv} nu={model.nu}, 方块: {rig.obj_names}")
    print(f"[MuJoCo] 夹爪 actuator idx={IDX_GRIP}, "
          f"boost idx={IDX_BOOST if IDX_BOOST is not None else '无'}")
    print(f"[MuJoCo] 抓取方案: {GRASP_MODE}"
          + (" (物块挂载: 旁路接触物理, 全程瞬移 → 无伺服滞后)"
             if GRASP_MODE == "attach" else " (纯物理摩擦: 接触切伺服)")
          + f" | 臂驱动: {'全程 teleport' if GRASP_MODE == 'attach' else DRIVE_MODE}")

    # ---- 物块随机摆放 (启动一次; 之后每次 A/B 回位再重新随机) ----
    def obj_layout_line(tag):
        if rig.obj_layout is None:
            return f"[布局] {tag}: 用 XML 原布局 (随机摆放已关)"
        ar, sep = rig.obj_layout_relax
        s = "  ".join(f"{n}=({x:.3f},{y:.3f})" for n, x, y in rig.obj_layout)
        extra = "" if ar >= OBJ_AVOID_TCP else f" [约束放宽: 避让{ar:.2f}m/间距{sep:.2f}m]"
        return f"[布局] {tag}: {s}{extra}"

    if OBJ_RANDOM:
        lay = rig.randomize_objects()
        if lay is None:
            print("[布局] ⚠ 随机摆放失败 (约束太紧), 保持 XML 原布局")
        else:
            print(obj_layout_line("启动随机摆放"))
            print(f"[布局]   区域 x∈{OBJ_RANDOM_X} y∈{OBJ_RANDOM_Y} "
                  f"最小间距 {OBJ_MIN_SEP}m (全部在臂可达区内, 已用真实 e2e 验证)")
    else:
        print("[布局] 随机摆放已关闭 (VR_TELEOP_OBJ_RANDOM=0), 用 XML 原布局")

    # ---- 相机视图 (VLA 数据采集): 三个独立窗口 ----
    camview = CameraViewer(MODEL_PATH, CAMVIEW_CAMS, size=CAMVIEW_SIZE,
                           fps=CAMVIEW_FPS, enabled=CAMVIEW_SHOW,
                           dump_dir=os.environ.get("VR_TELEOP_CAMDUMP") or None,
                           dump_every=int(os.environ.get("VR_TELEOP_CAMDUMP_EVERY", "0")),
                           positions=CAMVIEW_POS,
                           tilt_deg=CAMVIEW_TILT,
                           offset=CAMVIEW_OFFSET,
                           abs_cams=CAMVIEW_ABS,
                           extra_cams=CAMVIEW_NEW,
                           render_off=RENDER_OFF).start()

    # ---- 数据集采集 (LeRobot v2.1, π0 关节空间, 50Hz) ----
    # 提问已在 **主线程** 做完 (见 ask_startup_task / __main__), 这里只消费答案。
    # 千万不要在这里 input() —— 这个函数跑在子线程里, 而 camview 的 tkinter
    # 已经把 readline 钩子换成了 Tcl 的 EventHook → 在子线程调用会 SIGILL。
    rec = None
    task = getattr(state, "ds_task", None)
    if DATASET_ENABLE and task is None:
        # 兜底: 有脚本直接调 run_mujoco (如 tools/smoke_live.py), 没经过主线程提问
        task = (os.environ.get("VR_TELEOP_DS_TASK") or "").strip() or None
    if task:
        try:
            rec = DatasetRecorder(
                DATASET_ROOT, task, MODEL_PATH, DATASET_CAMS,
                image_size=DATASET_IMAGE, fps=DATASET_FPS,
                cam_tilt=CAMVIEW_TILT, cam_offset=CAMVIEW_OFFSET,
                cam_abs=CAMVIEW_ABS,
                cam_extra=CAMVIEW_NEW,
                render_off=RENDER_OFF,
                max_episode_s=DATASET_MAX_EP_S,
                min_episode_s=DATASET_MIN_EP_S,
                session_id=time.strftime("%Y%m%d_%H%M%S"))
            rec.start()
        except Exception as e:                                   # noqa: BLE001
            print(f"[数采] 启动失败({e}); **遥操作不受影响, 但不采集数据**")
            traceback.print_exc()
            rec = None
    elif DATASET_ENABLE:
        print("[数采] 本次不采集数据 (只做遥操作)")

    # ---- 运行日志 (每次运行一个文件, 事后发文件即可定位问题) ----
    # 给日志摘要用的极值 (放 dict 里, 好让 atexit 的兜底回调也能读到)
    lgstat = dict(max_qd=0.0, max_ik_step=0.0, max_loop=0.0, max_sync=0.0,
                  n_lines=0, prev_qik=None)
    lg = RunLogger(LOG_DIR, columns=LOG_COLUMNS, every=LOG_EVERY_STEPS,
                   enabled=LOG_ENABLE, gzip_on_close=LOG_GZIP,
                   meta=log_meta())
    if lg.enabled:
        _mb_min = lg.ncol * 4 * (500.0 / LOG_EVERY_STEPS) * 60 / 1e6
        print(f"[日志] 本次运行记录到 {lg.path_bin}")
        print(f"[日志]   列={lg.ncol} 采样={500.0/LOG_EVERY_STEPS:.0f}Hz "
              f"约 {_mb_min:.1f} MB/分钟 (10分钟约 {_mb_min*10:.0f} MB)"
              f" | 配套说明 {os.path.basename(lg.path_json)}")

    def _close_log(why="正常退出"):
        """正常退出/异常退出都走这里。close() 幂等, 重复调用无害。"""
        try:
            # ★ 网络质量也落盘（原来那行每秒诊断只打终端，事后无法横向比：
            #   "峰间隔上百 ms = 丢包/省电休眠"，见 maybe_report 的注释）
            _net = {}
            try:
                if diag.net_n:
                    _net = {
                        "net_msgs": diag.net_n,
                        "net_gap_mean_ms": round(diag.net_sum / diag.net_n * 1000.0, 2),
                        "net_gap_max_ms": round(diag.net_max * 1000.0, 2),
                        "net_gaps_gt50ms": diag.net_late,
                    }
            except NameError:                       # diag 还没建（atexit 兜底路径）
                _net = {}
            lg.close(summary={
                "exit": why,
                "samples": lgstat["n_lines"],
                "max_joint_speed_rad_s": round(lgstat["max_qd"], 4),
                "max_ik_solution_step_rad": round(lgstat["max_ik_step"], 5),
                "max_loop_ms": round(lgstat["max_loop"], 2),
                "max_viewer_sync_ms": round(lgstat["max_sync"], 3),
                "late_frames": diag.late,
                **_net,
                "note": ("max_joint_speed 看抽风; max_ik_solution_step 看解跳变(不带动"
                         "控制器时纯 IK 的连续性); max_loop_ms 看卡顿; "
                         "net_gap_max_ms 看网络(上百 = 丢包/省电休眠)"),
            })
            print(f"[日志] 已写出: {lg.path_bin} / {lg.path_json} / {lg.path_log}")
        except Exception as e:                      # noqa: BLE001 日志绝不能拖垮主流程
            print(f"[日志] 收尾出错 (忽略): {e}")

    atexit.register(_close_log, "atexit(可能是崩溃/强退)")

    # viewer 自己会起线程建 GL 窗口 —— 用同一把锁串起来, 避免和 camview/数采
    # 的 GL 上下文创建撞车 (见 glutil.py)
    from glutil import gl_guard
    with gl_guard():
        viewer_cm = mujoco.viewer.launch_passive(model, data,
                                                 show_left_ui=False,
                                                 show_right_ui=False)
    with viewer_cm as viewer:
        # 默认视角 = 机械臂和桌子的**左上后方** (数值由 tools/design_viewer_cam.py
        # 算出来并出图确认 —— 见 VIEWER_CAM 的注释)。免得每次启动都要用鼠标调。
        viewer.cam.distance = VIEWER_CAM["distance"]
        viewer.cam.elevation = VIEWER_CAM["elevation"]
        viewer.cam.azimuth = VIEWER_CAM["azimuth"]
        viewer.cam.lookat[:] = VIEWER_CAM["lookat"]

        step = 0
        dt = model.opt.timestep
        wall_start = time.time()

        # 手抖滤波器 (状态跨锁定/解锁保留, 松手再按无跳变)
        hand_pos_f = OneEuroFilter(FILTER_MIN_CUTOFF, FILTER_BETA, FILTER_D_CUTOFF)
        hand_quat_f = OneEuroQuat(FILTER_ROT_MIN_CUTOFF, FILTER_ROT_BETA, FILTER_D_CUTOFF)
        last_hand_seq = -1      # 上次统计/记录过的手柄样本序号
        hand_log_f = None
        if HAND_LOG:
            hand_log_f = open(HAND_LOG, "w", buffering=1)
            hand_log_f.write("t,px,py,pz,fx,fy,fz,qx,qy,qz,qw\n")
            print(f"[MuJoCo] 手柄位姿记录到 {HAND_LOG} (原始 px..pz / 滤波后 fx..fz / 滤波后四元数)")
        if FILTER_ENABLE:
            print(f"[MuJoCo] 手抖滤波: 开 min_cutoff={FILTER_MIN_CUTOFF}Hz beta={FILTER_BETA} "
                  f"旋转 min_cutoff={FILTER_ROT_MIN_CUTOFF}Hz beta={FILTER_ROT_BETA}")
        else:
            print("[MuJoCo] 手抖滤波: 关 (手柄位姿原样透传)")

        q_ik = None
        grip_cmd = 0.0
        held = None
        held_prev = None        # 供日志记"夹住/松开"事件
        contact_frames = 0      # >0 表示当前按"接触中"处理 (已弃用, 保留兼容)
        blend = 0.0             # 瞬移↔物理 的连续交接权重: 0=纯瞬移, 1=纯物理
        drive_mode = DRIVE_MODE
        sync_cost = 0.0         # 上次 viewer.sync 的耗时, **单位秒** (显示用)
        _t_prev = 0.0           # 上一帧起点 (算帧周期)
        reset_animating = False
        tp = None               # 指令目标 (没锁定时为 None); 日志要读它
        tq = None
        hand_pos = np.zeros(3)  # 手柄原始位置; 日志要读它
        hand_quat = np.array([0.0, 0.0, 0.0, 1.0])   # 手柄原始四元数
        hand_age_ms = 0.0       # 距最近一次收到新手柄消息的毫秒数 (输入延迟)
        locked = False          # 跟随中? (显式初始化: 回位分支里不赋值)
        _prev_ht = 0.0          # 上一次收到手柄消息的时刻 (算到达间隔)
        loop_ms = 0.0
        reset_q_start = None
        reset_q_target = None
        reset_start_step = 0
        RESET_DURATION = 0.7
        diag = Diag()
        prev_q_ik = None
        lim_js = [j for j in range(NQ_ARM) if model.jnt_limited[j]]

        while viewer.is_running() and not STOP.is_set():
            # 帧周期 (含等待): 正常 ~2.0ms; 卡顿时会跳大 —— 这是判断"卡没卡"的硬指标
            _t_now = time.perf_counter()
            loop_ms = (_t_now - _t_prev) * 1000.0 if _t_prev else 0.0
            _t_prev = _t_now
            sim_time = step * dt

            # ---- 网络质量: 检测头显新消息的到达, 记录到达间隔 ----
            # (放在循环顶部, 与日志开关无关 —— 终端要一直能看到)
            _ht = state.hand_t
            if _ht and _ht != _prev_ht:
                if _prev_ht:
                    diag.net_seen(_ht - _prev_ht)
                _prev_ht = _ht
            wall_elapsed = time.time() - wall_start
            sleep_left = sim_time - wall_elapsed
            if sleep_left > 0:
                time.sleep(sleep_left)
            else:
                diag.late += 1        # 没睡够 = 这一帧在追赶, 直接对应"顿"

            # ---- 回位请求 ----
            with state.lock:
                need_reset = state.reset_requested
                state.reset_requested = False
            if need_reset:
                held = None
                rig.detach()          # 挂载方案: 必须先解除挂载并恢复方块碰撞
                # 物块重新随机摆放 (用户要求: 按 A/B 回位时换一批新位置)。
                # 要避开**当前** TCP —— 此刻机械臂可能正悬在方块区上方, 方块若生成在
                # 夹爪里会被求解器弹飞。放过一次就给个干净的布局。
                if OBJ_RANDOM:
                    tcp_now = rig.tcp()
                    lay_new = rig.randomize_objects(avoid_xy=(tcp_now[0], tcp_now[1]))
                    if lay_new is not None:
                        print(obj_layout_line("回位 → 重新随机摆放"))
                        lg.event("布局", " ".join(
                            f"{n}=({x:.3f},{y:.3f})" for n, x, y in lay_new))
                else:
                    for n in rig.obj_names:
                        a = rig.obj_qadr[n]
                        data.qpos[a:a+7] = rig.obj_qpos0[n]
                data.qvel[:] = 0.0
                reset_animating = True
                reset_q_start = data.qpos[:NQ_GRIP_END].copy()
                tgt = np.concatenate([TASK_HOME, np.zeros(NQ_GRIP_END - NQ_ARM)])
                unlimited = ~model.jnt_limited[:NQ_GRIP_END].astype(bool)
                reset_q_target = np.where(unlimited,
                                          reset_q_start + angle_wrap(tgt - reset_q_start), tgt)
                reset_start_step = step
                q_ik = None
                grip_cmd = 0.0
                with state.lock:
                    state.locked = False
                    state.target_tcp = None
                    state.target_quat = None
                    state.grip_held = None
                print("[MuJoCo] 回位: 机械臂→任务初始位, 夹爪张开, "
                      + ("方块**重新随机摆放**" if OBJ_RANDOM else "方块归位"))
                lg.event("回位", f"锁定={state.locked}")

            # ---- 读手柄（DRIVE_HAND 决定哪只手驱动，另一只手负责采集按键）----
            with state.lock:
                right = state.right
                left = state.left
                was_locked = state.locked
                hand_seq = state.hand_seq
            drive = right if DRIVE_HAND == "right" else left
            trigger = squeeze = 0.0
            if drive is not None:
                trigger = float(drive.get('trigger', 0.0) or 0.0)
                squeeze = float(drive.get('squeeze', 0.0) or 0.0)

            # ---- 数据集采集: 手柄事件边沿检测 (在 500Hz 循环里做, 不会漏按键) ----
            # handle_hands(标签手, 驱动手)：驱动手换边时两者一起换
            if rec is not None:
                if DRIVE_HAND == "right":
                    rec.handle_hands(left, right)
                else:
                    rec.handle_hands(right, left)

            # ---- 夹爪命令跟随扳机 (带速率限制) ----
            grip_target = rig.grip_cmd_of(trigger)
            grip_cmd = (min(grip_target, grip_cmd + GRIP_RATE) if grip_target > grip_cmd
                        else max(grip_target, grip_cmd - GRIP_RATE))

            if reset_animating:
                progress = min((step - reset_start_step) * dt / RESET_DURATION, 1.0)
                s = progress * progress * (3.0 - 2.0 * progress)
                q_cur = reset_q_start + (reset_q_target - reset_q_start) * s
                data.qpos[:NQ_GRIP_END] = q_cur
                # ★ 回位是"每帧瞬移 qpos", 但若把上一帧的 qvel 留着, 求解器就会
                #   拿"旧速度 + 新位置"去解约束 → 速度自激振荡。
                #   实测(t=178.13 那次): |qvel| 冲到 2471.5°/s (joint_5);
                #   每帧清零后降到 176.6°/s (降 14 倍), 且回位路径与预期逐帧一致。
                #   只清到 NQ_GRIP_END(臂+夹爪, 都是被瞬移的); **绝不能清全部 qvel**
                #   —— 那会把方块/物体的自由落体也冻住(实测方块反而"飘"上去)。
                data.qvel[:NQ_GRIP_END] = 0.0
                data.ctrl[:NQ_ARM] = q_cur[:NQ_ARM]
                data.ctrl[IDX_GRIP] = 0.0
                set_boost(data.ctrl)
                if progress >= 1.0:
                    reset_animating = False
                    data.qpos[:NQ_GRIP_END] = reset_q_target
                    data.qvel[:NQ_GRIP_END] = 0.0
                    data.ctrl[:NQ_ARM] = reset_q_target[:NQ_ARM]
                    data.ctrl[IDX_GRIP] = 0.0
                    grip_cmd = 0.0
                mujoco.mj_forward(model, data)

            elif right is not None:
                p = np.array(right['pos'], float)
                q = np.array(right['quat'], float)
                hand_pos = p          # 日志记录手柄原始位姿
                hand_quat = q         # (只记位置没法回放, 必须连四元数一起)
                if len(q) != 4:
                    q = np.array([0, 0, 0, 1.0])

                # ---- 手抖滤波 (滤操作者输入, 不是 IK 结果) ----
                # 语义: 对应真机系统对操作者输入的平滑。只削高频抖动, 不改变可达范围。
                # 滤波器状态跨"锁定/解锁"保留, 所以松手再按不会有跳变。
                p_raw = p
                if FILTER_ENABLE:
                    p = hand_pos_f(p, dt)
                    q = hand_quat_f(q, dt)
                # 只在**收到新样本**时统计/记录: 手柄 72Hz 而循环 500Hz,
                # 重复样本上滤波器只是收敛, 计进去会把"滤除量"稀释掉。
                if hand_seq != last_hand_seq:
                    last_hand_seq = hand_seq
                    if FILTER_ENABLE:
                        diag.hand_seen(float(np.linalg.norm(p_raw - p)) * 1000.0)
                    if hand_log_f is not None:
                        try:
                            hand_log_f.write(
                                f"{sim_time:.4f},"
                                f"{p_raw[0]:.6f},{p_raw[1]:.6f},{p_raw[2]:.6f},"
                                f"{p[0]:.6f},{p[1]:.6f},{p[2]:.6f},"
                                f"{q[0]:.6f},{q[1]:.6f},{q[2]:.6f},{q[3]:.6f}\n")
                        except OSError:      # 磁盘满等: 关掉记录, 不影响遥操作
                            hand_log_f = None

                if squeeze > GRIP_THRESH:
                    if not was_locked:
                        with state.lock:
                            state.locked = True
                            state.hand0_pos = p
                            state.hand0_quat = q
                            state.tcp0_pos = rig.tcp().copy()
                            state.ee0_quat = mat_to_quat(
                                data.site_xmat[site_id].reshape(3, 3))
                        q_ik = None
                        lg.event("锁定", f"手柄={np.round(p,3)} tcp={np.round(rig.tcp(),3)}")
                    else:
                        with state.lock:
                            p0, q0 = state.hand0_pos, state.hand0_quat
                            tcp0, ee0q = state.tcp0_pos, state.ee0_quat
                        tt, tq2 = map_hand_to_target(p, q, p0, q0, tcp0, ee0q)
                        with state.lock:
                            state.target_tcp = tt
                            state.target_quat = tq2
                else:
                    if was_locked:
                        with state.lock:
                            state.locked = False
                        q_ik = None
                        lg.event("解锁", f"tcp={np.round(rig.tcp(),3)}")

            # ---- IK + 下发 ----
            if not reset_animating:
                with state.lock:
                    tp, tq = state.target_tcp, state.target_quat
                    locked = state.locked
                if locked and tp is not None and tq is not None:
                    mujoco.mj_kinematics(model, data)
                    mujoco.mj_comPos(model, data)
                    # 热启动: 传上一帧的解 (与原版"求解后不还原 qpos"等价),
                    # 保证解在分支上连续, 不会因为实际状态滞后而换解
                    q_ik = ik_solve(model, data, site_id, tp, tq, q_seed=q_ik)
                    # 诊断: 解之间的跳变 (大 = 换分支/近奇异) + 限位命中
                    if prev_q_ik is not None:
                        _dq = float(np.abs(q_ik - prev_q_ik).max())
                    else:
                        _dq = 0.0
                    prev_q_ik = q_ik.copy()
                    _lim = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                            for j in lim_js
                            if data.qpos[j] - model.jnt_range[j, 0] < 1e-3
                            or model.jnt_range[j, 1] - data.qpos[j] < 1e-3]
                    diag.tick(None, IK_LAST_POS, IK_LAST_ROT, _dq, _lim)
                    if GRASP_MODE == "physics":
                        # 判据 = **抓取意图** (扣了扳机 / 已经夹住), 而不是"碰到东西"。
                        # 只用"接触"当判据会让"靠近时蹭到物块"也切到伺服 → 伺服滞后
                        # 造成"卡住", 离开接触又跳回 → 用户实测的"卡住再瞬移"。
                        want_physics = (trigger > PHYS_TRIGGER_MIN) or (held is not None)
                        # 连续坡道: 不再有突变点, 交接在 TELEPORT_RAMP 秒内渐变
                        _sw = dt / TELEPORT_RAMP if TELEPORT_RAMP > 1e-6 else 1.0
                        blend = (min(1.0, blend + _sw) if want_physics
                                 else max(0.0, blend - _sw))
                        rig.drive(q_ik, "teleport", blend=blend)
                        drive_mode = ("物理" if blend >= 1.0
                                      else ("过渡" if blend > 0.0 else "瞬移"))
                    else:
                        # 挂载模式: 抓取不依赖接触物理 → 全程瞬移 (无伺服滞后)
                        drive_mode = rig.drive(q_ik, "teleport")
                else:
                    diag.tick(None, None, None, None, None)
                rig.set_gripper(grip_cmd)
                if GRASP_MODE == "attach":
                    held = rig.update_grasp(trigger, held)
                    with state.lock:
                        state.grip_held = held
                    rig.carry()      # 必须在 mj_step 之前: 先把方块摆到夹爪上
            else:
                diag.tick(None, None, None, None, None)

            mujoco.mj_step(model, data)

            if GRASP_MODE == "physics":
                # 抓取判定: 只看物理接触 (左右指垫是否都摸到同一个方块)
                held = rig.detect_grasp()
                with state.lock:
                    state.grip_held = held

            # ---- 运行日志: 把这一帧的状态落盘 ----
            # ★ 整块包 try/except: 日志只是旁观者, 它出任何错都不许弄死遥操作。
            #   (这层不是防患于未然 —— 之前就因为这里引用了已移除的 IK_LAST_MANIP
            #    让整个 mj 线程直接崩掉, 且现场只剩一个 0 字节文件。)
            if lg.enabled:
              try:
                _q = data.qpos[:NQ_ARM]
                _qd = data.qvel[:NQ_ARM]
                _tcp = rig.tcp()
                _tq = mat_to_quat(data.site_xmat[site_id].reshape(3, 3))
                # ★ "没有值"一律记 NaN —— 绝不用 0 代替: 0 是合法值(零点/无解),
                #   混在一起就分不出"没解"和"解在零点"。之前正是这么误判过一次
                #   (qik 从全 0 跳到真实解, 算出 180° 的假跳变)。
                _tgt = (np.asarray(tp, float) if tp is not None
                        else np.full(3, np.nan, np.float32))
                _tqt = (np.asarray(tq, float) if tq is not None
                        else np.full(4, np.nan, np.float32))
                _ik = (np.asarray(q_ik, float) if q_ik is not None
                       else np.full(NQ_ARM, np.nan, np.float32))
                _dc = {"teleport": 0, "瞬移": 0, "servo": 1, "物理": 2}.get(drive_mode, 2)
                # 输入延迟: 距最近一次**收到**新手柄消息多久 (属性读取是原子的, 不必加锁)
                _ht = state.hand_t
                hand_age_ms = (time.time() - _ht) * 1000.0 if _ht else 0.0
                # ★ 手柄位姿只在**侧键按下(跟随中)**时记录; 没按下时记 NaN。
                #   (用户要求: 没跟随时的手柄位姿是噪声, 不用存)
                #   仍保留 squeeze/trigger/locked 与 hand_age_ms —— 否则看不出
                #   "为什么这一行没有手柄数据", 也没法统计网络质量。
                if locked:
                    _hpos, _hquat = hand_pos, hand_quat
                else:
                    _hpos = np.full(3, np.nan, np.float32)
                    _hquat = np.full(4, np.nan, np.float32)
                lg.write(np.concatenate([
                    [sim_time, loop_ms],
                    _q, _qd, data.ctrl[:NQ_ARM], _ik,
                    _tgt, _tqt, _tcp, _tq,
                    [IK_LAST_POS, IK_LAST_ROT, -1.0],   # -1=本版不算可操作度
                    [_dc, blend],
                    [grip_cmd, data.qpos[NQ_ARM],
                     1.0 if held else 0.0],
                    [trigger, squeeze, 1.0 if locked else 0.0],
                    _hpos, _hquat, [hand_age_ms],
                    [loop_ms, sync_cost * 1000.0,       # sync_cost 是**秒**, 记成毫秒
                     1.0 if reset_animating else 0.0],
                ]))
                # 摘要极值 (便宜)
                lgstat["max_qd"] = max(lgstat["max_qd"], float(np.abs(_qd).max()))
                lgstat["max_loop"] = max(lgstat["max_loop"], loop_ms)
                lgstat["max_sync"] = max(lgstat["max_sync"], sync_cost * 1000.0)
                if lgstat["prev_qik"] is not None and np.isfinite(_ik).all():
                    lgstat["max_ik_step"] = max(
                        lgstat["max_ik_step"],
                        float(np.abs(_ik - lgstat["prev_qik"]).max()))
                lgstat["prev_qik"] = _ik.copy()
                lgstat["n_lines"] = lg.n
                # 抓取事件: 临界夹持时 held 会在 4ms 尺度反复翻转(实测 0.21s 内 53 次),
                # 直接记会刷屏。把一串抖动压成一条: 只记第一次 + 结束时汇总次数。
                _fn = getattr(lg, "_flip_n", 0)
                if held != held_prev:
                    if _fn == 0:
                        what = f"夹住 {held}" if held else f"松开 {held_prev}"
                        lg.event("抓取", f"{what}  tcp={np.round(_tcp,3)}")
                    lg._flip_n = _fn + 1
                    lg._flip_t = sim_time
                elif _fn > 0 and sim_time - getattr(lg, "_flip_t", 0.0) > 0.2:
                    if _fn > 1:
                        lg.event("抓取", f"   ↑ 0.2s 内判定抖动 {_fn} 次 "
                                         f"(临界夹持: 两指点时碰时不碰, 只影响判定显示)")
                    lg._flip_n = 0
                held_prev = held
              except Exception as _e:                      # noqa: BLE001
                # 日志坏了就关掉它, 遥操作照常跑 (绝不 re-raise)
                lg.enabled = False
                print(f"[日志] 记录出错, 已关闭日志 (遥操作不受影响): "
                      f"{type(_e).__name__}: {_e}", flush=True)

            if step % 500 == 0:
                tip = data.site_xpos[site_id]
                print(f"[MuJoCo] t={sim_time:6.2f}s locked={state.locked} "
                      f"grip={int(grip_cmd):3d} held={held} "
                      f"tip=({tip[0]:+.3f},{tip[1]:+.3f},{tip[2]:+.3f})")

            # ---- 相机视图: 发布当前 qpos 快照 (显示线程自己渲染) ----
            # 极便宜: 只在 50Hz 拷一次 36 个 float; 显示线程用自己的 MjData,
            # 不会碰这里的 data, 也不阻塞这条 500Hz 的循环。
            camview.publish(data.qpos)

            # ---- 数据集采集: 发布物理状态快照 (只拷贝, 微秒级) ----
            if rec is not None:
                rec.publish(data.qpos, data.ctrl,
                            float(np.clip(data.qpos[NQ_ARM] / GRIP_QMAX, 0.0, 1.0)))

            # viewer 同步。**与原版一致: 固定每 5 步 (100Hz), 不做任何墙钟节流**
            # (节流会把周期性节拍变成相位漂移, 实测掉到 31Hz, 肉眼一卡一卡 —— 见文件顶部说明)
            if step % VIEWER_SYNC_EVERY == 0:
                _t0 = time.perf_counter()
                viewer.sync(state_only=VIEWER_STATE_ONLY)
                _cost = time.perf_counter() - _t0
                sync_cost = _cost
                diag.sync_sum += _cost; diag.sync_max = max(diag.sync_max, _cost)
                diag.sync_n += 1
                diag.sync_seen()
                tcp = rig.tcp()
                mode = "跟随" if state.locked else "冻结"
                viewer.set_texts(
                    (mujoco.mjtFontScale.mjFONTSCALE_150,
                     mujoco.mjtGridPos.mjGRID_TOPLEFT,
                     "VR Teleop - Kinova Gen3 + 2F-85",
                     f"状态: {mode}   驱动: {drive_mode}"
                     f"{'' if blend == 0.0 else f' (物理交接 {blend*100:.0f}%)'}   "
                     f"夹爪: {grip_cmd/255*100:5.1f}%"
                     f"{'   夹住: ' + held if held else ''}\n"
                     f"抓取中心 x={tcp[0]:+.3f} y={tcp[1]:+.3f} z={tcp[2]:+.3f}\n"
                     f"侧键=跟随(A/B=回位)  扳机=夹爪  渲染 {sync_cost*1000:.1f}ms"))
            step += 1

            # 逐秒诊断 (定位"一顿一顿/奇异点")
            if rec is not None:
                diag.ds = rec.short_status()
            rep = diag.maybe_report(state.locked)
            if rep:
                print(rep, flush=True)

    # ---- 正常退出: 写出日志摘要 (幂等, atexit 兜底会再调一次但无害) ----
    if rec is not None:
        rec.stop()
    camview.stop()
    _close_log(_stop_why["why"] or "viewer 关闭/正常退出")
    print("[退出] 收尾完成 (数据集已 finalize, 相机窗口已关, 日志摘要已写)")


# ---------------- 手柄数据可视化 (可选, 默认关闭) ----------------
# 实测: 原来的实现每次重绘要 49ms, 而它 20Hz 刷新 → 约 99% 地占着 GIL,
# 会把仿真线程饿死 (表现就是"非常卡顿、不丝滑")。这是纯调试工具, 默认关闭。
# 需要时设 VIZ_ENABLE=True; 打开后也只更新已有 artist (不再 ax.clear()+全量重画),
# 并且只画最近 VIZ_POINTS 个点。
VIZ_ENABLE = False
VIZ_POINTS = 80
VIZ_HZ = 10


def run_hand_viz(state):
    if not VIZ_ENABLE:
        print("[Viz] 手柄可视化已关闭 (需要时把 server.py 的 VIZ_ENABLE 改成 True)")
        return
    try:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[Viz] matplotlib 不可用, 跳过手柄可视化: {e}")
        return
    plt.ion()
    fig = plt.figure(figsize=(10, 4))
    ax3 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)
    ax3.set_xlim(-0.8, 0.8); ax3.set_ylim(-0.8, 0.8); ax3.set_zlim(-0.8, 0.8)
    ax3.set_xlabel('VR x'); ax3.set_ylabel('VR z'); ax3.set_zlabel('VR y')
    ax3.set_title('hand traj + axes')
    ax2.set_ylim(-0.8, 0.8); ax2.set_title('pos components')
    # 只创建一次 artist, 之后 set_data 更新 (避免重复创建/清空)
    line3, = ax3.plot([], [], [], 'k-', alpha=0.4, lw=1)
    scat = ax3.scatter([], [], [], c='k', s=25)
    ax3.plot([0], [0], [0], 'k+')          # 原点参考
    l_r, = ax2.plot([], [], 'r-', lw=1)
    l_g, = ax2.plot([], [], 'g-', lw=1)
    l_b, = ax2.plot([], [], 'b-', lw=1)
    qv = [ax3.quiver(0, 0, 0, 1, 0, 0, color=c, length=0.15, normalize=True, lw=2)
          for c in ('r', 'g', 'b')]
    print("[Viz] 手柄数据可视化已打开 (增量重绘)")
    while True:
        with state.lock:
            hist = list(state.hand_history)[-VIZ_POINTS:]
        if hist:
            pts = np.array([h[0] for h in hist])
            line3.set_data(pts[:, 0], pts[:, 2])
            line3.set_3d_properties(pts[:, 1])
            scat._offsets3d = (pts[-1:, 0], pts[-1:, 2], pts[-1:, 1])
            l_r.set_data(np.arange(len(pts)), pts[:, 0])
            l_g.set_data(np.arange(len(pts)), pts[:, 2])
            l_b.set_data(np.arange(len(pts)), pts[:, 1])
            ax2.set_xlim(0, max(2, len(pts)))
            q = np.array(hist[-1][1], float)
            if len(q) == 4 and np.linalg.norm(q) > 1e-9:
                R = quat_to_mat(q / np.linalg.norm(q))
                p = pts[-1]
                segs = [[[p[0], p[2], p[1]],
                         [p[0] + R[0, i] * 0.15, p[2] + R[2, i] * 0.15, p[1] + R[1, i] * 0.15]]
                        for i in range(3)]
                for qq, s in zip(qv, segs):
                    qq.set_segments([s])
        try:
            plt.pause(1.0 / VIZ_HZ)
        except Exception:
            break



# ---------------- Web ----------------
# ★ 活动 WebSocket 登记表（收尾时主动关，别让 aiohttp 干等）：
#   aiohttp 的优雅关闭会等所有 handler 返回，而 WS handler 只有对端断开才返回。
#   实测（/tmp/probe_shutdown.py，客户端仍连着）：
#       默认(不传 shutdown_timeout)  → runner.cleanup() 121.9 s
#       客户端先断开                 → 0.0 s
#       shutdown_timeout=1.0        → 2.0 s
#   所以退出时**主动 close**，并把 shutdown_timeout 设小兜底。
_LIVE_WS = set()


async def ws_handler(request):
    ws = web.WebSocketResponse(max_msg_size=8192, timeout=2.0)
    #                          ↑ 只影响"关闭握手"的等待（web_ws.py 里 _timeout 仅用于那一处），
    #                            不影响收消息；默认 10.0s，调小让退出更利索。
    await ws.prepare(request)
    _LIVE_WS.add(ws)
    print("[WS] Quest 已连接")
    state = request.app['state']
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    if data.get('reset'):
                        state.request_reset()
                    else:
                        state.update_hands(data)
                except json.JSONDecodeError:
                    pass
            elif msg.type == WSMsgType.ERROR:
                break
    finally:
        _LIVE_WS.discard(ws)
        print("[WS] Quest 断开")
    return ws


async def index_handler(request):
    return web.FileResponse(os.path.join(ROOT, "index.html"))


@web.middleware
async def access_log(request, handler):
    print(f"[HTTP] {request.method} {request.raw_path} from {request.remote}")
    return await handler(request)


async def main(ds_task=None):
    # ★ 必须在这里(任何线程启动之前、前台线程里)把 GLFW 初始化掉。
    #   否则 viewer / camview / 数采 三路线程会同时首次 glfwInit(),
    #   GLFW 的 X11 后端断言失败 → 整个进程 abort(无 traceback)。见 glutil.py
    from glutil import init_glfw, status as glstatus
    init_glfw()
    print(f"[GL] GLFW 预初始化: {glstatus()}")

    state = SharedState()
    # 启动提问的答案 (None = 本次不采集)。★ 由 __main__ 在**主线程**问好后传进来:
    # 提问绝不能放在下面的子线程里 —— tkinter 的 readline 钩子会让 input() 崩。
    state.ds_task = ds_task

    # ★★ 顺序很重要：**先把端口绑上，再起仿真/数采线程**。
    #   反过来的话，端口被占也只是"晚一点报错"，而仿真线程**已经**构造了
    #   DatasetRecorder → LeRobotWriter._cleanup_orphans() 会删掉
    #   "编号 ≥ 已记录集数"的 parquet/mp4 —— 也就是**另一个实例刚采的集**。
    #   现在端口占不到就立刻退，完全不碰数据集。（2026-10-14 用户裁决）
    app = web.Application(middlewares=[access_log])
    app['state'] = state
    app.router.add_get('/', index_handler)
    app.router.add_get('/ws', ws_handler)

    # ★ 自签证书：第一次跑时自动生成（头显里会提示"不受信任"，点继续即可）
    if not (os.path.exists(CERT) and os.path.exists(KEY)):
        print("[SSL] 没找到 cert.pem/key.pem → 自动生成自签证书 ...")
        import subprocess
        r = subprocess.run([os.path.join(ROOT, "scripts", "make_cert.sh")],
                           capture_output=True, text=True)
        print(r.stdout.strip() or r.stderr.strip())
        if not (os.path.exists(CERT) and os.path.exists(KEY)):
            raise RuntimeError(
                "自签证书生成失败。请手动运行: ./scripts/make_cert.sh\n"
                "（需要 openssl >= 1.1.1）")

    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_ctx.load_cert_chain(CERT, KEY)

    runner = web.AppRunner(app, shutdown_timeout=1.0)   # ★ 兜底（见 _LIVE_WS 那段注释）
    await runner.setup()
    site = web.TCPSite(runner, HOST, PORT, ssl_context=ssl_ctx)
    try:
        await site.start()
    except OSError as e:
        # 端口被占: 最常见的原因是**上一次的服务还开着**(忘了关)。给一条可照抄的命令。
        # ★ 此时仿真/数采线程**还没起** ⇒ 数据集一个字节都没被碰。
        print(f"\n[HTTP] ✗ 端口 {PORT} 起不来: {e}")
        print(f"       多半是上一次的服务还没关。查/关:")
        print(f"         ss -tlnp | grep {PORT}")
        print(f"         pkill -f 'python -u server.py'")
        print(f"       或者换个端口: VR_TELEOP_PORT=8445 ./run.sh")
        print(f"       (数据集未被触碰; 仿真/数采都没启动)")
        raise SystemExit(1) from None
    print(f"[HTTP] 服务已启动: https://<本机IP>:{PORT}")

    # ---- 端口已就绪 ⇒ 现在才起仿真/手柄视图线程（它们会构造写入器、占 GL） ----
    sim_th = threading.Thread(target=run_mujoco, args=(state,), daemon=True)
    sim_th.start()
    threading.Thread(target=run_hand_viz, args=(state,), daemon=True).start()

    # ---- 等停止信号, 然后**等仿真线程自己收尾完**再退出 ----
    # ★ 为什么必须等: 主线程直接退出解释器时, 仿真线程可能正停在 MuJoCo 的 C 代码里
    #   → 解释器拆解 → 段错误 (实测 Ctrl-C 的退出码 -11 SIGSEGV)。
    #   等它跑完 finally/收尾, 关 viewer、停数采、写日志摘要都由它做。
    while not STOP.is_set():
        await asyncio.sleep(0.2)
    if sim_th.is_alive():
        for _ in range(200):            # 最多等 20s
            await asyncio.sleep(0.1)
            if not sim_th.is_alive():
                break
    if sim_th.is_alive():
        print("[退出] ⚠ 仿真线程 20s 还没退出, 直接离开 (可能仍在 MuJoCo C 代码里)")
    else:
        print("[退出] 仿真线程已收尾, 进程干净退出")

    # ★ 主动关掉仍在连的 WebSocket —— **不等头显浏览器**。
    #   否则 aiohttp 的优雅关闭会一直等 handler 返回（WS handler 只有对端断开才返回）。
    #   实测（客户端仍连着）: 默认 121.5 s → 主动关 + 小超时后 ~0.05 s。
    if _LIVE_WS:
        print(f"[退出] 关闭 {len(_LIVE_WS)} 个仍在连的 WebSocket（最多等 1s，不等头显）...",
              flush=True)
        try:
            await asyncio.wait_for(
                asyncio.gather(*[w.close(code=1001, message=b"server exit")
                                 for w in list(_LIVE_WS)], return_exceptions=True),
                timeout=1.0)
        except (asyncio.TimeoutError, TimeoutError):
            print("[退出] WS 关闭握手超时（头显可能已断网/关机）—— 直接继续，不等了")
    await runner.cleanup()
    print("[退出] HTTP 服务已关闭")


if __name__ == "__main__":
    # ★★★ 提问必须在**这里**: 主线程、且在任何线程 / GL 窗口 / 事件循环之前。
    #   放进 run_mujoco(子线程)会让 input() 撞上 tkinter 的 readline 钩子 → SIGILL。
    #   详见 ask_startup_task() 的注释 (gdb 抓到的 Tcl_Panic 堆栈)。
    #
    # ★ Ctrl-C / kill 都要走**干净收尾**: 装信号处理器 → 置 STOP → 主循环跳出 →
    #   关 viewer / 停数采 / 关相机窗口 / 写日志摘要 → 主线程等仿真线程退出后再离开。
    #   不装的话 Ctrl-C 直接把解释器拆掉, 而仿真线程可能停在 MuJoCo 的 C 代码里
    #   → 段错误(实测 -11), 数采与日志都不收尾。
    def _on_signal(signum, _frame):
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = str(signum)
        request_stop(f"收到信号 {name} (Ctrl-C)")

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    try:
        asyncio.run(main(ask_startup_task()))
    except KeyboardInterrupt:
        request_stop("KeyboardInterrupt")
    print("[退出] 已退出。若想确认没有残留: "
          "ss -tlnp | grep 8444 ; pgrep -x ffmpeg")
