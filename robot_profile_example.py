"""示例：把这一份复制成你自己的配置，改里面的 TODO，然后用环境变量指过去。

    cp robot_profile_example.py robot_profile_myarm.py
    # 改 robot_profile_myarm.py
    VR_TELEOP_PROFILE=robot_profile_myarm .venv/bin/python tools/check_profile.py

★ 这里的默认值就是参考实现（Kinova Gen3 + Robotiq 2F-85），所以**不改也能跑** ——
  它的作用是让你看到"哪些字段是机器人相关的"，而不是逼你填写。
"""
from robot_profile import CameraSpec, GripperSpec, RobotProfile, TeleopSpec

# ---------------------------------------------------------------------------
# 下面所有数字都来自参考实现。换成你的机器人时，逐个替换：
# ---------------------------------------------------------------------------
PROFILE = RobotProfile(
    # ---- 基本信息 ----
    name="myarm",                          # TODO: 你的名字（只用于打印/日志）
    description="把这里改成你的机器人说明",

    # ---- 模型 ----
    # 相对仓库根目录（推荐）。也可以写绝对路径，或用 VR_TELEOP_MODEL 覆盖。
    model_relpath="assets/scene_grasp.xml",   # TODO: 你的 MJCF/URDF

    # ---- 关节 / 执行器布局（**最容易填错**，用 tools/probe_model.py 核对）----
    # data.ctrl 的前 N 个 = 机械臂关节位置指令
    n_arm_actuators=7,                     # TODO: 6 轴臂就填 6
    # data.qpos 的前 N 个 = 机械臂关节角
    n_arm_qpos=7,                          # TODO
    # 夹爪在 qpos 里占几个（并行夹爪常 1~2；2F-85 这种四连杆是 8）
    n_gripper_qpos=8,                      # TODO

    # ---- 末端 ----
    ee_site="pinch",                       # TODO: 你的 site 名（probe_model.py 会列出）
    tcp_offset=0.0142,                     # TODO: site 沿自身 -z 到"两指中心"的距离（米）
    gripper_body="base",                   # TODO: 夹爪根 body 名（仅 attach 模式用）

    # ---- 初始位姿（长度必须 = n_arm_qpos；必须在关节限位内）----
    home_qpos=(0.65963298, 0.56115198, 3.36832309,
               -1.42121601, 8.41883945, 1.50929797, -5.84495401),   # TODO

    # ---- IK（先别动，除非你知道为什么要动）----
    ik_iterations=20,
    ik_pos_tol=1e-4,
    ik_rot_tol=2e-3,
    ik_damping=0.005,
    ik_max_dq=0.04,

    # ---- 夹爪 ----
    # 内部统一 [0,1]（0=全开, 1=全闭），这里把它映射到你的执行器刻度。
    gripper=GripperSpec(
        ctrl_index=7,          # TODO: 夹爪执行器在 ctrl 里的下标
        ctrl_open=0.0,         # TODO: 全开时下发什么
        ctrl_close=255.0,      # TODO: 全闭时下发什么
        boost_ctrl_index=8,    # TODO: 没有"增压执行器"就写 None（**不要留 -1**）
        boost_value=0.0,
        qpos_index=7,          # TODO: 用哪个 qpos 当开度（绝对下标）
        qpos_at_open=0.0,      # TODO: 该 qpos 全开时的值
        qpos_at_close=0.8,     # TODO: 全闭时的值
    ),

    # ---- 相机（模型里有的直接列；没有的写进 extra_cameras）----
    # 每一路两个开关：
    #   preview = 是否开预览窗口（不影响数据）
    #   record  = 是否写进数据集（每多一路 = 体积+渲染开销）
    # ★ 视频数据集建议只录两路：腕部 + 侧面全局/俯视全局 二选一。
    #   （下面三项都开着 record，是"参考实现"的默认；正式采集时按需关掉一路）
    cameras=[
        CameraSpec("全局相机", "cam_scene", "observation.images.scene",
                   preview=True, record=True),      # TODO: 不要侧面全局就 record=False
        CameraSpec("腕部相机", "wrist", "observation.images.wrist",
                   preview=True, record=True),      # 一般必留
        CameraSpec("全局俯视相机", "cam_top", "observation.images.top",
                   preview=True, record=True),      # TODO: 不要俯视就 record=False
    ],

    # ---- 遥操作映射（手柄 → 机械臂）：**可以完全自定义** ----
    # 哪只手驱动、轴方向、灵敏度、是否跟随姿态 —— 都在这里。
    # 5 轴/6 轴腕部不够灵活时，把 follow_orientation 设 False（只跟位置）。
    teleop=TeleopSpec(
        drive_hand="right",            # TODO: 想用左手操作就写 "left"
        follow_orientation=True,       # TODO: 5 轴 / 腕部受限 → False
        rot_map=((0.0, 0.0, 1.0),
                 (1.0, 0.0, 0.0),
                 (0.0, 1.0, 0.0)),     # TODO: 换头显/换手柄姿态约定时才需要改
        axis_signs=(-1.0, -1.0, 1.0),  # TODO: 前后/左右/上下 反了就把对应项改成 -1
        rot_signs=(-1.0, 1.0, -1.0),   # TODO: pitch/yaw/roll 反了同理
        scale=1.0,                     # TODO: 手移 1m → 末端移几米（大=省力，小=精细）
        squeeze_thresh=0.5,            # 侧键按下判定
        trigger_min=0.05,              # 扳机"开始抓"的阈值
        gripper_rate=4.0 / 255.0,      # 夹爪每步最大变化（小=更柔和）
    ),

    # ---- 被操作物体（任务相关，换机器人基本都要改）----
    object_bodies=("obj1", "obj2", "obj3", "obj4"),
    object_random_x=(0.46, 0.62),
    object_random_y=(-0.155, 0.155),
    object_min_sep=0.12,
    object_z=0.240,
)
