"""机器人 / 任务配置 —— **换一台机械臂，只需要改这一个文件**。

这个文件是整个项目里唯一知道"你用的是什么臂、什么夹爪、什么相机"的地方。
其余模块（server / camview / dataset_recorder）只按这里给出的**索引和名字**工作，
不假设任何具体型号。

用法
----
    # 想保留原配置、另开一套自己的
    cp robot_profile.py robot_profile_myarm.py
    # 改完让程序用它：
    VR_TELEOP_PROFILE=robot_profile_myarm ./run.sh

所有字段的含义和"该填什么"见 docs/PORT_YOUR_ROBOT.md。下面的默认值
= 参考实现（Kinova Gen3 7-DOF + Robotiq 2F-85），**不要直接改**，
复制一份再改，这样升级上游代码时不会冲突。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))


# ============================================================================
# 夹爪
# ============================================================================
@dataclass
class GripperSpec:
    """夹爪在 MuJoCo 模型里的接线方式 + 命令刻度。

    ★ 项目内部统一把夹爪归一化成 **[0,1]，0=全开、1=全闭**（写进数据集的
      observation.state[7] 和 action[7] 都是这个刻度）。所以这里要做两件事：
        1) 把"下发命令"从 [0,1] 映射到模型执行器接受的 ctrl 区间；
        2) 把"读回的关节角"从模型单位映射回 [0,1]。
    """

    # ---- 下发（遥操作 → 仿真）----
    ctrl_index: int = 7            # 夹爪执行器在 data.ctrl 里的下标
    ctrl_open: float = 0.0         # 全开时下发的 ctrl 值
    ctrl_close: float = 255.0      # 全闭时下发的 ctrl 值
    # 可选：额外的"夹持力增强"执行器（参考实现用来补足 2F-85 的 forcerange）
    boost_ctrl_index: int | None = 8
    boost_value: float = 0.0       # 常驻值（0 = 不出力）

    # ---- 读回（仿真 → 数据集）----
    qpos_index: int = 7            # 用 qpos 的哪一位当作"夹爪开度"
    qpos_at_open: float = 0.0      # 全开时该位的值
    qpos_at_close: float = 0.8     # 全闭时该位的值

    def cmd_to_ctrl(self, g01: float) -> float:
        """[0,1] → 执行器 ctrl。"""
        g01 = min(max(float(g01), 0.0), 1.0)
        return self.ctrl_open + g01 * (self.ctrl_close - self.ctrl_open)

    def qpos_to_01(self, qpos_value: float) -> float:
        """模型 qpos → [0,1]（自动处理"开度越大越闭合"还是相反）。"""
        span = self.qpos_at_close - self.qpos_at_open
        if abs(span) < 1e-9:
            return 0.0
        v = (float(qpos_value) - self.qpos_at_open) / span
        return min(max(v, 0.0), 1.0)


# ============================================================================
# 机械臂（一台 / 两台）
# ============================================================================
@dataclass
class ArmSpec:
    """一条机械臂在模型里的接线方式 + 初始位姿。

    单臂时可以完全不写这个类 —— `RobotProfile.arm_specs` 会用上面的
    `n_arm_actuators / ee_site / gripper / home_qpos` 等字段自动合成一条。
    **只有双臂（或两条异构臂）才需要显式写 `arms=[...]`。**

    ★ 索引约定（最容易填错的地方）：
      joint_offset    这条臂的第一个关节在 `model.njnt` 里的下标
      actuator_offset 这条臂的第一个执行器在 `data.ctrl` 里的下标
      gripper.ctrl_index / qpos_index 都是**相对这条臂**的（0 = 这条臂的起点），
                      和单臂时写的绝对值完全一样 —— 因为单臂的 offset 都是 0。

    为什么要 joint_offset 而不是直接写 qpos 下标：
      MuJoCo 的 qpos 下标会被 free joint（方块）、球关节（夹爪的 coupler/follower）
      打乱，从关节号推 qpos/dof 地址才是可靠做法。
    """

    name: str = "arm"                 # 只用于日志/窗口标题
    drive_hand: str = "right"         # 哪个手柄驱动这条臂："right" / "left"

    # ---- 模型索引 ----
    joint_offset: int = 0             # 第一个关节在 model.njnt 里的下标
    actuator_offset: int = 0          # 第一个执行器在 data.ctrl 里的下标
    n_arm_actuators: int = 7
    n_arm_qpos: int = 7
    n_gripper_qpos: int = 8           # 夹爪占用的连续 qpos 数（2F-85 = 8）

    # ---- 末端 ----
    ee_site: str = "pinch"            # 做 IK 的 site
    tcp_offset: float = 0.0142        # 抓取中心相对 site 沿 -z 的偏移（米）
    gripper_body: str = "base"        # 夹爪本体 body（只用于取朝向）

    # ---- 初始位姿 ----
    home_qpos: Tuple[float, ...] = ()

    # ---- 夹爪 ----
    gripper: GripperSpec = field(default_factory=GripperSpec)

    enabled: bool = True

    @property
    def nq_grip_end(self) -> int:
        """这条臂的 qpos 区间长度（关节 + 夹爪关节）。"""
        return self.n_arm_qpos + self.n_gripper_qpos


# ============================================================================
# 相机
# ============================================================================
@dataclass
class CameraSpec:
    """一路相机。

    model_name : MuJoCo 模型里的 <camera name=...>
    lerobot_key: 写进 LeRobot 数据集的列名（observation.images.<x>）
    label      : 预览窗口标题

    两个独立开关（默认都开）：
      preview : 是否开预览窗口（纯看画面，不占数据集体积）
      record  : 是否写进数据集（每一路都要多渲一遍 + 多一轨 mp4）
    ★ 只想要"看着方便"就把 record 关掉；只想要数据就把 preview 关掉。
      这个开关**就是**"设置某个视角打开/关闭"的接口。
    """

    label: str
    model_name: str
    lerobot_key: str
    preview: bool = True
    record: bool = True


# ============================================================================
# 遥操作映射（手柄 → 机械臂）
# ============================================================================
@dataclass
class TeleopSpec:
    """手柄输入到末端运动的映射 —— **可以完全自定义**。

    这个项目对"用什么手柄、怎么映射"没有任何硬性假设：
    下面每一项都能改，也可以只用其中一部分（比如改成 6 轴就只用位置、不跟随姿态）。

    ⚠ 只有**按键号**（扳机是 button[0]、侧键是 button[1] …）在 `index.html` 里，
      因为它跑在头显浏览器里。想改按键定义就改那一个文件的 `btns[i]` 下标。
    """

    # ---- 用哪只手操作 ----
    # "right" / "left"：右手柄按侧键驱动；另一只手只用来打标签（采集模式）
    drive_hand: str = "right"

    # ---- 坐标系映射（3×3 旋转矩阵，行 = 世界轴，列 = 手柄轴）----
    # 默认值是把 WebXR 的 (x右, y上, z后) 转成"x前, y左, z上"的世界系。
    # 换头显/换手柄姿态约定时改这里，或直接用下面的 *_signs 微调。
    rot_map: Tuple[Tuple[float, float, float], ...] = (
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    # 平移方向符号 [前后, 左右, 上下] —— 某个轴反了就把它改成 -1
    axis_signs: Tuple[float, float, float] = (-1.0, -1.0, 1.0)
    # 姿态方向符号 [pitch, yaw, roll]
    rot_signs: Tuple[float, float, float] = (-1.0, 1.0, -1.0)
    # 手柄位移 → 末端位移 的比例。1.0 = 一比一；调大更省手，调小更精细
    scale: float = 1.0

    # ---- 是否跟随姿态 ----
    # False = 只跟随位置，末端姿态始终保持在抓取瞬间的朝向（6 轴/5 轴腕部不够灵活时很实用）
    follow_orientation: bool = True

    # ---- 按键阈值 ----
    squeeze_thresh: float = 0.5   # 侧键按下判定（超过 = 开始跟随）
    trigger_min: float = 0.05     # 扳机超过此值 = 有抓取意图（切换物理驱动）

    # ---- 夹爪命令平滑（每步最大变化，单位 = 归一化行程度 0~1）----
    # 太大会"啪"一下闭合（接触冲击大），太小会跟不上手的动作
    gripper_rate: float = 4.0 / 255.0


# ============================================================================
# 机器人
# ============================================================================
@dataclass
class RobotProfile:
    # ---- 基本信息 ----
    name: str = "kinova_gen3_robotiq_2f85"
    description: str = "Kinova Gen3 (7-DOF) + Robotiq 2F-85，桌面抓取参考实现"

    # ---- 模型 ----
    # 相对本文件，或绝对路径。可用环境变量 VR_TELEOP_MODEL 覆盖。
    model_relpath: str = "assets/scene_grasp.xml"

    # ---- 关节 / 执行器布局（**最容易填错的地方**）----
    # data.ctrl 的前 n_arm_actuators 个 = 机械臂关节位置指令（弧度）
    n_arm_actuators: int = 7
    # data.qpos 的前 n_arm_qpos 个 = 机械臂关节角
    n_arm_qpos: int = 7
    # 夹爪在 qpos 里占用的连续区间 [n_arm_qpos, n_arm_qpos + n_gripper_qpos)
    n_gripper_qpos: int = 8

    # ---- 末端 ----
    # 用来做 IK 的 site（它在模型里通常挂在夹爪掌部）
    ee_site: str = "pinch"
    # 真正的"抓取中心"相对该 site 沿其本地 -z 的偏移（米）。
    # 参考实现 = 0.0142（两指垫中心），不填 0 的原因见 docs/PORT_YOUR_ROBOT.md
    tcp_offset: float = 0.0142
    # 夹爪本体 body 名（只用于取朝向/挂载方案，纯物理抓取不用）
    gripper_body: str = "base"

    # ---- 初始位姿（关节角，弧度；长度 = n_arm_qpos）----
    # 遥操作开始时机械臂停在这里，A/B 键也回这里。
    home_qpos: Tuple[float, ...] = (
        0.65963298, 0.56115198, 3.36832309, -1.42121601,
        8.41883945, 1.50929797, -5.84495401,
    )

    # ---- IK / 控制 ----
    ik_iterations: int = 20          # 每次调用的 DLS 迭代次数（外层还会反复热启动）
    ik_pos_tol: float = 1e-4         # 收敛判据（米）
    ik_rot_tol: float = 2e-3         # 收敛判据（弧度）
    ik_damping: float = 0.005        # 阻尼系数 λ（越大越稳、越慢）
    ik_max_dq: float = 0.04          # 单次迭代每个关节最多走多少弧度

    # ---- 夹爪 ----
    gripper: GripperSpec = field(default_factory=GripperSpec)

    # ---- 机械臂 ----
    # 单臂：留空即可（会自动用上面那些字段合成一条）。
    # 双臂：在这里列两条 ArmSpec，见 robot_profile_dual_example.py 与 docs/08。
    # ★ 一旦 arms 非空，上面那组单臂字段（n_arm_* / ee_site / home_qpos / gripper）
    #   就不再参与控制 —— 只保留给"单臂路径"和数据集维度推导。
    arms: List[ArmSpec] = field(default_factory=list)

    # ---- 相机 ----
    # 每一路都有 preview（是否显示窗口）和 record（是否写进数据集）两个开关。
    # 数据集通常**不需要三路全开**：腕部 + （侧面全局 or 俯视全局）各一路就够。
    # 见 docs/04 第 2 节"给数据集选哪几路"。
    cameras: List[CameraSpec] = field(default_factory=lambda: [
        CameraSpec("全局相机", "cam_scene", "observation.images.scene"),
        CameraSpec("腕部相机", "wrist", "observation.images.wrist"),
        CameraSpec("全局俯视相机", "cam_top", "observation.images.top"),
    ])
    # 模型里**没有**、需要在内存里注入的相机：{名字: {pos, quat, fovy}}
    # 参考实现用它加"全局俯视相机"，一个 XML 都不改。
    extra_cameras: Dict[str, dict] = field(default_factory=lambda: {
        "cam_top": dict(pos=(0.062000, 0.000000, 1.255000),
                        quat=(0.68689282, 0.16786381, -0.16786381, -0.68689282),
                        fovy=46.5),
    })
    # 在内存里覆盖已有相机的绝对位姿：{名字: {pos, quat, fovy}}
    # 换了机器人，这些值必须重调（相机位姿和机械臂几何强相关）。置 {} 就用 XML 原值。
    camera_overrides: Dict[str, dict] = field(default_factory=lambda: {
        "wrist": dict(pos=(0.0, -0.23639, -0.058475),
                      quat=(0.0, 0.0, -0.32556815, 0.94551858),
                      fovy=78.0),
        "cam_scene": dict(pos=(-0.180000, 0.660000, 0.970000),
                          quat=(0.37378155, 0.19549804, -0.42021259, -0.80342346),
                          fovy=52.0),
        "cam_top": dict(pos=(0.062000, 0.000000, 1.255000),
                        quat=(0.68689282, 0.16786381, -0.16786381, -0.68689282),
                        fovy=46.5),
    })
    # 关掉的渲染通道（给多路相机腾时间）。参考实现实测：3 路 640x480 时
    # 不关 → 37Hz；关 shadow+reflection → 50Hz。置 () 恢复默认渲染。
    render_off: Tuple[str, ...] = ("shadow", "reflection")

    # ---- 遥操作映射（手柄 → 机械臂）----
    # 换手、换轴、改灵敏度、只跟位置不跟姿态 —— 都在 TeleopSpec 里
    teleop: TeleopSpec = field(default_factory=TeleopSpec)

    # ---- 被操作物体（任务相关，换机器人一般要改）----
    object_bodies: Tuple[str, ...] = ("obj1", "obj2", "obj3", "obj4")
    object_random_x: Tuple[float, float] = (0.46, 0.62)
    object_random_y: Tuple[float, float] = (-0.155, 0.155)
    object_min_sep: float = 0.12
    object_z: float = 0.240

    # ---- 派生值 ----
    @property
    def model_path(self) -> str:
        env = os.environ.get("VR_TELEOP_MODEL")
        if env:
            return os.path.abspath(env)
        return self.model_relpath if os.path.isabs(self.model_relpath) \
            else os.path.join(ROOT, self.model_relpath)

    @property
    def nq_grip_end(self) -> int:
        return self.n_arm_qpos + self.n_gripper_qpos

    @property
    def state_dim(self) -> int:
        """π0 关节空间维度 = 关节数 + 1（夹爪）。"""
        return self.n_arm_qpos + 1

    @property
    def joint_names(self) -> List[str]:
        return [f"joint_{i}" for i in range(1, self.n_arm_qpos + 1)] + ["gripper"]

    # ---- 机械臂列表 ----
    @property
    def arm_specs(self) -> List[ArmSpec]:
        """实际参与控制的臂。单臂配置时由标量字段自动合成。"""
        if self.arms:
            return [a for a in self.arms if a.enabled]
        return [ArmSpec(
            name="arm",
            drive_hand=self.teleop.drive_hand,
            joint_offset=0,
            actuator_offset=0,
            n_arm_actuators=self.n_arm_actuators,
            n_arm_qpos=self.n_arm_qpos,
            n_gripper_qpos=self.n_gripper_qpos,
            ee_site=self.ee_site,
            tcp_offset=self.tcp_offset,
            gripper_body=self.gripper_body,
            home_qpos=tuple(self.home_qpos),
            gripper=self.gripper,
        )]

    @property
    def n_arms(self) -> int:
        return len(self.arm_specs)

    @property
    def is_dual_arm(self) -> bool:
        return self.n_arms > 1


# ============================================================================
# 载入
# ============================================================================
_PROFILE_CACHE = {}


def load_profile(module_name: str | None = None) -> RobotProfile:
    """载入配置。

    优先级：显式参数 > 环境变量 VR_TELEOP_PROFILE > 本文件里的 PROFILE。
    环境变量写的是**模块名**（不含 .py），例如 `VR_TELEOP_PROFILE=robot_profile_myarm`。
    """
    key = module_name or os.environ.get("VR_TELEOP_PROFILE") or "__default__"
    if key in _PROFILE_CACHE:
        return _PROFILE_CACHE[key]
    if key == "__default__":
        prof = PROFILE
    else:
        import importlib
        mod = importlib.import_module(key)
        prof = getattr(mod, "PROFILE", None)
        if prof is None:
            raise RuntimeError(f"{key}.py 里没有定义 PROFILE")
        print(f"[机器人] 已载入自定义配置: {key}  ({prof.name})")
    _PROFILE_CACHE[key] = prof
    return prof


#: 默认配置 —— 参考实现（Kinova Gen3 + Robotiq 2F-85）
PROFILE = RobotProfile()
