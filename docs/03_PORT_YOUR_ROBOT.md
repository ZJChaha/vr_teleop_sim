# 03 · 换成你自己的机械臂

> 这一篇是整个项目最重要的文档。核心思想：**所有"我和机器人有关"的东西都收在
> `robot_profile.py` 里**，`server.py` / `dataset_recorder.py` / `camview.py` 一行都不用改。

---

## 0. 先搞清楚核心需要你的模型提供什么

代码对你的 MuJoCo 模型只有 **4 个硬性要求**：

| # | 要求 | 用在哪 |
|---|---|---|
| 1 | **一个末端 site**（名字任意），挂在夹爪掌部，本地 **-z 指向两指之间** | IK 解算、TCP |
| 2 | **前 N 个执行器 = 机械臂关节位置指令**（弧度，position 执行器） | 写 `data.ctrl[:N]` |
| 3 | **前 N 个 qpos = 机械臂关节角**，紧随其后是夹爪的 qpos | 状态记录、回位 |
| 4 | **一个夹爪执行器**（下标任意），ctrl 在全开/全闭之间线性变化 | 扳机 → 夹爪 |

其余全部可配置：关节数、夹爪自由度、site 名、body 名、home 位姿、TCP 偏移、
相机、可达区、IK 增益。

---

## 1. 准备 MJCF

三条路，推荐第一条：

```bash
# ① MuJoCo Menagerie（最省事，模型质量高）
git clone https://github.com/google-deepmind/mujoco_menagerie
#   例：ur5e / franka_emika_panda / kuka_iiwa_14 / xarm7 / iiwa14 ...

# ② 自己的 URDF → MJCF
#   MuJoCo 能直接 <mujococompile>，或用 mujoco 的 urdf 导入：
python -c "import mujoco; mujoco.MjModel.from_xml_path('robot.urdf')"   # 直接加载 URDF

# ③ 自己写 MJCF（控制精度最高，但工作量大）
```

然后**包一层场景**（桌子、灯、地面、相机）——参考 `assets/scene_grasp.xml`：
它 `<include>` 机器人文件，再往 `worldbody` 里加 `table` / `obj1..4` / `cam_scene`。
把 `include` 和 `meshdir` 都写成**相对本文件**的路径（这样别人 clone 下来就能跑）。

---

## 2. 打印模型结构（找名字用）

```bash
.venv/bin/python tools/probe_model.py
```

它会打印：执行器列表（名字/类型/ctrlrange）、关节列表（名字/量程/限位）、
site 列表（名字/位置）、相机列表、夹爪行程曲线。

**你要从这里抄三样东西**：末端 site 名、夹爪执行器下标、夹爪 qpos 下标。

> 如果模型里**没有**合适的 site：在夹爪掌部加一个，例如
> ```xml
> <site name="ee" pos="0 0 0.06" size="0.005" rgba="1 0 0 0.5"/>
> ```
> 放在夹爪 `base` body 里，`+z` 朝外（两指伸出方向）。代码按 `-z` 算 TCP 偏移，
> 所以 `pos` 的 z 值就是"掌部到指垫中心"的近似距离。

---

## 3. 写你的 profile

```bash
cp robot_profile_example.py robot_profile_myarm.py   # 仓库里带了一份带 TODO 的模板
```

逐个字段填（**只有前 8 项是必填的**）：

```python
from robot_profile import RobotProfile, GripperSpec, CameraSpec

PROFILE = RobotProfile(
    name="ur5e_robotiq",
    model_relpath="assets/ur5e_scene.xml",     # ← 相对仓库根

    # ---- 布局（最容易填错，务必用 probe_model.py 核对）----
    n_arm_actuators=6,      # data.ctrl 前 6 个是关节
    n_arm_qpos=6,           # data.qpos 前 6 个是关节角
    n_gripper_qpos=2,       # 夹爪占几个 qpos（并行夹爪常是 1~2，多指手可能 8+）

    # ---- 末端 ----
    ee_site="ee",           # 你的 site 名
    tcp_offset=0.045,       # site 沿本地 -z 到"抓取中心"的距离（见第 6 节标定）
    gripper_body="robotiq_base",   # 可选，仅 attach 模式用

    # ---- 初始位姿（长度必须 = n_arm_qpos）----
    home_qpos=(0.0, -1.57, 0.0, -1.57, 0.0, 0.0),

    # ---- 夹爪 ----
    gripper=GripperSpec(
        ctrl_index=6,        # 夹爪执行器在 ctrl 里的下标
        ctrl_open=0.0,       # 全开时下发什么
        ctrl_close=0.05,     # 全闭时下发什么（单位随你的执行器）
        qpos_index=6,        # 用哪个 qpos 当"开度"
        qpos_at_open=0.0,    # 该 qpos 全开时的值
        qpos_at_close=0.05,  # 全闭时的值
        boost_ctrl_index=None,   # 没有增压执行器就写 None
    ),

    # ---- 相机 ----
    cameras=[
        CameraSpec("全局相机", "cam_scene", "observation.images.scene"),
        CameraSpec("腕部相机", "wrist", "observation.images.wrist"),
    ],
    camera_overrides={},     # 先用 XML 原值；调好再填（见 docs/04）
)
```

### 关于 `n_gripper_qpos` 和 `qpos_index`

代码假设 **qpos = [臂关节 × N] ++ [夹爪关节 × M]**，且 `qpos_index` 指向
"随开合单调变化"的那一位。参考实现是 `qpos[7]`（2F-85 的 `right_driver_joint`）。

* 单自由度夹爪（很多并行夹爪）：`n_gripper_qpos=1, qpos_index=N`
* Robotiq 2F-85 这种四连杆：`n_gripper_qpos=8, qpos_index=N`（只看 driver 那一位）
* 如果夹爪 qpos **不在** `qpos[N:]` 的位置（比如夹爪关节排在最前面）：
  把 `qpos_index` 写成**绝对下标**即可 —— 代码只拿它去索引 `data.qpos`。

---

## 4. 关节数不是 7（5 轴 / 6 轴）

代码**不假设关节数**，一切按 `n_arm_qpos` 走。但换轴数会连带改变**数据集的形状**，
这一点必须提前想清楚。

### 4.1 要改的字段

```python
PROFILE = RobotProfile(
    n_arm_actuators=6,              # 6 轴就 6
    n_arm_qpos=6,
    n_gripper_qpos=2,               # 按你的夹爪实际 qpos 数
    ee_site="ee",
    home_qpos=(...),                # 长度必须是 6
    gripper=GripperSpec(ctrl_index=6, qpos_index=6, ...),   # 下标跟着前移
    object_bodies=(...),
)
```

**没有**"改关节数开关"，只有这三个数字 + `home_qpos` 长度 + 夹爪下标。

### 4.2 数据集形状会跟着变

`observation.state` / `action` 的维度 = **`n_arm_qpos + 1`**：

| 机器人 | state/action 维度 | 列名 |
|---|---|---|
| 7 轴 + 夹爪 | **8** | `joint_1..joint_7, gripper` |
| 6 轴 + 夹爪 | **7** | `joint_1..joint_6, gripper` |
| 5 轴 + 夹爪 | **6** | `joint_1..joint_5, gripper` |
| 双臂各 6 轴 + 2 夹爪 | 这一版**不采数** | 见 [docs/08](08_DUAL_ARM.md) |

同时 `meta/info.json` 里的 `robot_type` 取的是 **`PROFILE.name`**，所以换机器人后
连"身份"都不一样了。

### 4.3 这意味着三件事

**① 不同轴数的数据集不能混在一起。**
`dataset_recorder._check_compatible()` 会逐项比对 `robot_type / fps / features 的 shape 和 dtype`，
不一致直接拒绝续写。这是**故意的**——混进去等于把数据写坏。

所以换机器人请换目录：

```bash
VR_TELEOP_PROFILE=robot_profile_myarm VR_TELEOP_DS_ROOT=data/lerobot_myarm ./run.sh
```

**② 训练侧要跟着改。**
π0 / OpenPI / LeRobot 的策略配置里都有 `state_dim` / `action_dim`
（或从数据集的 `features` 里推）。6 轴的数据拿去喂 8 维的 checkpoint 会直接报形状错误。
常见做法：

* 从数据集里读 `info.json` 的 `features["observation.state"]["shape"]` 来决定维度
* 或者把不同机器人各自训一个（推荐，混训需要统一动作空间，是另一个话题）

**③ 归一化统计不能跨机器人复用。**
`stats.json` 是按数据集算的，关节范围差很多。别把别人的 `stats.json` 拷过来。

### 4.4 5 轴 / 6 轴还会影响"能不能跟随姿态"

IK 用的是 6 维任务空间（位置 + 姿态）。轴数少于 6 时：

* **6 轴**：一般能同时跟位置和姿态，但腕部构型受限，容易在某些姿态顶限位
* **5 轴**（少一个腕部自由度）：通常**无法**同时满足位置 + 完整姿态
* **4 轴 SCARA / 直角坐标**：只能跟位置 + 绕竖直轴

这种时候把姿态跟随关掉：

```python
teleop=TeleopSpec(follow_orientation=False)   # 只跟位置，末端朝向固定在抓取瞬间
```

IK 的 `mj_jacSite` 会自动按实际自由度（`nv`）算，所以不需要改 IK 代码；
但**姿态残差会一直是大的**，`ik_rot_tol` 也就失去意义了 —— 这是物理限制，不是 bug。

> 判据：跑 `tools/check_profile.py` 之后，手动在 `python` 里把末端设到工作区中心，
> 调一次 `ik_solve`，看 `ik_rot_err`。如果关掉姿态跟随后仍收敛不了，那是位置目标不可达，
> 与轴数无关。

### 4.5 两条臂

两条臂不用写两个 profile，而是在**同一个** `RobotProfile` 里列 `arms`：

```python
arms=[
    ArmSpec(name="left",  drive_hand="left",  joint_offset=0,  actuator_offset=0,  ...),
    ArmSpec(name="right", drive_hand="right", joint_offset=15, actuator_offset=9,  ...),
]
```

只有两个数字要现数：`joint_offset`（第几个关节）和 `actuator_offset`（第几个执行器）。
其余字段和单臂写法一样，`gripper.ctrl_index` 之类的下标**相对本臂**写。

完整步骤、验证方法、以及"为什么双臂暂不采数"见
**[docs/08_DUAL_ARM.md](08_DUAL_ARM.md)**。

---

## 5. 自检（反复迭代这一步）

```bash
VR_TELEOP_PROFILE=robot_profile_myarm .venv/bin/python tools/check_profile.py
```

逐项 ✅ 才算过。典型报错与修法：

| 报错 | 原因 | 修法 |
|---|---|---|
| `site 'xxx' 不存在` | 名字抄错 / site 在 include 的子文件里但没导出 | 用 `probe_model.py` 看真实名字 |
| `home_qpos 在关节限位内` ❌ | 初始位姿越界 | 换一个位姿，或确认该关节确实无限位 |
| `夹爪方向一致` ❌ | `qpos_at_open/close` 填反了 | 交换两个值 |
| `夹爪刻度数值接近实测` ⚠️ | 数值差 > 0.15 | 用 `probe_model.py` 的行程曲线重标 |
| `相机 'x' 可用` ❌ | 模型里没有，也没写进 `extra_cameras` | 加进 `extra_cameras`（见 docs/04） |

---

## 6. 标定 `tcp_offset`（决定"能不能真夹到"）

**错误表现**：机械臂看起来对准了方块，夹爪合上却夹空 / 停在方块上方。

`tcp_offset` = 你那个 site 到**两指垫中心**沿 site 本地 `-z` 的距离。

### 标定方法（三步，不用猜）

```python
# 1) 把 site 放到物体中心，渲染看指垫在哪
# 2) 用碰撞几何直接量：找左右指垫 geom 的中心，取平均
import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path(MODEL)
d = mujoco.MjData(m); mujoco.mj_forward(m, d)
pads = [i for i in range(m.ngeom)
        if "pad" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "")]
centers = np.array([d.geom_xpos[i] for i in pads])
print("指垫中心（世界）:", centers.mean(0))
sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee")
site = d.site_xpos[sid]; z = d.site_xmat[sid].reshape(3, 3)[:, 2]
print("site:", site, " offset =", float((site - centers.mean(0)) @ z))
```

第三个量就是 `tcp_offset`（沿 +z 的量；代码里按 `-z` 减，符号按你觉得顺的方向填，
`check_profile.py` 会打印 home 位姿下的 TCP 位置供你目视核对）。

---

## 7. 选 `home_qpos`

两个原则：

1. **离工作区近**，且在关节限位内有余量（越靠限位，遥操作时越容易"顶死"）。
2. **和你的相机机位匹配**（相机按这个姿态调，见 docs/04）。

找法：

```python
# 从模型自带的 keyframe 开始
mujoco.mj_resetDataKeyframe(m, d, m.key("home").id); print(d.qpos[:N])

# 或者用 IK 把一个舒服的 TCP 位置解出来当 home
```

> 参考实现踩过的坑：夹爪"竖直朝下"看着自然，但 7 轴臂为了朝下需要腕部几乎弯 90°，
> 离限位只剩 0.33 rad，手一拧就顶死（表现为"一顿一顿"）。改成"斜朝前 45°"后
> 余量 0.53 rad，问题消失。**home 姿态不只是好看，它决定了整个工作区的余量。**

---

## 8. 定 `object_random_*`（物块随机区）

```python
object_random_x=(0.35, 0.55),   # 世界坐标，米
object_random_y=(-0.20, 0.20),
object_min_sep=0.12,            # 中心距，给夹爪接近留空间
object_z=0.03,                  # 方块中心高度 = 台面 z + 半高
object_bodies=("cube1", "cube2"),
```

量法（**别只看终点可达，要看整条接近轨迹**）：

```python
# 对每个候选点：从上方 20cm 垂直下探 + 合爪 + 抬升 10cm，全部成功才算可达
# 参考实现见 tools/validate_e2e.py 的 run_pick()
```

两个容易忽略的约束：

* **区域太扁采不出点**：4 个物体两两 ≥ `min_sep`，若区域是 10×60cm 这种长条，
  随机采样可能无解 → 采样器要限次数、超了整组重来（否则死循环）。
* **区域中心不等于桌面中心**：臂的工作区是有限的，桌心往往够不到。
  参考实现桌面中心 x=0.71 但可达上限只有 x≈0.62。

---

## 9. 手柄操作方式完全自定义

**这一套遥操作不绑定任何特定手柄/按键习惯。** 下面每一项都能改，改完重启即可生效。
全部集中在 `robot_profile.py` 的 `TeleopSpec` 里：

```python
from robot_profile import RobotProfile, TeleopSpec

PROFILE = RobotProfile(
    ...
    teleop=TeleopSpec(
        drive_hand="right",              # 用哪只手驱动：right / left
        follow_orientation=True,         # False = 只跟位置，姿态固定在抓取瞬间
        rot_map=((0,0,1),(1,0,0),(0,1,0)),   # 手柄坐标 → 世界坐标
        axis_signs=(-1,-1,1),            # 平移 [前后, 左右, 上下] 的方向
        rot_signs=(-1,1,-1),             # 旋转 [pitch, yaw, roll] 的方向
        scale=1.0,                       # 手移 1m → 末端移几米
        squeeze_thresh=0.5,              # 侧键按下阈值
        trigger_min=0.05,                # 扳机"开始抓"的阈值
        gripper_rate=4.0/255.0,          # 夹爪每个控制步最多变化多少
    ),
)
```

### 9.1 各字段改什么

| 想达到什么 | 改哪个 |
|---|---|
| **换成左手操作** | `drive_hand="left"`（采集按键会自动挪到右手） |
| **只跟位置，不跟姿态**（5 轴 / 腕部受限） | `follow_orientation=False` |
| **手划得太远**，想更省力 | `scale=1.5`（手移 1m → 末端 1.5m） |
| **想更精细** | `scale=0.6` |
| **前后/左右/上下方向反了** | 改 `axis_signs` 对应的那一项为 -1 |
| **手腕俯仰/偏航/翻滚反了** | 改 `rot_signs` 对应项 |
| **侧键太灵敏/太钝** | `squeeze_thresh`（0.5 = 压一半才算按下） |
| **夹爪闭合太猛（有撞击）** | 调小 `gripper_rate` |
| **夹爪跟不上手** | 调大 `gripper_rate` |

### 9.2 怎么判断方向对不对（不用算）

按住侧键，做下面 5 个动作，反了就把对应符号取反：

```
① 把手往"自己身前"推  → 末端应往"远离你"走
② 手往右移            → 末端往右（你的视角）
③ 手往上抬            → 末端往上
④ 手腕俯仰            → 夹爪俯仰同向
⑤ 手腕左右转（偏航）  → 夹爪同向转
```

一次只改一项、改完立刻试，比一次改一堆快得多。

> 参考实现里 `scale=1.0` 是实测挑的；`1.2` 会让人不自觉地把手收得很紧、
> 结果够不到远处。

### 9.3 按键本身

按键号（扳机 = `buttons[0]`、侧键 = `buttons[1]`、A/B = `buttons[4]/[5]`、
左手 X/Y = `buttons[4]/[5]`）在 **`index.html`** 里，因为它在头显浏览器中运行。
想换按键就改那一个文件的 `btns[i]` 下标：

```js
// index.html 约 290 行
if (btns[0]) trigger = btns[0].value;     // 扳机
if (btns[1]) squeeze = btns[1].value;     // 侧键
const b4 = btns[4], b5 = btns[5];         // A/B 或 X/Y
```

不同头显/浏览器的按键编号可能不一样。页面右下角有个调试面板会实时打印
`buttons[i]:pressed/value`，照着它填就行。

### 9.4 想改得更彻底

`map_hand_to_target()`（`server.py`）就是"手柄相对运动 → 末端目标"的全部逻辑，
一共 15 行。想换成别的映射（绝对位置映射、比率控制、带死区的增量映射…）
直接改这个函数即可，其余代码不用动。

---

## 10. 接真机（下一步）

这个仓库**只做仿真**，但抽象是照着真机设计的：

```python
# 需要你做的只有一件事：把"下发位置指令"替换成你的驱动
# 现在（仿真）:  data.ctrl[:N] = q_ik
# 真机（Kortex）: 按 50Hz 把 q_ik 发出去
```

对接要点：

| 项 | 建议 |
|---|---|
| 频率 | **50Hz 足够**（IK 内部 500Hz 只是仿真步进）；更高频率对真机是负担 |
| 指令类型 | 位置指令 + 速度/加速度限制交给控制器，不要在指令侧再做限幅 |
| 关节角 | 注意**无限旋转关节**（Kinova j1/j3/j5/j7）不要按 `ctrlrange` 裁剪 |
| 夹爪 | `[0,1]` 归一化 → 你的 SDK 刻度（Robotiq rFR 0~255） |
| 安全 | 真机必须有独立的限位/碰撞/急停层；**本项目的 `auto` 模式逻辑不能替代安全层** |
| 标定 | `tcp_offset`、`home`、关节零位、方向符号都要在真机上重标 |

> 强烈建议先做"零指令只看不动的空跑"：把 IK 解只打印不发送，确认方向和量级都对。

---

## 11. 换机器人时的常见坑（按出现频率排序）

1. **夹爪 qpos 下标错** → 数据集里夹爪一直是 0 或 1，看不出开合。
   `check_profile.py` 的"方向一致"检查就是为这个。
2. **执行器顺序不是"臂在前"** → 手臂指令被发给夹爪。
   用 `probe_model.py` 确认 `actuator0..N-1` 是关节。
3. **site 在夹爪外 3~4cm** → 抓空。见第 6 节。
4. **home 姿态贴着限位** → 遥操作"一顿一顿"、IK 解跳变。见第 7 节。
5. **模型里有自碰撞 bug**（相邻连杆互相碰撞）→ 抖动、力爆表。
   检查 `<contact><exclude>` 是否把相邻连杆排除了。
6. **桌面/物体与机器人穿模** → 检查物体初始 z 是否等于台面 + 半高。
7. **相机位姿沿用参考实现的** → 画面对不上（相机和几何强相关）。必须重调。
8. **模型用绝对路径 include** → 别人 clone 下来直接崩。全部改相对路径。
