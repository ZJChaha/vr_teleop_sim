# 08 · 双臂

这一版支持**双臂遥操作**：左手柄驱动一条臂、右手柄驱动另一条，两条臂各自跟随、
互不影响。但**双臂不做数据采集**（原因见第 7 节）。

如果你只要一台臂，完全不用看这篇 —— 单臂路径一行代码都没变。

> **仓库不自带双臂模型。** 双臂的几何差异太大（臂长、关节数、夹爪形式、
> 两个基座的间距和朝向），给一个"示意模型"只会让人误以为照着改就行。
> 这一版只提供**接口**：你把两条臂放进一个 MJCF，按第 2 节填 profile 即可。
> 第 9 节给了一份可以直接抄的 MJCF 骨架。

## 1. 这一版支持到什么程度

| | 单臂 | 双臂 |
|---|---|---|
| 手柄跟随 | 一只手（`teleop.drive_hand`） | **两只手，一手一条臂** |
| 夹爪 | 跟扳机 | 各自跟各自的扳机 |
| 姿态跟随 | `follow_orientation` | 同一套（两条臂共用） |
| 相机窗口 | 任意路数 | 同左，照常工作 |
| 回位（A/B） | 一条臂 → home | **两条臂 → 各自的 home** |
| 数据采集 | 支持 | ❌ 不支持（见第 7 节） |
| 物块挂载（attach） | 支持 | ❌ 只有纯物理接触 |
| 运行日志（logs/*.bin） | 支持 | ❌ 列是按单臂 7 关节定的 |

## 2. 三步加进来

### 第 1 步：两条臂放进同一个 MJCF

MuJoCo 一个模型可以有任意多条臂，只要**关节/执行器不共用**。两种做法：

```xml
<!-- 做法 A：手写两个 body 树，名字加前缀 -->
<body name="l_base"> … <site name="l_tcp"/> … </body>
<body name="r_base"> … <site name="r_tcp"/> … </body>

<!-- 做法 B：include 两次（需要 MuJoCo 支持 prefix，或自己写两遍）-->
```

⚠ 两条臂的 `site` 名、关节名、body 名**必须唯一**。MuJoCo 允许重名，但
`mj_name2id` 只返回第一个找到的 —— 两条臂会抢同一个 site。

没有任何双臂 MJCF 时，从第 9 节的骨架抄起。写完先让程序把索引表打出来：

```bash
python tools/list_model.py path/to/your_dual.xml
```

它会列出每个关节 / 执行器的下标、qpos 与 dof 地址、限位、传动类型，
以及所有 site / camera / body 名 —— 第 2 步要填的数字全部来自这张表。

### 第 2 步：数两个数，写 profile

只用数两个东西：

```
joint_offset     这条臂的第一个关节是 model.njnt 里的第几个
actuator_offset  这条臂的第一个执行器是 model.nu 里的第几个
```

`qpos_index` / `ctrl_index` 这些**相对本臂**写（和单臂时一模一样），
qpos / dof 的真实地址由程序从 `jnt_qposadr` / `jnt_dofadr` 推出来 ——
所以方块 free joint、夹爪球关节怎么打乱顺序都不影响。

```python
from robot_profile import ArmSpec, GripperSpec, RobotProfile

GRIP = GripperSpec(ctrl_index=7, ctrl_open=0.0, ctrl_close=255.0,
                   boost_ctrl_index=None, qpos_index=7,
                   qpos_at_open=0.0, qpos_at_close=0.8)

PROFILE = RobotProfile(
    name="my_dual_arm",
    model_relpath="assets/dual_arm.xml",
    arms=[
        ArmSpec(name="left",  drive_hand="left",
                joint_offset=0,  actuator_offset=0,
                n_arm_actuators=7, n_arm_qpos=7, n_gripper_qpos=8,
                ee_site="l_pinch", tcp_offset=0.0142, gripper_body="l_base",
                home_qpos=(… 7 个 …), gripper=GRIP),
        ArmSpec(name="right", drive_hand="right",
                joint_offset=15, actuator_offset=9,     # ← 数出来的
                n_arm_actuators=7, n_arm_qpos=7, n_gripper_qpos=8,
                ee_site="r_pinch", tcp_offset=0.0142, gripper_body="r_base",
                home_qpos=(… 7 个 …), gripper=GRIP),
    ],
    cameras=[…],      # 和单臂一样
    object_bodies=(), # 纯遥操作可以先不填
)
```

关节数不同的两条臂（比如 7 轴 + 6 轴）也支持：各自写自己的 `n_arm_qpos` /
`home_qpos` / `n_gripper_qpos` 即可。**但液压/舵机夹爪混用时** `ctrl_open/close`
要按各自的实际刻度写。

**没有夹爪的臂**（纯定位任务）把 `n_gripper_qpos` 写成 `0`：夹爪相关的
下标检查、标定、下发全部跳过，其余照常。

### 第 3 步：自检 + 无头验证

```bash
# ① 索引、site、home 位姿、区间重叠（不连头显，不写文件）
VR_TELEOP_PROFILE=robot_profile_mydual python tools/check_profile.py

# ② 假手柄消息驱动两条臂，验独立性（不需要显示器）
VR_TELEOP_PROFILE=robot_profile_mydual python tools/validate_dual.py

# ③ 真跑
VR_TELEOP_PROFILE=robot_profile_mydual ./run.sh
```

`validate_dual.py` 会检查五件事，全绿再连头显：

```
[1] 索引解析        qpos/ctrl 区间在模型内、互不重叠
[2] 跟随正确性      手柄移 5cm → 末端到 map_hand_to_target 算出的目标点
[3] 独立性          只动左手柄时，右臂关节变化 < 静止基线漂移
[4] 夹爪独立        左手扣扳机只动左臂夹爪执行器
[5] 双臂同时        两手各移 4cm，两臂位移方向不同（没串成一件事）
```

参考输出（两条 6 轴臂 + 单自由度夹爪的实测；你的模型数字会不同，
但**五项都应该是 ✅**）：

```
 ✅ left  site/qpos/ctrl 均在模型内  —  qpos[0:7) ctrl[0:6) site=0
 ✅ right site/qpos/ctrl 均在模型内  —  qpos[7:14) ctrl[7:13) site=1
 ✅ 末端到达映射目标点  —  目标=[-0.368,-0.05,1.252] 实到=… 误差=0.0mm
 ✅ 右臂未被带动  —  Δq=1.22e-07 rad
 ✅ 左臂夹爪收到接近闭合的指令  —  ctrl[6]=0.0350
 ✅ 两臂位移方向不同  —  cos=+0.00  左Δ=[0,-40,0]mm  右Δ=[0,0,-40]mm
```

## 3. 控制器怎么分

**不使用 `teleop.drive_hand`**：双臂时以每条 `ArmSpec.drive_hand` 为准。
两条臂写同一个手会在启动时直接报错退出（两个人抢一个手柄没法独立控制）。

```
左臂 drive_hand="left"  →  左手侧键 = 左臂跟随，左扳机 = 左夹爪
右臂 drive_hand="right" →  右手侧键 = 右臂跟随，右扳机 = 右夹爪
```

头显那一侧不需要改任何东西：`index.html` 本来就把两只手的位姿都发给服务端了。
校准参考系（按下侧键那一瞬间用头显朝向建立坐标系）现在**两只手都会触发**，
所以哪只手先按都行。

## 4. 两条臂共用什么、各自有什么

| | 共用 | 各自独立 |
|---|---|---|
| 手柄映射 | `rot_map` / `axis_signs` / `rot_signs` / `scale` / `follow_orientation` | — |
| 跟随状态 | — | 锁定/解锁、锚点（按下侧键那一刻的位姿） |
| 控制 | — | IK 热启动种子、teleport↔物理的 blend |
| 夹爪 | — | `grip_cmd`（各自的扳机） |
| 回位 | — | 各自的 `home_qpos` |

一个人两只手的坐标系约定是一样的，所以映射参数不分开。如果你确实需要左右手
不同的灵敏度，改 `server.py` 里 `map_hand_to_target` 的调用处传不同参数即可
（`ArmRuntime.on_hand` 是唯一调用点）。

## 5. 一帧里发生什么

```
读两只手柄 (state.right / state.left)
      ↓
每条臂各自：侧键锁定？→ 记锚点 / 映射目标
      ↓
每条臂各自：IK 解 q → 按 blend 写 qpos/ctrl → 写夹爪
      ↓
mj_step 一次（两条臂在同一个物理世界里）
```

`ik_solve` 内部会**保存并还原整份 qpos**，所以先算哪条臂都不影响另一条
（`tools/validate_dual.py` 的 [3] 就是在验这一点）。

## 6. 性能

两条臂 = 每帧两次 IK + 两次 `mj_kinematics`。实测**两条 6 轴臂 + 单自由度夹爪**
在 500Hz 下能跑满；**两条 7 轴臂 + 三路相机**会更吃紧，需要注意：

* 关掉 `render_off` 未覆盖的渲染通道（参考实现是 `("shadow","reflection")`）
* 相机只开 `preview` 需要的路数，`record` 反正双臂用不上
* 实在跟不上就降 `timestep` 或减少 IK 迭代（`ik_iterations`）

## 7. 为什么双臂不采数

LeRobot 的数据集特征是按**一台臂 + 一个夹爪**定的：

```
observation.state = [joint_1 … joint_7, gripper]     # 8 维
action           = [joint_1 … joint_7, gripper]     # 8 维
```

两条臂要变成 14+2 维（或拆成两组 8 维），这会带来三个必须一起决定的问题：

1. **特征维度变了** ⇒ `meta/info.json` 不兼容，旧数据集不能续写、不能混训
2. **动作空间语义变了** ⇒ π0 / OpenPI 的 `state_dim` / `action_dim` 要跟着改，
   微调脚本的归一化统计（`stats.json`）也要重算
3. **时序对齐** ⇒ 两条臂各自 50Hz 采样但网络到达不同步，要决定"以谁的时间轴为准"

这三点属于**训练格式设计**，不是遥操作能替用户拍板的。所以这一版只做遥操作，
把接口留好；等你确定用哪种格式（拼接 / 双流 / 主从），在
`dataset_recorder.py` 里加一个 `DualArmRecorder` 即可，`server.py` 那边只需要
在 `step_arms` 之后多一行 publish。

## 8. 常见坑

**① 两条臂抢同一个 site / body**
MuJoCo 允许重名，`mj_name2id` 只返回第一个。症状是"右手柄动的是左臂"。
→ `check_profile.py` 会打印每条臂解析到的 `site id`，两个 id 一样就是这个问题。

**② `joint_offset` 数错**
关节列表里可能夹着夹爪的 coupler/follower 关节、方块 free joint。
不要数 `<body>`，要数 `<joint>`（`model.njnt`）。
→ 用 `tools/list_model.py` 打印完整列表，或看 `check_profile.py` 报的
`joint[start:end)`。数的是 `<joint>`（含夹爪的 coupler/follower、方块的 free joint），
不是 `<body>`。

**③ 两条臂的 qpos 区间重叠**
第二条臂的 `joint_offset` 写成了第一条臂的关节数（忘了算夹爪关节）。
→ `check_profile.py` 的「各臂 qpos 区间互不重叠」会直接指出来。

**④ 只跟位置还是也跟姿态**
两条臂共用一个 `follow_orientation`。6 轴臂建议 `False`；7 轴可以试 `True`，
IK 收敛不了（末端抖/求解超时）再退回 `False`。判断依据是
`validate_dual.py` 的 [2] 误差和实际观感。

**⑤ 把 `n_gripper_qpos` 写成 0 之后仍然填了 `gripper`**
这不报错（夹爪相关直接跳过），但你会以为夹爪在动 —— 实际上那条臂夹爪永远不动。
→ 没有夹爪就把 `n_gripper_qpos=0`，并且不要指望扳机有反应。

**⑥ 回位后胳膊撞在一起**
两条臂的 home 位姿是各写各的，程序**不检查自碰撞**。把两条臂的 home 摆到
互不干涉的位置（工程上让它们在桌子两侧成镜像，最省事）。

## 9. 写一个双臂 MJCF（骨架）

仓库不提供双臂模型，但骨架是通用的。下面这份**已实测能编译、能被
`check_profile.py` 和 `validate_dual.py` 全部通过** —— 把它存成自己的 XML，
按真实臂改关节长度/限位，再用 `tools/list_model.py` 核一遍下标。

```xml
<mujoco model="my_dual_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast"/>

  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.05" pos="0 0 0"/>
    <geom name="table" type="box" size="0.6 0.4 0.02" pos="0 0 0.72"/>

    <!-- ============ 左臂：所有名字加 l_ 前缀 ============ -->
    <body name="l_base" pos="-0.35 0 0.78">
      <joint name="l_j1" axis="0 0 1" range="-3.1 3.1"/>
      <geom type="cylinder" size="0.05 0.04"/>
      <body name="l_link1" pos="0 0 0.06">
        <joint name="l_j2" axis="0 1 0" range="-3.0 3.0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.22" size="0.03"/>
        <body name="l_link2" pos="0 0 0.22">
          <joint name="l_j3" axis="0 1 0" range="-3.0 3.0"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.20" size="0.026"/>
          <body name="l_link3" pos="0 0 0.20">
            <joint name="l_j4" axis="0 0 1" range="-3.1 3.1"/>
            <joint name="l_j5" axis="0 1 0" range="-2.2 2.2"/>
            <joint name="l_j6" axis="1 0 0" range="-2.2 2.2"/>
            <geom type="sphere" size="0.03"/>
            <!-- 手指：单自由度 slide（没有夹爪就删掉这个 body 和它的 joint） -->
            <body name="l_finger" pos="0 0 0.03">
              <joint name="l_grip" type="slide" axis="1 0 0" range="0 0.04" damping="2"/>
              <geom type="box" size="0.006 0.025 0.015" pos="0 0 0.012"/>
            </body>
            <!-- ★ 必须有一个 site 给 IK 用（ee_site 填这个名字） -->
            <site name="l_tcp" pos="0 0 0.06" size="0.008"/>
          </body>
        </body>
      </body>
    </body>

    <!-- ============ 右臂：同一套结构，名字换 r_ 前缀，x 取反 ============ -->
    <body name="r_base" pos="+0.35 0 0.78">
      <joint name="r_j1" axis="0 0 1" range="-3.1 3.1"/>
      <geom type="cylinder" size="0.05 0.04"/>
      <body name="r_link1" pos="0 0 0.06">
        <joint name="r_j2" axis="0 1 0" range="-3.0 3.0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.22" size="0.03"/>
        <body name="r_link2" pos="0 0 0.22">
          <joint name="r_j3" axis="0 1 0" range="-3.0 3.0"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.20" size="0.026"/>
          <body name="r_link3" pos="0 0 0.20">
            <joint name="r_j4" axis="0 0 1" range="-3.1 3.1"/>
            <joint name="r_j5" axis="0 1 0" range="-2.2 2.2"/>
            <joint name="r_j6" axis="1 0 0" range="-2.2 2.2"/>
            <geom type="sphere" size="0.03"/>
            <body name="r_finger" pos="0 0 0.03">
              <joint name="r_grip" type="slide" axis="-1 0 0" range="0 0.04" damping="2"/>
              <geom type="box" size="0.006 0.025 0.015" pos="0 0 0.012"/>
            </body>
            <site name="r_tcp" pos="0 0 0.06" size="0.008"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <!-- 位置伺服：每条臂 6 个关节 + 1 个夹爪，顺序与 joints 表一致 -->
    <position name="l_a1" joint="l_j1" kp="260" kv="26"/>
    <position name="l_a2" joint="l_j2" kp="260" kv="26"/>
    <position name="l_a3" joint="l_j3" kp="260" kv="26"/>
    <position name="l_a4" joint="l_j4" kp="260" kv="26"/>
    <position name="l_a5" joint="l_j5" kp="260" kv="26"/>
    <position name="l_a6" joint="l_j6" kp="260" kv="26"/>
    <position name="l_ag" joint="l_grip" kp="120" kv="6"/>

    <position name="r_a1" joint="r_j1" kp="260" kv="26"/>
    <position name="r_a2" joint="r_j2" kp="260" kv="26"/>
    <position name="r_a3" joint="r_j3" kp="260" kv="26"/>
    <position name="r_a4" joint="r_j4" kp="260" kv="26"/>
    <position name="r_a5" joint="r_j5" kp="260" kv="26"/>
    <position name="r_a6" joint="r_j6" kp="260" kv="26"/>
    <position name="r_ag" joint="r_grip" kp="120" kv="6"/>
  </actuator>
</mujoco>
```

配上这份 MJCF 的 profile（数字来自 `tools/list_model.py` 的输出）：

```python
from robot_profile import ArmSpec, GripperSpec, RobotProfile, TeleopSpec

# 单自由度夹爪：ctrl 直接是米（0 = 全开，0.035 = 合拢，留 5mm 别顶限位）
GRIP = GripperSpec(ctrl_index=6, ctrl_open=0.0, ctrl_close=0.035,
                   boost_ctrl_index=None,
                   qpos_index=6, qpos_at_open=0.0, qpos_at_close=0.035)

HOME = (0.0, 0.85, -1.55, 0.0, 0.70, 0.0)

PROFILE = RobotProfile(
    name="my_dual_arm",
    model_relpath="assets/my_dual_arm.xml",
    arms=[
        ArmSpec(name="left", drive_hand="left",
                joint_offset=0, actuator_offset=0,        # 左臂：从第 0 个开始
                n_arm_actuators=6, n_arm_qpos=6, n_gripper_qpos=1,
                ee_site="l_tcp", tcp_offset=0.0, gripper_body="l_link3",
                home_qpos=HOME, gripper=GRIP),
        ArmSpec(name="right", drive_hand="right",
                joint_offset=7, actuator_offset=7,        # 右臂：左臂占了 7 个
                n_arm_actuators=6, n_arm_qpos=6, n_gripper_qpos=1,
                ee_site="r_tcp", tcp_offset=0.0, gripper_body="r_link3",
                home_qpos=HOME, gripper=GRIP),
    ],
    cameras=[],                # 纯遥操作可以先不开相机
    teleop=TeleopSpec(follow_orientation=False),   # 6 轴腕部受限，只跟位置
    object_bodies=(),
)
```

三个要点：

1. **`joint_offset` / `actuator_offset` 是"第几个"**，不是 str。左臂从头开始是 0，
   右臂 = 左臂的关节数（含夹爪关节，上面是 6+1=7）。
2. **`n_gripper_qpos` 是夹爪占几个 qpos**。单自由度 slide = 1；Robotiq 2F-85
   那种四连杆是 8。没有夹爪写 0。
3. **`ee_site` 必须是 `<site>`**，不是 body 名。IK 是围绕 site 解的。

改完自查一遍：

```bash
VR_TELEOP_PROFILE=robot_profile_mydual python tools/check_profile.py
VR_TELEOP_PROFILE=robot_profile_mydual python tools/validate_dual.py
```

两项全绿，再 `./run.sh` 连头显。

> 上面这份 XML 只是**语法骨架**，关节长度/限位/kp 都是占位值。真实的臂请用厂商
> MJCF（Kinova、UR、Franka 等都有官方 MuJoCo 模型），把它 include 进来、名字加前缀，
> 不要照着这份手写运动学。

## 10. 这个版本没有的东西

| | 状态 | 说明 |
|---|---|---|
| 双臂数据采集 | ❌ | 训练格式要先定，见第 7 节 |
| 双臂物块挂载（attach） | ❌ | 只有纯物理接触 |
| 双臂运行日志 | ❌ | `LOG_COLUMNS` 按单臂 7 关节写死 |
| 双臂自碰撞检查 | ❌ | home 位姿要自己摆到互不干涉 |
| 左右手不同灵敏度 | ❌ | 共用一套 `TeleopSpec` |
| 双臂示例模型 | ❌ | **有意不提供**，见开头 |
