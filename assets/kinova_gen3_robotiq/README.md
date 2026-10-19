# Kinova Gen3 (7-DOF) + Robotiq 2F-85 组合模型

机械臂：Kinova Gen3 7 自由度；夹爪：Robotiq 2F-85（对应
ros-industrial-attic/robotiq 的 robotiq_2f_85_gripper_visualization 包）。
两个模型使用 Kinova 官方 kortex_description 的安装方式组合：夹爪底座挂在
机械臂腕部 `end_effector_link` 上，固定关节 `gripper_base_joint`
origin `xyz="0 0 0" rpy="0 0 1.57"`。

## 文件

| 文件 | 说明 |
|------|------|
| `gen3_robotiq_2f85.xml` | MuJoCo 原生模型（MJCF）。**在 MuJoCo 里用这个** |
| `gen3_robotiq_2f85.urdf` | 标准 URDF（ROS / Gazebo / MoveIt 用） |
| `scene.xml` | 组合 MJCF 的场景封装（加地面、灯光、天空盒） |
| `view.py` | 显式 viewer 演示脚本 |
| `assets/` | MJCF 的网格（17 个 STL，机械臂 9 + 夹爪 8） |
| `meshes/` | URDF 的网格（17 个 STL，机械臂 8 + 夹爪 9） |

## 模型参数

- 机械臂：9 刚体，7 关节（joint_1 ~ joint_7），7 个 position 执行器
- 夹爪：8 关节（左右各 4 连杆），1 个 general 执行器 `fingers_actuator`
- MJCF 合计：nbody=23，njnt=15，nq=15，nu=8
- 机械臂 ctrl = 目标关节角（弧度）；夹爪 ctrl 范围 0~255（0=闭合，255=全开）

## 运行（MuJoCo viewer）

```bash
python3 view.py
```

view.py 会让机械臂在 home 位姿附近摆动，夹爪缓慢开合。

## 用 MJCF 写自己的控制代码

```python
import mujoco, numpy as np

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)

data.ctrl[:7] = [0, 0.26, 3.14, -2.27, 0, 0.96, 1.57]  # 机械臂关节角
data.ctrl[7] = 255                                     # 夹爪全开 (0~255)
mujoco.mj_step(model, data)
```

## 两种夹爪建模方式的区别

- **MJCF（2f85 机构）**：来自 MuJoCo Menagerie，完整建模了 4 杆连杆机构
  （driver/coupler/follower/spring_link）+ tendon + equality 约束，物理更精确。
- **URDF（mimic 关节）**：采用 ROS 社区标准做法，`finger_joint` 为主动关节，
  其余手指关节用 `<mimic>` 跟随。这是 ROS/Gazebo/MoveIt 的通用表示。

两者描述的是同一台物理夹爪，只是机构建模精度不同。

## 网格来源与单位

- 机械臂 STL：Kinova 官方 kortex_description（米，无缩放）
- 夹爪：Robotiq 官方 DAE（毫米）已转为 STL，URDF 中 `scale="0.001"`；
  底座 `robotiq_arg2f_base_link.stl` 为米（无缩放）
- MJCF 的夹爪 STL 来自 MuJoCo Menagerie（毫米，MJCF 内 `scale=0.001`）

## 关键帧

- `home`：机械臂标准 home 位姿，夹爪闭合
- `retract`：机械臂收拢位姿，夹爪闭合
