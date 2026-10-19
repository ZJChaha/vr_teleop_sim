# 07 · Agent 交接文档

> 给"下一个接手这个仓库的 AI agent"。目标：**读这一篇就能安全地改东西、验证、交付**，
> 不需要重新摸一遍。

---

## 1. 项目性质

一个 **VR 遥操作 + 数据采集** 栈：Meta Quest（WebXR）→ WebSocket → Python →
DLS IK → MuJoCo 仿真机械臂；同时按 50Hz 录 LeRobot v2.1 数据集。
**只做仿真**，不带真机驱动。参考机器人：Kinova Gen3 + Robotiq 2F-85。

---

## 2. 仓库地图

| 你想改 | 改这里 | 备注 |
|---|---|---|
| 换机器人 / 关节数 / 夹爪刻度 / home / 相机 | `robot_profile.py` | **唯一**该改的配置口 |
| 控制循环 / IK / 抓取模式 / 手柄映射 | `server.py` | 顶部是常量区，都有"为什么"注释 |
| Quest 端页面 / 手柄采集 | `index.html` | WebXR，无构建步骤 |
| 相机预览 | `camview.py` | 独立线程；`apply_abs_cams` / `load_model_with_cams` / `apply_render_flags` |
| 数据集格式 / 落盘 | `dataset_recorder.py` | LeRobot v2.1 writer |
| 场景（桌子/方块/相机 XML） | `assets/scene_grasp.xml` | |
| 证书 | `scripts/make_cert.sh` | SAN 覆盖本机所有 IP |
| 自检 | `tools/list_model.py`、`tools/check_profile.py`、`tools/probe_model.py`、`tools/validate_e2e.py`、`tools/validate_dual.py` | 只读 |
| 双臂 | `robot_profile.py` 的 `ArmSpec`（用户自己的 profile 写 `arms=[...]`） | 见 docs/08；仓库不带双臂模型 |

---

## 3. 常用命令

```bash
# 环境
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 静态自检（不启 GL）——每次改完必跑
.venv/bin/python tools/check_profile.py

# 离线端到端（IK 稳定性 + 物理抓取搬运，无 viewer）
.venv/bin/python tools/validate_e2e.py

# 起服务（会问采集/任务描述）
./run.sh
VR_TELEOP_DS=0 ./run.sh                  # 不采集
VR_TELEOP_PORT=8445 VR_TELEOP_DS=0 ./run.sh   # 换端口，便于并行测试

# 数据集体检（不依赖 lerobot 库）
.venv/bin/python - <<'PY'
import json, pyarrow.parquet as pq
J = json.load(open("data/lerobot/meta/info.json"))
print(J["total_episodes"], J["total_frames"], list(J["features"]))
t = pq.read_table("data/lerobot/data/chunk-000/episode_000000.parquet")
print(t.column_names, t.num_rows)
PY
```

---

## 4. 不变量

| # | 不变量 | 破坏的后果 |
|---|---|---|
| 1 | `observation.state` / `action` 是 **8 维**（N 关节 + 1 夹爪），夹爪 `[0,1]` | 数据集与已有模型/加载器不兼容 |
| 2 | 仿真 **500Hz**、数采 **50Hz**、viewer 同步**每 5 步** | 变频率会让手感/数据时间轴变形 |
| 3 | **不要**给 IK 目标加单帧限幅 / 位移球 / 姿态墙 | 实测跟踪误差 0.01mm → 51mm，且"一顿一顿" |
| 4 | **不要**用墙钟节流去"降 viewer 频率" | 实测掉到 31Hz 并产生相位漂移 |
| 5 | IK 必须**热启动**（`q_seed=上一帧解`） | 远距离会掉进局部极小，表现为"够不到" |
| 6 | 工具点 = `site - TCP_OFF*z`，不是 site 本身 | 夹爪会停在物体上方抓空 |
| 7 | 相机只改**内存**（`extra_cameras` / `camera_overrides`），不改 XML | 与共享模型冲突、无法回滚 |
| 8 | `Renderer` 必须与 `data` 属于**同一个 model** | 相机覆盖不生效，产出假数据 |
| 9 | 数据集特征集一旦确定**不能改**（要改就换目录） | `_check_compatible` 拒绝续写 / 写坏旧集 |
| 10 | 端口要**先绑再起线程** | 端口冲突时清理逻辑会删掉别的实例刚采的集 |
| 11 | 退出时主动关 WS + `shutdown_timeout=1` | 会白等 ~122 秒 |
| 12 | 失败集按 Y **整集真删**，不占编号 | 会污染训练集（参考实现的既定策略） |
| 13 | 双臂时每条臂的 `joint_offset` / `actuator_offset` 必须唯一且不重叠 | 两条臂抢同一个关节，表现为"动一只手两边都动" |
| 14 | 双臂**不做数采**（当前版本） | 硬开会让特征维度与旧集冲突；训练格式未定 |

---

## 5. 调试顺序

```
① 模型/配置层   .venv/bin/python tools/check_profile.py
①' 双臂        .venv/bin/python tools/list_model.py <你的.xml> ; tools/validate_dual.py
② 运动学       .venv/bin/python tools/validate_e2e.py
③ 服务层       ./run.sh 看终端：模型路径 / 相机 / 端口 / 有无 Traceback
④ 网络层       ss -ltn | grep 8444 ; curl -k https://127.0.0.1:8444
⑤ 手柄层       看 runlog 的 hand_age_ms（>50ms 人就能感觉到卡）
⑥ 视觉层       看相机预览窗口 dump（VR_TELEOP_CAMDUMP=/tmp/x）
```

每层都有可观测信号，**不要跳过①直接改代码**。

---

## 6. 交付前的验证门

改完任何东西，至少跑：

```bash
.venv/bin/python tools/check_profile.py          # 必须全 ✅
.venv/bin/python tools/validate_e2e.py           # 抓取通过数不应低于改动前
VR_TELEOP_DS=0 VR_TELEOP_PORT=8445 timeout 20 ./run.sh   # 能起来 + 无 Traceback
```

如果改的是**双臂**相关，再加两条：

```bash
VR_TELEOP_PROFILE=<用户的双臂 profile> .venv/bin/python tools/check_profile.py
VR_TELEOP_PROFILE=<用户的双臂 profile> .venv/bin/python tools/validate_dual.py   # 必须全 ✅
```

如果改了**采集**相关：

```bash
VR_TELEOP_DS=1 VR_TELEOP_DS_TASK=测试 VR_TELEOP_DS_ROOT=/tmp/ds_test ./run.sh
# 用手柄采一段，确认 info.json 有 N 个 observation.images.*、parquet 帧数与视频一致
```

**不允许**用"看起来对"代替上述输出。报告里必须贴真实命令输出。

---

## 7. 危险操作

```
✗ 删除/覆盖 data/ 下的数据集（尤其别人采的）
✗ 用 --force 推送到共享分支
✗ 改 assets/ 里的共享模型（先复制一份）
✗ 把"相机参数"写回 XML（应走 camera_overrides）
✗ 在采集运行时重启服务（会截断正在写的集）
✗ 装 GPU 驱动 / 改 GL 后端（可能让 MuJoCo 起不来）
```

安全操作（可以自由做）：跑 `check_profile.py`、`validate_e2e.py`、读日志、
在**新目录**里做实验、加新 profile 文件、加新文档。

---

## 8. 已知未解问题

| 问题 | 现状 | 线索 |
|---|---|---|
| 竖直朝下抓取"夹得住抬不起" | 抬升 Δz 只有 10~90mm（正常 160+） | 疑似指垫接触点少 + 腕部近限位；换斜 45° home 可规避 |
| 放置时"落上去保持不住" | 松爪后物块偶发滑动/跳动 | 与接触刚度、松爪速度（`GRIP_RATE`）相关 |
| 多机位自动化选点 | 目前是"投影 + 像素数 + 遮挡"半自动 | `docs/04` 第 4.3 节有流程，可做成工具 |

---

## 9. 参考环境

```
OS      Ubuntu 20.04/22.04, X11
Python  3.11 + MuJoCo 3.14 + numpy 2.x + aiohttp 3.x
GPU     AMD RX 5500 XT (Mesa radeonsi)，640x480 单路渲染 ~6.1ms（未裁通道）
吞吐    3 路相机 + 3 预览窗口：裁 shadow+reflection 后 50Hz
头显    Meta Quest 1（WebXR），控制器 20~72Hz 上报
```

---

## 10. 交接清单

```
[ ] 读 README → docs/02（架构）→ docs/03（换机器人）
[ ] 跑 check_profile.py，确认 ✅
[ ] 跑 validate_e2e.py，记下通过数（基线）
[ ] 起一次服务，浏览器打开 https://<IP>:8444（证书警告→继续）
[ ] 找到 data/lerobot（如果存在），用 pyarrow 读一集，确认 8 维 + N 路视频
[ ] 把基线数字（e2e 通过数 / 吞吐 / 版本）记下来，之后任何改动都对比它
```

---

## 11. 报告风格

* 只报**实测**结果，贴真实命令输出；没验证的写"未验证"。
* 先说"改了什么文件 / 为什么"，再说"怎么验证的"。
* 结构化的坑（踩过一次就再也不会踩的）写进 `docs/` 对应章节，不要只留在对话里。
* 不可逆操作先说明 + 等确认；系统不拦 ≠ 可以擅自做。
