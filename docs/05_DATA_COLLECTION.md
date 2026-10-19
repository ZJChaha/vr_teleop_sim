# 05 · 数据采集

## 1. 一句话

**用 VR 手柄操作机械臂，按 50Hz 把 `(关节角, 夹爪, 图像, 语言指令)` 记成
LeRobot v2.1 数据集，直接能喂 π0 / OpenPI 之类的 VLA 做微调。**

> ⚠ 本页讲的是**单臂**采集。双臂配置（`arms=[...]`）下采集会被自动关闭：
> LeRobot 的特征是按"一台臂 + 一个夹爪"定的，两条臂要先决定拼成 2N+2 维还是
> 拆成两组，属于训练格式设计。见 [docs/08_DUAL_ARM.md](08_DUAL_ARM.md) 第 7 节。

---

## 2. 操作流程（采集模式）

```
./run.sh
  ├─ "要采集数据吗? [y/N]" → y
  └─ "本次任务描述: " → 把红色方块放到黄色方块上      ← 写进 tasks.jsonl，就是训练用的语言指令

┌────────────────────────────────────────────────────────────┐
│ ① 右手柄 侧键（按住）  → 进入跟随                            │
│ ② 右手柄 侧键（再按一次/重新按住）→ ★ 开始录这一集           │
│ ③ 操作：靠近 → 扣扳机夹住 → 抬起 → 移动 → 松扳机放下         │
│ ④ 左手柄 扳机          → ★ 切片，本段结束                    │
│ ⑤ 左手柄 X = 成功保存   |   Y = 失败，整段丢弃                │
│ ⑥ 回到 ①，开始下一集（编号自动 +1，失败集不占号）            │
└────────────────────────────────────────────────────────────┘
```

* **右手柄侧键的"上升沿"= 开集**。所以按住不放不会重复开集，必须松开再按。
* 单段上限 `DATASET_MAX_EP_S=300s`（忘了切片时会自动切）。
* 短于 `DATASET_MIN_EP_S=0.2s` 的段当误触丢弃。

> 采集线程是**独立的 50Hz 线程**，和 500Hz 控制循环解耦：
> 渲染慢了只会让数采掉帧（会打印警告），不会影响你操作。

---

## 3. 数据格式（LeRobot v2.1）

```
data/lerobot/
├── meta/
│   ├── info.json          # 特征定义 + 总集数/帧数 + 路径模板
│   ├── episodes.jsonl     # 每集的长度/任务/成功标记
│   ├── episodes_stats.jsonl
│   ├── tasks.jsonl        # 语言指令（同一个 task_index 可被多集复用）
│   └── stats.json
├── data/chunk-000/episode_000000.parquet
└── videos/chunk-000/
    ├── observation.images.scene/episode_000000.mp4
    ├── observation.images.wrist/episode_000000.mp4
    └── observation.images.top/episode_000000.mp4
```

### 特征：维度取决于你的关节数

`observation.state` / `action` 的维度 = **`n_arm_qpos + 1`**（关节 + 1 个夹爪）。
下面以参考实现（7 轴）为例：

```jsonc
"observation.state": float32[8]   // [joint_1..joint_7, gripper]
"action":            float32[8]   // 同上（下发值）
"timestamp":         float32[1]
"frame_index" / "episode_index" / "index" / "task_index": int64[1]
"success":           bool[1]
"observation.images.*": video, 480×640×3, avc1 / yuv420p / 50fps
```

换了机械臂，维度会跟着变（由 `robot_profile.py` 的 `n_arm_qpos` 决定）：

| 机器人 | state / action 维度 | 列名 |
|---|---|---|
| 7 轴 + 夹爪 | **8** | `joint_1..joint_7, gripper` |
| 6 轴 + 夹爪 | **7** | `joint_1..joint_6, gripper` |
| 5 轴 + 夹爪 | **6** | `joint_1..joint_5, gripper` |

`meta/info.json` 里的 `robot_type` 记录的是 `PROFILE.name`，所以不同机器人的数据集
**连身份都不同**。

**夹爪口径**：`0 = 全开`，`1 = 全闭`。

* `observation.state[-1]` = `qpos[qpos_index]` 归一化（读回的**实际**开度）
* `action[-1]` = 下发的命令归一化（可能领先实际值，这是正常的）

> 这两个不是同一个东西，别混。参考实现里 `ctrl=255 → state≈0.57` 而不是 1.0，
> 因为还有指垫厚度/物体占位。

### 相机录哪几路

每一路相机在 `robot_profile.py` 里都有 `record` 开关。**视频数据集建议两路就够**
（`腕部` + `侧面全局` 或 `俯视全局` 二选一），选法见 **[docs/04 §3](04_CAMERAS.md)**。

```python
CameraSpec("全局相机",     "cam_scene", "observation.images.scene", preview=True, record=False),
CameraSpec("腕部相机",     "wrist",     "observation.images.wrist", preview=True, record=True),
CameraSpec("全局俯视相机", "cam_top",   "observation.images.top",   preview=True, record=True),
```

改 `record` 会改变数据集的相机路数 ⇒ 特征集变了，**不能往旧目录续写**（见下一节）。

### 视频编码

```
libx264, -preset veryfast -crf 18, yuv420p
```

* 体积：640×480×3 路 × 50fps ≈ **每集每秒 ~1.5MB**（30 秒的集约 45MB）
* 想看画质是否够：`ffprobe` 单帧导出对比原渲染

### 一致性约束

`meta/info.json` 里的特征集**一旦确定就不能改**。`LeRobotWriter._check_compatible()`
会在续写时逐项比对 `fps / robot_type / features 的 shape 和 dtype`，不一致直接拒绝。

**下列任一变化都等于"新数据集"**，必须换目录（`VR_TELEOP_DS_ROOT=...`）：

```
关节数（5 轴 ↔ 6 轴 ↔ 7 轴）
夹爪 qpos 数
相机路数 / 相机分辨率
fps
robot_type（= PROFILE.name）
```

### 不同机器人的数据不能混

这条比上面更硬：**不同轴数的数据集在训练侧是不兼容的**。

* 6 轴数据是 7 维，7 轴数据是 8 维 —— 拿去喂同一个 checkpoint 会直接报形状错误
* 即使轴数相同，关节范围/零位不同，`stats.json` 的归一化也不能复用
* 想混训必须先统一动作空间（重映射到末端位姿、或补齐缺失维度），那是另一个话题

实践建议：**一台机器人一个目录、一个数据集、一套 stats**。目录名带上机器人名：

```bash
VR_TELEOP_PROFILE=robot_profile_myarm VR_TELEOP_DS_ROOT=data/lerobot_myarm ./run.sh
```

---

## 4. 失败集：真删，不留痕

按 Y（失败）时：

```
关掉视频管道 → 删掉半截 mp4 → 不写 parquet → 不写元数据 → 不消耗 episode 编号
```

也就是说失败集**完全不存在**，不是"标记为失败但保留"。

* 想改成保留：`VR_TELEOP_D2_DROP_FAILED=0`
* 想事后清理：直接看 `meta/episodes.jsonl` 的 `success` 字段，删对应编号的
  parquet + mp4，再重写索引（本仓库没带 prune 工具，因为各人策略不同）

---

## 5. 每集该多长？

经验值（参考实现，桌面抓取 + 堆叠）：

| 任务 | 单集时长 | 备注 |
|---|---|---|
| 抓一个方块放到指定位置 | 4~8 s | 最干净，适合起步 |
| 两个方块堆叠 | 8~15 s | 需要更多"对准"动作 |
| 三个方块依次搬运 | 20~40 s | 长序列，容易出累积误差 |

> 单集越长，VLA 越难学（误差累积）；建议**先做短任务**，把成功率做上去。

---

## 6. 采集多少集够用

不是这个仓库能替你回答的问题（取决于任务/模型/是否 LoRA），但两条经验：

1. **先做 50 集小规模**，跑通"导出 → 训练 → 在仿真里回放"整条链路，再扩量。
   否则采了 500 集才发现语言指令写错、相机装反，返工成本很高。
2. **成功率比集数重要**。同一任务里混着"夹空/掉件"的数据，模型会学到错误动作。
   宁可 100 集全成功，不要 300 集里 1/3 是失败的。

---

## 7. 用 lerobot 库加载（自检）

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("data/lerobot")          # 或 repo_id
print(ds.features)                            # 检查相机路数/维度
item = ds[0]
print(item["observation.state"].shape,      # 7 轴 → torch.Size([8])；6 轴 → [7]
      item["action"].shape,
      item["observation.images.scene"].shape)  # [3, 480, 640]
```

维度**不要写死**，永远从数据集里读：

```python
import json
J = json.load(open("data/lerobot/meta/info.json"))
d = J["features"]["observation.state"]["shape"][0]     # 8 / 7 / 6 …
print(J["robot_type"], d, J["features"]["observation.state"]["names"])
```

不想装 lerobot 也能自检（只依赖 pyarrow）：

```python
import pyarrow.parquet as pq, json
t = pq.read_table("data/lerobot/data/chunk-000/episode_000000.parquet")
print(t.column_names, t.num_rows)
print(json.load(open("data/lerobot/meta/info.json"))["features"].keys())
```

---

## 8. 时间轴对齐

三类数据的频率不同，必须**同帧写入**：

```
控制循环 500Hz ──┐
                 ├─ 数采线程取"最新快照"（不是队列！）→ 每 20ms 落一帧
渲染      ~55Hz ─┘
```

* 用**快照**而不是队列：宁可丢中间帧，也不要在采集线程里排队积压（会越来越慢）。
* 快照里同时带 `qpos / ctrl / 夹爪`，所以 state 和 action 天然同帧。
* 掉帧时会有 `慢拍` 计数 + 警告，说明这一段的时间轴被压缩了 —— **这种集建议丢弃**。

---

## 9. 上传到训练服务器

```bash
# 只传数据集目录（不要传代码/日志/模型）
rsync -av --progress data/lerobot/ user@server:/data/datasets/lerobot_$(date +%Y%m%d)/
```

或者先打包：

```bash
tar -czf lerobot_$(date +%Y%m%d).tar.gz data/lerobot
sha256sum lerobot_*.tar.gz > lerobot_*.tar.gz.sha256   # 传完在服务器上核验
```
