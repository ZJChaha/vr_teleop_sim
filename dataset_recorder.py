"""LeRobot v2.1 数据集采集器 (π0 关节空间约定, 50Hz, 双相机)。

设计目标
--------
· 格式 = **LeRobot v2.1**(每个 episode 一个 parquet + 每相机一个 mp4), 这样
  openpi / LeRobot 的加载器能直接吃。选 v2.1 而不是 v3.0: v3.0 把多个 episode
  拼进共享文件、靠元数据切分, 对**边采边追加**的记录器太复杂, v2.1 一集一文件天然合适。
· 动作空间 = **π0 关节空间**: 官方 norm_stats.md 规定
      "7-DoF 机器人: 前 7 维 = 关节角(弧度), 第 8 维 = 夹爪位置 [0,1] (0=全开, 1=全闭)"
      "state 与 action 用同一套定义";  ARX/Trossen 就是 **50Hz**  → 与用户要求一致。
  · `observation.state` = 实际关节角 qpos[0:7] (rad) + 夹爪实际开度
  · `action`            = **下发的目标关节角** ctrl[0:7] (rad) + 夹爪指令
    (遥操作里 action 的物理含义就是"操作者通过手柄下发的目标", 不是下一帧状态)
  · 夹爪统一到 **0=全开 / 1=全闭**

交互(用户指定)
-------------
  启动前: 终端里**手动输入任务描述**(写进 tasks.jsonl, 也是训练时的语言指令)
  操作中: 用右手柄照常操作
  切片:   **左手柄扳机** → 结束当前这一段
  打标签: **左手柄 X = 成功 / Y = 失败**
  下一轮: 打完标签后, **右手柄侧键**再次按下 → 开始下一轮采集
  循环直到关闭程序

★ 三条性能铁律 (都是实测踩出来的)
------------------------------------------------
1. **视频边录边流式写 ffmpeg, 绝不把帧攒在内存里。**
   640x480 一帧 921KB, 两个相机 50Hz → 每秒 92MB。攒 300s 就是 27GB。
   而且攒到最后一次性编码, 那一瞬间的 CPU/GIL 会把 500Hz 控制循环直接饿死
   (实测: 步频从 500/s 掉到 **5.8/s**, 跟不上 351 帧)。
   现在: episode 开始就开 ffmpeg 管道, 每帧立刻写走, 结束时只需收尾。
2. **图像统计是"每通道 3 个数", 不是每像素。**
   LeRobot 的图像 stats 形状是 (3,)(归一化后 per-channel mean/std/min/max)。
   我一开始按像素存 → JSON 里塞了 480*640*3*4 个数, 每次 finalize 都写一遍。
3. **锁只在"取快照"那一瞬间持有**, 渲染/编码/写盘时绝不持锁, 否则会卡住控制循环。

线程模型(与 camview/runlog 同一套纪律)
------------------------------------
· 控制循环(500Hz)只调 `publish()` / `handle_hands()` —— 只做拷贝和边沿判断, 微秒级。
· 采集线程自己 50Hz: 快照 → 渲染两路相机 → 立刻写进 mp4 管道。
· 整个线程 try/except 包住: 采集出任何问题只是自己停下, 绝不弄死遥操作。
"""
from __future__ import annotations

import collections
import json
import os
import subprocess
import threading
import time
import traceback

import numpy as np

# ---- π0 关节空间: 关节数 + 1(夹爪) ----
# ★ 开源版: 维度/名字/夹爪刻度都来自 robot_profile.py，这样换机器人不用改本文件
from robot_profile import load_profile                    # noqa: E402

PROFILE = load_profile()
DIM_STATE = PROFILE.state_dim
GRIP_QMAX = PROFILE.gripper.qpos_at_close - PROFILE.gripper.qpos_at_open
GRIP_CMD_MAX = PROFILE.gripper.ctrl_close - PROFILE.gripper.ctrl_open

# ★ 失败集不入库（用户裁决 2026-10-14）：按 Y(失败) ⇒ **这一集彻底丢弃** ——
#   关掉视频管道并删掉半截 mp4，不写 parquet / 不写任何元数据，**不消耗 episode 编号**
#   （与既有"太短丢弃"路径完全同构：都不调用 sink.finish() ⇒ _commit 未执行）。
#   回滚（想照旧保留失败集）：VR_TELEOP_D2_DROP_FAILED=0
DROP_FAILED = os.environ.get("VR_TELEOP_D2_DROP_FAILED", "1").lower() not in (
    "0", "false", "no", "off")

FEATURE_NAMES = PROFILE.joint_names
N_CH = 3                 # 图像统计按通道


# ============================================================================
# 标量统计 (关节/动作等): 增量 min/max/mean/std, std 用总体标准差(ddof=0)
# ============================================================================
class _Stats:
    """Welford 式累积 (count/mean/M2), 这样**跨 episode / 跨 session 池化**时
    能用标准公式合并, 不必把数据重读一遍。"""

    def __init__(self, shape):
        self.shape = tuple(shape)
        n = int(np.prod(self.shape))
        self.count = 0
        self.mean = np.zeros(n, dtype=np.float64)
        self.m2 = np.zeros(n, dtype=np.float64)
        self.vmin = np.full(n, np.inf)
        self.vmax = np.full(n, -np.inf)

    def update(self, arr):
        a = np.asarray(arr, dtype=np.float64).reshape(-1, int(np.prod(self.shape)))
        n = a.shape[0]
        if n == 0:
            return
        bm = a.mean(axis=0)
        b2 = ((a - bm) ** 2).sum(axis=0)
        if self.count == 0:
            self.mean, self.m2 = bm, b2
        else:
            d = bm - self.mean
            tot = self.count + n
            self.mean = self.mean + d * (n / tot)
            self.m2 = self.m2 + b2 + d ** 2 * (self.count * n / tot)
        self.count += n
        self.vmin = np.minimum(self.vmin, a.min(axis=0))
        self.vmax = np.maximum(self.vmax, a.max(axis=0))

    def merge(self, other):
        if other is None or other.count == 0:
            return self
        if self.count == 0:
            self.count, self.mean, self.m2 = other.count, other.mean.copy(), other.m2.copy()
            self.vmin, self.vmax = other.vmin.copy(), other.vmax.copy()
            return self
        n1, n2 = self.count, other.count
        tot = n1 + n2
        d = other.mean - self.mean
        self.mean = self.mean + d * (n2 / tot)
        self.m2 = self.m2 + other.m2 + d ** 2 * (n1 * n2 / tot)
        self.count = tot
        self.vmin = np.minimum(self.vmin, other.vmin)
        self.vmax = np.maximum(self.vmax, other.vmax)
        return self

    def to_json(self):
        std = np.sqrt(self.m2 / self.count) if self.count > 0 else np.zeros_like(self.m2)
        out = {"count": [int(self.count)] if not self.shape else
               np.full(self.shape, int(self.count)).tolist()}
        for k, v in (("min", self.vmin), ("max", self.vmax),
                     ("mean", self.mean), ("std", std)):
            out[k] = v.reshape(self.shape).tolist() if self.shape else float(v)
        return out

    @classmethod
    def from_json(cls, d, shape):
        s = cls(shape)
        if not d:
            return s
        cnt = d.get("count")
        s.count = int(cnt[0]) if isinstance(cnt, (list, tuple)) else int(cnt or 0)
        for key, attr in (("min", "vmin"), ("max", "vmax"), ("mean", "mean")):
            v = np.asarray(d.get(key, 0.0), dtype=np.float64).reshape(-1)
            if v.size == int(np.prod(shape)):
                setattr(s, attr, v)
        std = np.asarray(d.get("std", 0.0), dtype=np.float64).reshape(-1)
        s.m2 = (std ** 2) * s.count if s.count > 0 else np.zeros_like(s.m2)
        return s


DS = 2                   # 图像统计的降采样倍数 (取像素的 1/DS²)


class _ChannelStats:
    """图像的**每通道**统计 (形状 (3,), 归一化到 [0,1])。

    ★ 为什么降采样: 现场实测 (480x640x3, 真服务里跑, 有 500Hz 控制循环抢 GIL/内存带宽)
        reshape(-1,3).astype(float64) + f.T@f   25.0 ms/帧   ← 离线只 3.9ms, 现场崩了
        降采样 2x + 整数累加(不分配大数组)        ~1-2 ms/帧
      每帧 astype(float64) 要新分配 22MB, 还要让 BLAS 开线程, 和主循环对着抢。
      降采样 2x 只取 1/4 像素, 实测标准差只差 0.2% —— 归一化统计用这个精度绰绰有余。
      (LeRobot 自己的 stats 是全像素算的, 这里明确是估计值, 已记进 README。)
    """

    def __init__(self):
        self.n = 0
        self.s = np.zeros(N_CH, dtype=np.int64)
        self.ss = np.zeros(N_CH, dtype=np.int64)
        self.mn = np.full(N_CH, 255, dtype=np.int64)
        self.mx = np.zeros(N_CH, dtype=np.int64)
        self.m2 = np.zeros(N_CH, dtype=np.float64)
        self.mean = np.zeros(N_CH, dtype=np.float64)

    def add(self, frame_u8):
        """frame_u8: (H,W,3) uint8 —— 必须**在渲染缓冲被下一次 render 覆盖之前**调用。

        统计口径: 对**降采样样本**算均值/方差, 但按**全图像素数**归一化与加权 ——
        这样 `count` 与 LeRobot 原版语义一致 (全像素数), 加载器做加权合并时权重才对;
        均值/方差本身是估计值 (实测标准差与全像素差 <0.3%)。
        """
        g = frame_u8[::DS, ::DS] if DS > 1 else frame_u8
        if not g.flags['C_CONTIGUOUS']:
            # [::2,::2] 是跨步视图, einsum 在这种访问模式下慢得离谱(实测 7.6ms),
            # 先拷成连续内存 (只 0.23MB) 再算
            g = np.ascontiguousarray(g)
        n_s = int(g.shape[0]) * int(g.shape[1])                   # 样本像素数
        n_full = int(frame_u8.shape[0]) * int(frame_u8.shape[1])  # 实际像素数(权重)
        scale = n_full / n_s
        s = g.sum(axis=(0, 1), dtype=np.int64)
        ss = np.einsum('hwc,hwc->c', g, g, dtype=np.int64)
        bm = s / n_s
        # b2 按全像素尺度还原, 这样 m2/count 就是"全图"的方差(估计)
        b2 = (ss.astype(np.float64) - (s.astype(np.float64) ** 2) / n_s) * scale
        n = n_full
        if self.n == 0:
            self.mean, self.m2 = bm, b2
        else:
            d = bm - self.mean
            tot = self.n + n
            self.mean = self.mean + d * (n / tot)
            self.m2 = self.m2 + b2 + d ** 2 * (self.n * n / tot)
        self.n += n
        self.mn = np.minimum(self.mn, g.reshape(-1, N_CH).min(axis=0))
        self.mx = np.maximum(self.mx, g.reshape(-1, N_CH).max(axis=0))

    def to_json(self):
        if self.n == 0:
            z = [0.0] * N_CH
            return {"min": list(z), "max": list(z), "mean": list(z), "std": list(z),
                    "count": [0]}
        var = np.maximum(self.m2 / self.n, 0.0)
        return {"min": (self.mn / 255.0).tolist(), "max": (self.mx / 255.0).tolist(),
                "mean": (self.mean / 255.0).tolist(),
                "std": (np.sqrt(var) / 255.0).tolist(), "count": [int(self.n)]}

    @classmethod
    def from_json(cls, d):
        c = cls()
        if not d:
            return c
        n = int(np.asarray(d.get("count", [0])).reshape(-1)[0])
        c.n = n
        if n > 0:
            c.mean = np.asarray(d["mean"], dtype=np.float64) * 255.0
            var = (np.asarray(d["std"], dtype=np.float64) * 255.0) ** 2
            c.m2 = var * n
            c.mn = (np.asarray(d["min"], dtype=np.float64) * 255.0)
            c.mx = (np.asarray(d["max"], dtype=np.float64) * 255.0)
        return c


# ============================================================================
# 单路相机的 mp4 写入 (管道喂 ffmpeg, 系统自带 libx264, 不需要 av/torchvision)
# ============================================================================
class _VideoWriter:
    def __init__(self, path, w, h, fps, crf=18, preset="veryfast"):
        self.path = path
        self.w, self.h, self.fps = int(w), int(h), float(fps)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.proc = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{self.w}x{self.h}", "-r", f"{self.fps}", "-i", "-",
             "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
             "-pix_fmt", "yuv420p", path],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def write(self, frame_rgb):
        if self.proc is None or self.proc.stdin is None:
            return
        a = frame_rgb if frame_rgb.flags['C_CONTIGUOUS'] else np.ascontiguousarray(frame_rgb)
        self.proc.stdin.write(a.tobytes())

    def close(self, delete=False):
        if self.proc is None:
            return
        p = self.proc
        self.proc = None
        try:
            if p.stdin:
                p.stdin.close()
            p.wait(timeout=60)
        except Exception:                                        # noqa: BLE001
            try:
                p.kill()
            except Exception:                                    # noqa: BLE001
                pass
        if delete:
            try:
                os.remove(self.path)
            except OSError:
                pass


# ============================================================================
# 一个 episode 的流式写入 (边录边写, 不在内存里攒帧)
# ============================================================================
class EpisodeSink:
    def __init__(self, writer, task, ep, task_index):
        self.w = writer
        self.task = task
        self.ep = ep
        self.task_index = task_index
        self.n = 0
        self.closed = False
        H, W = writer.image_size
        self.vw = {k: _VideoWriter(writer.video_path(ep, k), W, H, writer.fps)
                   for k in writer.camera_keys}
        self._st = []
        self._ac = []
        self._img_stats = {k: _ChannelStats() for k in writer.camera_keys}

    def new_frame(self):
        self._pending = True

    def add_camera(self, key, frame):
        """★ 必须在渲染缓冲被下一次 render 覆盖前调用: 先写走, 再算统计。"""
        self.vw[key].write(frame)               # 立刻写走, 不攒内存
        self._img_stats[key].add(frame)         # 同一份缓冲, 不额外拷贝

    def finish_frame(self, state8, action8):
        self._st.append(np.asarray(state8, dtype=np.float32))
        self._ac.append(np.asarray(action8, dtype=np.float32))
        self.n += 1

    def add_frame(self, state8, action8, frames):
        """便捷版 (测试用): state8/action8 为 (8,) float; frames 为 {cam_key: (H,W,3) uint8}。"""
        self.new_frame()
        for k in self.w.camera_keys:
            self.add_camera(k, frames[k])
        self.finish_frame(state8, action8)

    def abort(self):
        """丢弃这一段 (太短/出错): 关掉并删掉半截 mp4, 不落任何元数据。"""
        if self.closed:
            return
        self.closed = True
        for v in self.vw.values():
            v.close(delete=True)

    def finish(self, success, session_id=None):
        if self.closed:
            raise RuntimeError("sink 已关闭")
        self.closed = True
        for v in self.vw.values():
            v.close()
        return self.w._commit(self, success, session_id)


# ============================================================================
# LeRobot v2.1 写入器
# ============================================================================
class LeRobotWriter:
    """把 episode 追加写进一个 LeRobot v2.1 数据集 (支持跨 session 续写)。"""

    def __init__(self, root, fps=50.0, robot_type="kinova_gen3_robotiq_2f85",
                 camera_keys=("observation.images.scene", "observation.images.wrist"),
                 image_size=(480, 640), chunks_size=1000, task=None, verbose=True):
        self.root = os.path.abspath(root)
        self.fps = float(fps)
        self.robot_type = robot_type
        self.camera_keys = list(camera_keys)
        self.image_size = tuple(image_size)
        self.chunks_size = int(chunks_size)
        self.verbose = verbose
        self.task = task
        self.meta_dir = os.path.join(self.root, "meta")
        # RLock: _commit() 持锁时还会调 finalize() -> 非重入锁会**自死锁**
        self._lock = threading.RLock()

        self.features = self._build_features()
        self.episodes = []
        self.ep_stats = []
        self.tasks = []
        self.total_frames = 0
        self.global_stats = {}
        self.ok = False
        self._check_ffmpeg()
        self._resume_or_init()

    # ---------- 结构 ----------
    def _check_ffmpeg(self):
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=True, timeout=20)
        except Exception as e:                                   # noqa: BLE001
            raise RuntimeError(f"找不到可用的 ffmpeg (写 mp4 要用): {e}") from e

    def _build_features(self):
        H, W = self.image_size
        f = {
            "observation.state": {"dtype": "float32", "shape": [DIM_STATE],
                                  "names": FEATURE_NAMES},
            "action": {"dtype": "float32", "shape": [DIM_STATE], "names": FEATURE_NAMES},
            "success": {"dtype": "bool", "shape": [1], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }
        for k in self.camera_keys:
            f[k] = {"dtype": "video", "shape": [H, W, 3],
                    "names": ["height", "width", "channels"],
                    "info": {"video.fps": self.fps, "video.codec": "avc1",
                             "video.pix_fmt": "yuv420p",
                             "video.width": W, "video.height": H, "video.channels": 3}}
        return f

    def _scalar_feats(self):
        return [k for k, v in self.features.items() if v["dtype"] != "video"]

    def data_path(self, ep):
        return os.path.join(self.root, "data", f"chunk-{ep // self.chunks_size:03d}",
                            f"episode_{ep:06d}.parquet")

    def video_path(self, ep, key):
        return os.path.join(self.root, "videos", f"chunk-{ep // self.chunks_size:03d}",
                            key, f"episode_{ep:06d}.mp4")

    # ---------- 续写 / 初始化 ----------
    def _read_jsonl(self, name):
        p = os.path.join(self.meta_dir, name)
        out = []
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.append(json.loads(line))
        return out

    def _write_jsonl(self, name, rows):
        p = os.path.join(self.meta_dir, name)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, p)

    def _resume_or_init(self):
        info_p = os.path.join(self.meta_dir, "info.json")
        if os.path.exists(info_p):
            with open(info_p, "r", encoding="utf-8") as f:
                old = json.load(f)
            self._check_compatible(old)
            self.episodes = self._read_jsonl("episodes.jsonl")
            self.ep_stats = self._read_jsonl("episodes_stats.jsonl")
            self.tasks = self._read_jsonl("tasks.jsonl")
            self.total_frames = int(old.get("total_frames", 0))
            self.global_stats = self._pool_existing_stats()
            self._log(f"[数据集] 续写已有数据集: {self.root}\n"
                      f"         已有 {len(self.episodes)} 集 / {self.total_frames} 帧 / "
                      f"{len(self.tasks)} 个任务")
        else:
            os.makedirs(self.meta_dir, exist_ok=True)
            self._log(f"[数据集] 新建数据集: {self.root}")
        self._cleanup_orphans()
        self.ok = True

    def _cleanup_orphans(self):
        """清掉上次崩溃/强杀留下的半截文件。

        强杀时采集线程来不及收尾 (`EpisodeSink.abort()` 跑不到), 目录里会留下
        "有 mp4、没有 parquet、也不在 episodes.jsonl 里"的残片 —— 它们的 episode
        编号一定 >= 已记录的集数。下次采到同一编号时本来会被 `-y` 覆盖, 但若用户就此
        停采, 残片会一直留着让人困惑 (也可能被当成有效数据拷走)。

        所以启动续写时清掉, 并逐条打印 (本项目习惯: 任何清理都要可见)。
        """
        import re
        n = len(self.episodes)
        pat = re.compile(r"^episode_(\d{6})\.(parquet|mp4)$")
        bases = [os.path.join(self.root, "data")]
        vroot = os.path.join(self.root, "videos")
        if os.path.isdir(vroot):
            bases += [os.path.join(vroot, d) for d in sorted(os.listdir(vroot))
                      if os.path.isdir(os.path.join(vroot, d))]
        removed = []
        for base in bases:
            if not os.path.isdir(base):
                continue
            for dirpath, _dirnames, filenames in os.walk(base):
                for fn in filenames:
                    m = pat.match(fn)
                    if m and int(m.group(1)) >= n:
                        fp = os.path.join(dirpath, fn)
                        try:
                            os.remove(fp)
                            removed.append(os.path.relpath(fp, self.root))
                        except OSError:
                            pass
        if removed:
            self._log(f"[数据集] 清掉 {len(removed)} 个上次未收尾的残片 "
                      f"(不属于数据集, 已被记录的集数={n}): "
                      f"{', '.join(sorted(removed))}")
        return removed

    def _check_compatible(self, old):
        """续写前校验配置一致 —— 不一致就拒绝, 免得把数据写坏。"""
        bad = []
        if abs(float(old.get("fps", 0)) - self.fps) > 1e-6:
            bad.append(f"fps {old.get('fps')} != {self.fps}")
        if old.get("robot_type") != self.robot_type:
            bad.append(f"robot_type {old.get('robot_type')} != {self.robot_type}")
        of = old.get("features", {})
        for k, v in self.features.items():
            ov = of.get(k)
            if ov is None:
                bad.append(f"缺少特征 {k}")
                continue
            if list(ov.get("shape", [])) != list(v["shape"]):
                bad.append(f"{k} shape {ov.get('shape')} != {v['shape']}")
            if ov.get("dtype") != v["dtype"]:
                bad.append(f"{k} dtype {ov.get('dtype')} != {v['dtype']}")
        if bad:
            raise RuntimeError(
                "已有数据集的配置与当前不一致, 拒绝续写(会把数据写坏):\n  - "
                + "\n  - ".join(bad)
                + f"\n如果确实要新建, 请换一个目录或删掉 {self.root}")

    def _pool_existing_stats(self):
        out = {}
        for row in self.ep_stats:
            st = row.get("stats") or {}
            for k, v in self.features.items():
                if k not in st:
                    continue
                if v["dtype"] == "video":
                    s = _ChannelStats.from_json(st[k])
                    if k not in out:
                        out[k] = _ChannelStats()
                    tgt = out[k]
                    # 池化: 用 count 加权合并
                    if tgt.n == 0:
                        tgt.n, tgt.mean, tgt.m2 = s.n, s.mean.copy(), s.m2.copy()
                        tgt.mn, tgt.mx = s.mn.copy(), s.mx.copy()
                    elif s.n > 0:
                        d = s.mean - tgt.mean
                        tot = tgt.n + s.n
                        tgt.mean = tgt.mean + d * (s.n / tot)
                        tgt.m2 = tgt.m2 + s.m2 + d ** 2 * (tgt.n * s.n / tot)
                        tgt.n = tot
                        tgt.mn = np.minimum(tgt.mn, s.mn)
                        tgt.mx = np.maximum(tgt.mx, s.mx)
                else:
                    s = _Stats.from_json(st[k], v["shape"])
                    out.setdefault(k, _Stats(v["shape"])).merge(s)
        return out

    # ---------- 写 ----------
    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def task_index(self, task):
        for t in self.tasks:
            if t["task"] == task:
                return int(t["task_index"])
        idx = len(self.tasks)
        self.tasks.append({"task_index": idx, "task": task})
        return idx

    def next_episode_index(self):
        return len(self.episodes)

    def begin_episode(self, task):
        """开始一集: 立刻打开视频管道, 之后 add_frame 边录边写。"""
        ti = self.task_index(task)
        return EpisodeSink(self, task, self.next_episode_index(), ti)

    def _commit(self, sink, success, session_id=None):
        """把 sink 里攒下的**标量行**落 parquet, 更新元数据。视频已经写完了。"""
        with self._lock:
            ep, n = sink.ep, sink.n
            if n == 0:
                raise ValueError("空 episode")
            st = np.asarray(sink._st, dtype=np.float32)
            ac = np.asarray(sink._ac, dtype=np.float32)
            ti = sink.task_index
            base = self.total_frames

            import pyarrow as pa
            import pyarrow.parquet as pq

            def f32lst(a2d):
                return pa.array([r.tolist() for r in np.asarray(a2d, dtype=np.float32)],
                                type=pa.list_(pa.float32()))

            def i64lst(vals):
                return pa.array([[int(v)] for v in vals], type=pa.list_(pa.int64()))

            cols = {
                "observation.state": f32lst(st),
                "action": f32lst(ac),
                "success": pa.array([[bool(success)]] * n, type=pa.list_(pa.bool_())),
                "timestamp": pa.array([[float(i) / self.fps] for i in range(n)],
                                      type=pa.list_(pa.float32())),
                "frame_index": i64lst(range(n)),
                "episode_index": i64lst([ep] * n),
                "index": i64lst(base + np.arange(n)),
                "task_index": i64lst([ti] * n),
            }
            dp = self.data_path(ep)
            os.makedirs(os.path.dirname(dp), exist_ok=True)
            pq.write_table(pa.table(cols), dp)

            # ---- 每集统计 + 全局池化 ----
            estats = {}
            arrs = {
                "observation.state": st, "action": ac,
                "success": np.full((n, 1), float(bool(success))),
                "timestamp": (np.arange(n, dtype=np.float64) / self.fps).reshape(n, 1),
                "frame_index": np.arange(n, dtype=np.float64).reshape(n, 1),
                "episode_index": np.full((n, 1), float(ep)),
                "index": (base + np.arange(n, dtype=np.float64)).reshape(n, 1),
                "task_index": np.full((n, 1), float(ti)),
            }
            for k in self._scalar_feats():
                s = _Stats(self.features[k]["shape"])
                s.update(arrs[k])
                estats[k] = s.to_json()
                self.global_stats.setdefault(k, _Stats(self.features[k]["shape"])).merge(s)
            for k in self.camera_keys:
                cs = sink._img_stats[k]
                estats[k] = cs.to_json()
                if k not in self.global_stats:
                    self.global_stats[k] = _ChannelStats()
                tgt = self.global_stats[k]
                if tgt.n == 0:
                    tgt.n, tgt.mean, tgt.m2 = cs.n, cs.mean.copy(), cs.m2.copy()
                    tgt.mn, tgt.mx = cs.mn.copy(), cs.mx.copy()
                elif cs.n > 0:
                    d = cs.mean - tgt.mean
                    tot = tgt.n + cs.n
                    tgt.mean = tgt.mean + d * (cs.n / tot)
                    tgt.m2 = tgt.m2 + cs.m2 + d ** 2 * (tgt.n * cs.n / tot)
                    tgt.n = tot
                    tgt.mn = np.minimum(tgt.mn, cs.mn)
                    tgt.mx = np.maximum(tgt.mx, cs.mx)

            # ---- 元数据 ----
            self.episodes.append({"episode_index": ep, "tasks": [sink.task],
                                  "length": n, "success": bool(success),
                                  "session_id": session_id})
            self.ep_stats.append({"episode_index": ep, "stats": estats})
            self.total_frames += n
            self._write_jsonl("episodes.jsonl", self.episodes)
            self._write_jsonl("episodes_stats.jsonl", self.ep_stats)
            self._write_jsonl("tasks.jsonl", self.tasks)
            self._write_info()
            self.finalize()
            return ep, n

    # 末级若只是包装目录（lerobot/），取上一级作为数据集名
    _WRAPPER_DIRS = ("lerobot",)

    def _dataset_name(self):
        """info.json 的 `name`：用数据集目录名。

        - `<名字>/lerobot/`  → 取 `<名字>`（本项目 first_train_test/lerobot）
        - `<名字>/`（直接就是数据集，如老的 dataset/）→ 用 `<名字>` 本身

        为什么放在**写入器里**而不是事后手改 info.json：每次落盘/收尾都会重写
        info.json，手加的字段时间一长必然被覆盖掉（2026-10-14 实测踩过）。
        """
        root = self.root.rstrip("/")
        base = os.path.basename(root)
        if base in self._WRAPPER_DIRS:
            up = os.path.basename(os.path.dirname(root))
            if up:
                base = up
        return base

    def _write_info(self):
        n_ep = len(self.episodes)
        info = {
            "codebase_version": "v2.1",
            "name": self._dataset_name(),
            "robot_type": self.robot_type,
            "total_episodes": n_ep,
            "total_frames": int(self.total_frames),
            "total_tasks": len(self.tasks),
            "total_videos": n_ep * len(self.camera_keys),
            "total_chunks": max(1, (n_ep + self.chunks_size - 1) // self.chunks_size),
            "chunks_size": self.chunks_size,
            "fps": self.fps,
            "splits": {"train": f"0:{n_ep}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/"
                          "episode_{episode_index:06d}.mp4",
            "features": self.features,
        }
        p = os.path.join(self.meta_dir, "info.json")
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        os.replace(tmp, p)

    def finalize(self):
        """写全局 stats.json + README。可重复调用。"""
        with self._lock:
            if not self.ok:
                return
            stats = {}
            for k, s in self.global_stats.items():
                if getattr(s, "count", None) == 0 or (hasattr(s, "n") and s.n == 0):
                    continue
                stats[k] = s.to_json()
            p = os.path.join(self.meta_dir, "stats.json")
            tmp = p + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False)
            os.replace(tmp, p)
            self._write_readme()

    def _write_readme(self):
        n_ep = len(self.episodes)
        ok = sum(1 for e in self.episodes if e.get("success"))
        lines = [
            "# VR 遥操作数据集 (LeRobot v2.1)",
            "",
            f"- robot_type: `{self.robot_type}`",
            f"- fps: {self.fps}",
            f"- episodes: {n_ep} (成功 {ok} / 失败 {n_ep - ok})",
            f"- frames: {self.total_frames}",
            f"- 相机: {', '.join(self.camera_keys)} @ "
            f"{self.image_size[1]}x{self.image_size[0]}",
            "",
            "## 动作空间 (π0 关节空间)",
            "",
            "`observation.state` = 实际关节角 qpos[0:7] (rad) + 夹爪实际开度;",
            "`action` = 下发的目标关节角 ctrl[0:7] (rad) + 夹爪指令。",
            "夹爪统一 **0=全开 / 1=全闭** (π0 约定)。共 8 维:",
            "",
            "```",
            " , ".join(FEATURE_NAMES),
            "```",
            "",
            "`success` (bool) = 该 episode 的人工标签 (左手柄 X=成功 / Y=失败), 逐帧同一值,",
            "训练时可直接按帧筛。",
            "",
            "## 采集流程",
            "",
            "1. 启动时在终端输入任务描述 → 进 tasks.jsonl (也是训练用的语言指令)",
            "2. 右手柄操作机械臂",
            "3. 左手柄**扳机** → 结束本段(切片)",
            "4. 左手柄 **X**=成功 / **Y**=失败 → 打标签并保存",
            "5. 右手柄**侧键**再按 → 开始下一轮",
        ]
        with open(os.path.join(self.root, "README.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


# ============================================================================
# 采集状态机 + 50Hz 采样线程
# ============================================================================
S_WAIT = "WAIT_RIGHT"        # 等右手柄侧键 → 开始录一段
S_REC = "RECORDING"          # 正在录
S_LABEL = "WAIT_LABEL"       # 等左手柄 X/Y 打标签


class DatasetRecorder:
    def __init__(self, root, task, model_path, cameras, image_size=(480, 640),
                 fps=50.0, cam_tilt=None, cam_offset=None, cam_abs=None, cam_extra=None,
                 render_off=(), max_episode_s=300.0, min_episode_s=0.2, session_id=None,
                 verbose=True):
        self.root = root
        self.task = task
        self.model_path = model_path
        self.cameras = list(cameras)
        self.image_size = tuple(image_size)
        self.fps = float(fps)
        self.cam_tilt = dict(cam_tilt or {})
        self.cam_abs = dict(cam_abs or {})     # 绝对覆盖（pos/quat/fovy），只改内存
        self.cam_extra = dict(cam_extra or {})  # 模型里没有的新相机，MjSpec 内存注入
        self.render_off = tuple(render_off or ())  # 关掉的渲染通道，见 camview.apply_render_flags
        self.cam_offset = dict(cam_offset or {})
        self.max_frames = int(max_episode_s * fps)
        self.min_frames = int(min_episode_s * fps)
        self.session_id = session_id
        self.verbose = verbose
        self.enabled = False
        self.state = S_WAIT
        self.episodes_done = 0
        self.episodes_skipped = 0
        self.frames_this_ep = 0
        self.n_slow = 0

        self._lk = threading.Lock()
        self._snap = None
        self._events = collections.deque()
        self._prev = None
        self._stop = threading.Event()
        self._th = None
        self._th_err = None
        self._writer = None
        self._sink = None

    # ---------------- 控制循环调用 (必须极便宜) ----------------
    def publish(self, qpos, ctrl, grip_state):
        with self._lk:
            if self._snap is None:
                self._snap = (np.array(qpos, dtype=np.float32),
                              np.array(ctrl, dtype=np.float32),
                              float(grip_state))
            else:
                self._snap[0][:] = qpos
                self._snap[1][:] = ctrl
                self._snap = (self._snap[0], self._snap[1], float(grip_state))

    def _snapshot(self):
        with self._lk:
            if self._snap is None:
                return None
            return (self._snap[0].copy(), self._snap[1].copy(), self._snap[2])

    def handle_hands(self, left, right):
        """边沿检测 (在 500Hz 控制循环里跑, 所以不会漏掉一次按键)。
        左手扳机=切片; 左手 X/Y=成功/失败; 右手侧键上升沿=开始下一轮。"""
        with self._lk:
            l = left or {}
            r = right or {}
            cur = (float(l.get("trigger", 0.0) or 0.0),
                   float(r.get("squeeze", 0.0) or 0.0),
                   bool(l.get("x", False)), bool(l.get("y", False)))
            prev = self._prev
            self._prev = cur
        if prev is None:
            return
        p_trig, p_sq, p_x, p_y = prev
        ev = []
        if cur[0] > 0.5 and p_trig <= 0.5:
            ev.append("left_trigger")
        if cur[2] and not p_x:
            ev.append("left_x")
        if cur[3] and not p_y:
            ev.append("left_y")
        if cur[1] > 0.5 and p_sq <= 0.5:
            ev.append("right_squeeze")
        if ev:
            with self._lk:
                self._events.extend(ev)

    # ---------------- 状态机 ----------------
    def _consume_events(self):
        with self._lk:
            ev = list(self._events)
            self._events.clear()
        for e in ev:
            if self.state == S_WAIT:
                if e == "right_squeeze":
                    self._begin_episode()
            elif self.state == S_REC:
                if e == "left_trigger":
                    self._end_episode()
            elif self.state == S_LABEL:
                if e == "left_x":
                    self._save_episode(True)
                elif e == "left_y":
                    self._save_episode(False)

    def _begin_episode(self):
        self.frames_this_ep = 0
        self._ep_t0 = time.perf_counter()
        self._ep_slow0 = self.n_slow
        try:
            self._sink = self._writer.begin_episode(self.task)
        except Exception as e:                                   # noqa: BLE001
            self._say(f"[数采] ✘ 开新段失败: {e}")
            traceback.print_exc()
            self._sink = None
            return
        self.state = S_REC
        self._say(f"[数采] ▶ 开始录制第 {self._sink.ep} 集 (任务: {self.task})")

    def _end_episode(self):
        n = self.frames_this_ep
        if self._sink is None:
            self.state = S_WAIT
            return
        if n < self.min_frames:
            self._sink.abort()
            self._sink = None
            self.episodes_skipped += 1
            self.state = S_WAIT
            self._say(f"[数采] ⚠ 这一段只有 {n} 帧 (<{self.min_frames}), 太短, 丢弃")
            return
        self.state = S_LABEL
        wall = max(1e-6, time.perf_counter() - getattr(self, "_ep_t0", time.perf_counter()))
        hz = n / wall
        slow = self.n_slow - getattr(self, "_ep_slow0", 0)
        warn = ""
        if hz < self.fps * 0.9:
            warn = (f"  ⚠ 只跑到 {hz:.1f}Hz (<{self.fps:.0f}Hz 的 90%), "
                    f"这一段在时间上被压缩了, 建议降低分辨率或关掉相机预览窗口")
        self._say(f"[数采] ⏸ 切片完成, 共 {n} 帧 / 实际 {wall:.1f}s = {hz:.1f}Hz "
                  f"(慢拍 {slow}, 累计 {self.n_slow})。"
                  f" 左手柄 X=成功 / Y=失败 → 打标签{warn}")

    def _save_episode(self, success):
        sink = self._sink
        self._sink = None
        if sink is None:
            self.state = S_WAIT
            return
        self.state = S_WAIT
        # ★ 失败集不入库（用户裁决 2026-10-14）：与上面的"太短丢弃"同构 ——
        #   绝不调用 sink.finish() ⇒ writer._commit() 未执行 ⇒ 集号不被消耗
        #   ⇒ 无需重编号、meta 不会不一致。
        if DROP_FAILED and not success:
            n = int(getattr(sink, "n", 0))
            try:
                sink.abort()                 # 关 ffmpeg 管道 + 删掉半截 mp4
            except Exception as e:           # noqa: BLE001
                self._say(f"[数采] ⚠ 丢弃时 abort 失败: {e}")
            self.episodes_skipped += 1
            self._say(f"[数采] ✘ 失败集已丢弃（不入库, 真删）: {n} 帧 —— "
                      f"不占用集号, 累计丢弃 {self.episodes_skipped}")
            self._say(f"[数采] 等右手柄侧键 → 开始第 "
                      f"{self._writer.next_episode_index()} 集")
            return
        try:
            ep, n = sink.finish(success, session_id=self.session_id)
            self.episodes_done += 1
            self._say(f"[数采] ✔ 第 {ep} 集已保存: {n} 帧, "
                      f"{'成功' if success else '失败'}  → {self.root}")
        except Exception as e:                                   # noqa: BLE001
            self._say(f"[数采] ✘ 保存失败: {e}")
            traceback.print_exc()
        self._say(f"[数采] 等右手柄侧键 → 开始第 {self._writer.next_episode_index()} 集")

    def _say(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def status_line(self):
        if not self.enabled:
            return ""
        s = {S_WAIT: "等待右手柄侧键开始", S_REC: "● 录制中",
             S_LABEL: "等待打标签(X成功/Y失败)"}[self.state]
        return (f"数采[{s}] 本段{self.frames_this_ep}帧 已存{self.episodes_done}集 "
                f"任务『{self.task}』")

    def short_status(self):
        if not self.enabled:
            return ""
        if self.state == S_REC:
            return f"●录{self.frames_this_ep}帧"
        if self.state == S_LABEL:
            return f"等标签(第{self._sink.ep if self._sink else '?'}集)"
        return f"等侧键(已存{self.episodes_done})"

    def _init_writer(self):
        self._writer = LeRobotWriter(
            self.root, fps=self.fps,
            robot_type=PROFILE.name,          # ★ 数据集里记录"这是哪台机器人"（换臂必须变）
            camera_keys=[k for k, _ in self.cameras],
            image_size=self.image_size, task=self.task, verbose=self.verbose)
        self.enabled = True
        self.state = S_WAIT
        self._say(f"[数采] 数据集就绪 → {self.root}\n"
                  f"       任务『{self.task}』  {self.fps:.0f}Hz  "
                  f"{self.image_size[1]}x{self.image_size[0]}  "
                  f"相机 {[k for k, _ in self.cameras]}\n"
                  f"       流程: 右手柄侧键开始 → 操作 → 左手柄扳机切片 → "
                  f"X成功/Y失败 → 右手柄侧键开始下一轮")

    # ---------------- 采样线程 ----------------
    def start(self):
        """初始化写入器(**在调用线程里**, 好让配置错误立刻报出来) 再起采样线程。"""
        if not self.enabled:
            self._init_writer()          # 失败会抛, 由调用方兜住
        self._th = threading.Thread(target=self._run, name="dataset", daemon=True)
        self._th.start()
        return self

    def stop(self):
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=8.0)
        try:
            if self._sink is not None:
                self._sink.abort()
            if self._writer is not None:
                self._writer.finalize()
        except Exception:                                        # noqa: BLE001
            pass

    def _run(self):
        try:
            import mujoco
            from camview import _tilt_cam_about_x, apply_abs_cams, apply_render_flags, \
                load_model_with_cams

            # 独立线程 → 独立 GL 上下文, 所以模型/Renderer 都在本线程建
            m = load_model_with_cams(self.model_path, self.cam_extra)
            d = mujoco.MjData(m)
            apply_abs_cams(m, self.cam_abs)     # ★ 绝对覆盖（含 fovy），先在全部相机上应用
            for key, cname in self.cameras:
                deg = self.cam_tilt.get(cname, 0.0)
                dv = self.cam_offset.get(cname, (0, 0, 0))
                cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cname)
                if cid < 0:
                    raise RuntimeError(f"模型里没有相机 '{cname}'")
                if abs(deg) > 1e-9:
                    m.cam_quat[cid] = _tilt_cam_about_x(m.cam_quat[cid], deg)
                if np.any(dv):
                    m.cam_pos[cid] = m.cam_pos[cid] + np.asarray(dv, float)
            H, W = self.image_size
            from glutil import gl_guard
            with gl_guard():          # GL 上下文创建必须串行 (见 glutil.py)
                ren = mujoco.Renderer(m, H, W)
            apply_render_flags(ren, self.render_off)
            cids = {key: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cname)
                    for key, cname in self.cameras}
            self._say(f"[数采] 相机已就绪 (渲染 {W}x{H} @ {self.fps:.0f}Hz)")
        except Exception as e:                                   # noqa: BLE001
            self._th_err = traceback.format_exc()
            self._say(f"[数采] 相机初始化失败, 不采集(遥操作不受影响): {e}")
            return

        period = 1.0 / self.fps
        t_next = time.perf_counter()
        prof_on = os.environ.get("VR_TELEOP_DS_PROF", "0") not in ("0", "", "false")
        prof = {"snap": [], "fwd": [], "cam": [], "wr": [], "st": [], "tot": [], "n": 0}
        try:
            while not self._stop.is_set():
                self._consume_events()
                if self.state == S_REC and self._sink is not None:
                    t_it = time.perf_counter()
                    snap = self._snapshot()
                    t_snap = time.perf_counter()
                    if snap is not None:
                        q, c, g = snap
                        d.qpos[:len(q)] = q
                        mujoco.mj_forward(m, d)
                        t_fwd = time.perf_counter()
                        st = np.concatenate([q[:PROFILE.n_arm_qpos], [g]]).astype(np.float32)
                        ac = np.concatenate([
                            c[:PROFILE.n_arm_actuators],
                            [float(c[PROFILE.gripper.ctrl_index]) / GRIP_CMD_MAX
                             if c.shape[0] > PROFILE.gripper.ctrl_index else 0.0]]
                        ).astype(np.float32)
                        # 逐相机: render → 立刻写走+算统计 → 再渲染下一路
                        # (ren.render() 复用同一块缓冲, 所以不能先收集再处理)
                        self._sink.new_frame()
                        t_r0 = time.perf_counter()
                        t_rend = t_wr = t_stt = 0.0
                        for key, _cname in self.cameras:
                            ren.update_scene(d, camera=cids[key])
                            fr = ren.render()
                            a = time.perf_counter()
                            t_rend += a - t_r0
                            self._sink.vw[key].write(fr)
                            b = time.perf_counter()
                            t_wr += b - a
                            self._sink._img_stats[key].add(fr)
                            t_r0 = time.perf_counter()
                            t_stt += t_r0 - b
                        self._sink.finish_frame(st, ac)
                        if prof_on:
                            t_cam = time.perf_counter()
                            prof["snap"].append((t_snap - t_it) * 1e3)
                            prof["fwd"].append((t_fwd - t_snap) * 1e3)
                            prof["cam"].append((t_rend) * 1e3)
                            prof["wr"].append(t_wr * 1e3)
                            prof["st"].append(t_stt * 1e3)
                            prof["tot"].append((t_cam - t_it) * 1e3)
                            prof["n"] += 1
                            if prof["n"] % 100 == 0:
                                m1 = lambda a: sorted(a)[len(a) // 2]            # noqa: E731
                                self._say(
                                    f"[数采PROF] 快照{m1(prof['snap']):.2f} "
                                    f"forward{m1(prof['fwd']):.2f} | "
                                    f"渲染{m1(prof['cam']):.2f} 写管道{m1(prof['wr']):.2f} "
                                    f"统计{m1(prof['st']):.2f} | "
                                    f"合计{m1(prof['tot']):.2f} ms/帧 "
                                    f"(预算 {period * 1e3:.0f}ms)")
                                for k in prof:
                                    if k != "n":
                                        prof[k].clear()
                        self.frames_this_ep += 1
                        if self.frames_this_ep >= self.max_frames:
                            self._say(f"[数采] ⚠ 单段超过 {self.max_frames} 帧, 自动切片")
                            self._end_episode()
                t_next += period
                now = time.perf_counter()
                if t_next < now - period:
                    # 落后超过一整个周期才放弃追赶(重新对齐), 否则下一轮不睡、
                    # 直接补上 —— 这样偶发的一次超时不会变成永久性漂移
                    self.n_slow += 1
                    t_next = now
                else:
                    dt = t_next - now
                    if dt > 0:
                        time.sleep(dt)
        except Exception:                                        # noqa: BLE001
            self._th_err = traceback.format_exc()
            self._say("[数采] 采集线程异常退出 (遥操作不受影响):")
            traceback.print_exc()
        finally:
            try:
                if self._sink is not None:
                    self._sink.abort()
                if self._writer is not None:
                    self._writer.finalize()
            except Exception:                                    # noqa: BLE001
                pass
            self._say(f"[数采] 已停止 (保存 {self.episodes_done} 集, "
                      f"丢弃 {self.episodes_skipped} 段, 慢拍 {self.n_slow})")
