"""运行日志: 每次运行把关节/末端/指令等数据落盘, 事后直接发文件就能定位问题。

设计要点 (都是从之前踩的坑换来的):
  · **二进制流式写**, 不在内存里攒 —— 长会话(30min)若攒内存会到 GB 级
  · 每 LOG_EVERY_STEPS 步采一行 (默认 250Hz)。500Hz 全量太占空间, 100Hz 又看不出
    单帧跳变。250Hz 是折中。
  · **周期性 flush**: 崩溃/强杀也留下可用数据
  · 配套 `.json` 记录 **列定义 + 配置快照 + 源码指纹 + 结束摘要** —— 拿到日志就能
    知道"这是哪一版、什么配置、跑成什么样", 不用再回头问
  · 另有 `.log` 文本事件流 (启动/回位/抓取/异常), 是读起来最快的那部分
  · 写入失败(磁盘满等)只关掉日志, **绝不影响遥操作**

用法 (在主循环里):
    lg = RunLogger(LOG_DIR, cols=..., meta=..., every=LOG_EVERY_STEPS, enabled=LOG_ENABLE)
    lg.event("抓取", "挂上 obj1")
    lg.write(t, values)          # values 与 columns 等长
    lg.close()
"""
import gzip
import hashlib
import json
import os
import time

import numpy as np


def file_fingerprint(path, n=12):
    """源码指纹: 用来确认"这段日志是哪个版本的代码跑出来的"。"""
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:n]
    except OSError:
        return "N/A"


class RunLogger:
    def __init__(self, log_dir, columns, meta=None, every=2, enabled=True,
                 flush_every=500, gzip_on_close=False):
        self._closed = False          # close() 要幂等 (atexit 可能再调一次)
        self.enabled = bool(enabled)
        self.dir = log_dir
        self.columns = list(columns)
        self.ncol = len(self.columns)
        self.every = max(1, int(every))
        self.flush_every = max(1, int(flush_every))
        self.gzip_on_close = gzip_on_close
        self.n = 0                 # 已写样本数
        self.step_seen = 0
        self.buf = []
        self.events = []
        self.t0_wall = time.time()
        self.t0 = time.time()
        self.errors = 0
        self.path_bin = None
        self.path_json = None
        self.path_log = None
        self._f = None
        self._flog = None
        self.meta = dict(meta or {})
        if self.enabled:
            try:
                os.makedirs(self.dir, exist_ok=True)
                stamp = time.strftime("%Y%m%d_%H%M%S")
                base = os.path.join(self.dir, f"run_{stamp}")
                self.path_bin = base + ".bin"
                self.path_json = base + ".json"
                self.path_log = base + ".log"
                self._f = open(self.path_bin, "wb")
                self._flog = open(self.path_log, "w", buffering=1)
                self.event("启动", f"列数={self.ncol} 采样间隔={self.every}步")
                # ★ 立刻先写一份"开头版"摘要 —— 万一进程被强杀(kill -9), 收尾的
                #   close() 不会执行, 但**列定义必须留下**, 否则日志就解不出来了。
                #   (真踩过: 用户 kill 掉进程, .bin 有 12MB 数据但 .json 是 0 字节。)
                self._dump_meta({"status": "运行中 (启动时写的开头版)"})
            except OSError as e:
                print(f"[日志] 无法建立日志目录/文件 ({e}) → 本次不记录")
                self.enabled = False
                self._f = self._flog = None

    # ---- 文本事件 (读起来最快) ----
    def event(self, tag, msg=""):
        try:
            rec = (f"{time.strftime('%H:%M:%S')}  "
                   f"t={self.n * self.every * 0.002:8.2f}s  [{tag}] {msg}")
            self.events.append(rec)
            if self._flog is not None:
                self._flog.write(rec + "\n")
        except Exception:            # noqa: BLE001 记日志绝不许影响主流程
            self.errors += 1

    # ---- 数值采样 ----
    def write(self, values, force=False):
        """每 every 次调用真正落一行。values 长度必须等于列数。"""
        if not self.enabled:
            return
        self.step_seen += 1
        if not force and (self.step_seen % self.every):
            return
        v = np.asarray(values, dtype=np.float32)
        if v.shape[0] != self.ncol:
            # 长度不对说明代码写错了; 补/裁到正确长度, 别让日志把主循环带崩
            self.errors += 1
            out = np.zeros(self.ncol, dtype=np.float32)
            k = min(self.ncol, v.shape[0])
            out[:k] = v[:k]
            v = out
        self.buf.append(v)
        self.n += 1
        if len(self.buf) >= self.flush_every:
            self._flush()

    def _flush(self):
        if not self.buf or self._f is None:
            self.buf.clear()
            return
        try:
            np.vstack(self.buf).tofile(self._f)
            self._f.flush()
        except (OSError, ValueError):
            self.errors += 1
            self.enabled = False          # 写不进去就关掉, 不拖累遥操作
        self.buf.clear()

    # ---- 摘要 (启动时先写一份, 收尾时覆盖成完整版) ----
    def _dump_meta(self, extra=None):
        meta = dict(self.meta)
        meta.update({
            "columns": self.columns,
            "ncol": self.ncol,
            "samples": self.n,
            "every_steps": self.every,
            "rows_per_sec": 500.0 / self.every,
            "dtype": "float32",
            "layout": "行主序, 每行 ncol 个 float32, 无表头 (列名见 columns)",
            "start_wall": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.t0_wall)),
            "write_errors": self.errors,
            "events": list(self.events),
            "read_hint": ("np.frombuffer(open(path,'rb').read(), dtype=np.float32)"
                          ".reshape(-1, ncol)  → 再按 columns 取列"),
        })
        if extra:
            meta.update(extra)
        try:
            with open(self.path_json, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
        except OSError:
            self.errors += 1

    # ---- 结束 ----
    def close(self, extra_meta=None, summary=None):
        if self._closed:
            return None
        self._closed = True
        if self._f is None and self.path_json is None:
            return None
        self._flush()
        try:
            if self._f is not None:
                self._f.close()
        except OSError:
            pass
        self._f = None
        wall = time.time() - self.t0
        try:
            size = os.path.getsize(self.path_bin) if self.path_bin else 0
        except OSError:
            size = 0
        extra = {"wall_seconds": round(wall, 2), "bin_bytes": size,
                 "end_wall": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "samples": self.n}
        if extra_meta:
            extra.update(extra_meta)
        if summary:
            extra["summary"] = summary
        self._dump_meta(extra)
        if self._flog is not None:
            try:
                self.event("结束", f"样本={self.n} 时长={wall:.1f}s 大小={size/1e6:.1f}MB"
                                   f" 写错误={self.errors}")
                self._flog.close()
            except OSError:
                pass
            self._flog = None
        if self.gzip_on_close and self.path_bin and os.path.exists(self.path_bin):
            try:
                with open(self.path_bin, "rb") as fi, \
                        gzip.open(self.path_bin + ".gz", "wb", compresslevel=6) as fo:
                    while True:
                        chunk = fi.read(1 << 20)
                        if not chunk:
                            break
                        fo.write(chunk)
                os.remove(self.path_bin)
                self.path_bin += ".gz"
            except OSError:
                pass
        return dict(bin=self.path_bin, json=self.path_json, log=self.path_log,
                    samples=self.n, bytes=size)
