"""双目显示: 把腕部相机和全局相机的视野**各自独立**开一个窗口显示。

给 VLA 数据采集看画面用。设计要点 (都是踩过的坑换来的):

· **独立线程**: 渲染+显示跑在自己的线程里, 控制循环 500Hz 一步都不能等它。
· **独立 MjData**: 显示线程用自己的 MjData, 绝不碰控制线程的 data。
  控制线程只把"当前 qpos 快照"写进一个共享数组 (36 个 float, 50Hz)。
· **GL 上下文与 tkinter 必须在同一线程**: 两者都绑线程, 所以 Renderer 和
  Tk 都在这个线程里创建。跨线程用会崩。
· **失败绝不拖累遥操作**: 整个线程包在 try/except 里; tkinter / GL / 显示
  任何一个起不来, 就打印一行然后退出, 遥操作照常。
· 窗口可以单独关掉 (关掉就停, 不影响仿真)。

用法 (在 server 里):
    view = CameraViewer(MODEL_PATH, [("全局", "cam_scene"), ("腕部", "wrist")])
    view.start()                 # 起线程
    view.publish(data.qpos)      # 控制循环里调 (它会自己限频)
    view.stop()                  # 收尾
"""
import os
import threading
import time
import traceback

import numpy as np
import mujoco


def _mat_to_quat_wxyz(R):
    """3x3 旋转矩阵 → 四元数 (w,x,y,z)。MuJoCo 的 cam_quat 是这个顺序。"""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                      (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                      0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def _tilt_cam_about_x(q_orig_wxyz, deg):
    """在相机**自身**坐标系里绕 x 轴转 deg 度 (在原有朝向上叠加)。

    相机光轴 = 自身 -z。绕自身 x 轴转 +deg 会让光轴的 y 分量变正
    → 对腕部相机就是"朝夹爪那一侧偏"(夹爪在相机局部系的 +y 侧)。
    """
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)
    w, x, y, z = q_orig_wxyz
    R_orig = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return _mat_to_quat_wxyz(Rx @ R_orig)


def apply_render_flags(renderer, off=()):
    """关掉渲染器的某些通道 —— 用来给 50Hz 腾时间。

    off: 可迭代的通道名，支持 'shadow' / 'reflection' / 'skybox' / 'fog' / 'haze'
         （对应 mjtRndFlag 的 mjRND_*）。

    ★ 为什么需要它（2026-10-16 实测，640x480）:
        mjr_render 默认会为**阴影**和**反射**各多渲一趟整个场景 —— 单帧 6.12ms；
        关阴影 4.06ms；再关反射 **1.81ms**；再预分配 out 缓冲 1.62ms。
        加第三路相机后 3×6.12≈18.4ms 已经顶满 20ms 预算（实测只跑到 37Hz），
        关掉反射后 3×1.81≈5.4ms，50Hz 就回来了。
    """
    if not off:
        return
    F = mujoco.mjtRndFlag
    for name in off:
        f = getattr(F, "mjRND_" + str(name).upper(), None)
        if f is None:
            print(f"[渲染] 未知通道 '{name}'，忽略")
            continue
        renderer.scene.flags[f] = 0
    print(f"[渲染] 已关闭通道: {', '.join(off)}")


def load_model_with_cams(model_path, extra_cams=None):
    """加载模型，并把 extra_cams 里的**新相机注入到内存模型** —— 不改任何 XML 文件。

    extra_cams: {相机名: {"pos":(3,), "quat":(4,), "fovy":float}}
      已经存在的相机名会被忽略（不要用这个去改已有相机，那个走 apply_abs_cams）。

    ★ 为什么需要它: MuJoCo 编译完成后 ncam 是固定的，加不了新相机；
      但 `MjSpec`（3.x）可以在**编译前**往世界里加 camera。
      实测 `MjSpec.from_file(p).compile()` 与 `MjModel.from_xml_path(p)` 编译出的模型
      除相机数外**完全一致**（nq/nv/nu/nbody/ngeom… 全同，同一 qpos 步进后 xpos 差 0.0）。
    """
    if not extra_cams:
        return mujoco.MjModel.from_xml_path(model_path)
    spec = mujoco.MjSpec.from_file(model_path)
    have = {c.name for c in spec.cameras}
    for name, ov in extra_cams.items():
        if name in have:
            continue
        spec.worldbody.add_camera(
            name=name,
            pos=[float(v) for v in ov.get("pos", (0.0, 0.0, 1.0))],
            quat=[float(v) for v in ov.get("quat", (1.0, 0.0, 0.0, 0.0))],
            fovy=float(ov.get("fovy", 45.0)))
        print(f"[相机] 新增相机 '{name}' (内存注入, XML 未改): "
              f"pos={np.round(np.asarray(ov.get('pos'), float), 5).tolist()} fovy={ov.get('fovy')}")
    return spec.compile()


def apply_abs_cams(model, abs_cams):
    """把绝对参数 (pos/quat/fovy) 覆盖到**内存里的**相机上 —— 不动任何 XML 文件。

    abs_cams: {相机名: {"pos":(3,)|None, "quat":(4,)|None, "fovy":float|None}}
    ★ 为什么需要它: 有些调整（例如同时改目标点+俯角）**不是**"绕相机自身 x 轴转"
      能表达的（实测 cam_scene 的残差 0.177 ≫ 1e-6），只能直接给绝对位姿。
    ★ 恢复原状: 把 abs_cams 置空即可（XML 原始值从未被写回）。
    """
    if not abs_cams:
        return
    for cname, ov in abs_cams.items():
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cname)
        if cid < 0:
            print(f"[相机] ✗ 模型里没有相机 '{cname}'，跳过")
            continue
        if ov.get("pos") is not None:
            model.cam_pos[cid] = np.asarray(ov["pos"], float)
        if ov.get("quat") is not None:
            model.cam_quat[cid] = np.asarray(ov["quat"], float)
        if ov.get("fovy") is not None:
            model.cam_fovy[cid] = float(ov["fovy"])
        print(f"[相机] '{cname}' 绝对覆盖: pos={np.round(model.cam_pos[cid], 5).tolist()} "
              f"quat={np.round(model.cam_quat[cid], 4).tolist()} "
              f"fovy={float(model.cam_fovy[cid]):.2f}")


class CameraViewer:
    def __init__(self, model_path, cameras, size=(640, 480), fps=20.0,
                 title="VR Teleop 相机视图", enabled=True, dump_dir=None,
                 dump_every=0, positions=None, tilt_deg=None, offset=None, abs_cams=None,
                 extra_cams=None, render_off=()):
        self.model_path = model_path
        self.cameras = list(cameras)          # [(显示名, 模型里的相机名), ...]
        self.size = size
        self.fps = max(1.0, float(fps))
        self.title = title
        self.enabled = bool(enabled)
        # 相机"偏转": {相机名: 绕相机自身 x 轴的偏转角(度)}
        # 只改**内存里** m.cam_quat (相对父 body 的局部四元数), 不动任何 XML 文件。
        # 对腕部相机而言 +角度 = 光轴朝夹爪侧偏(夹爪进入画面), 见 tools/render_wrist_tilt.py。
        self.tilt_deg = dict(tilt_deg) if tilt_deg else {}
        # 相机"位移": {相机名: (dx,dy,dz)} —— 在**父 body 局部系**里的平移(米)。
        # 腕部相机: -z = 沿工具轴往外伸(离开臂体/伸出掌部), 见 tools/render_wrist_far.py。
        self.offset = dict(offset) if offset else {}
        # 绝对覆盖: {相机名: {"pos":(3,), "quat":(4,), "fovy":float}} —— 只改内存, 不动 XML。
        # 在 tilt/offset **之前**应用, 所以两者可共存(这里两者都为空, 见 server.CAMVIEW_ABS)。
        self.abs_cams = dict(abs_cams) if abs_cams else {}
        # 新增相机: {相机名: {"pos","quat","fovy"}} —— 模型里本来没有, 用 MjSpec 在
        # 编译前注入(同样不改 XML)。见 camview.load_model_with_cams()。
        self.extra_cams = dict(extra_cams) if extra_cams else {}
        # 关掉的渲染通道 (阴影/反射…) —— 为了给 50Hz 腾时间, 见 apply_render_flags()
        self.render_off = tuple(render_off or ())
        # 落盘调试: dump_dir 非空则把"正在显示的帧"存成 png (每 dump_every 帧存一张,
        # 0 = 每帧都存)。用来验证"窗口里到底显示的是什么", 不用靠桌面截图。
        self.dump_dir = dump_dir
        self.dump_every = int(dump_every)
        # 每个窗口的摆放位置 [("+x+y"), ...], None = 交给窗口管理器
        self.positions = list(positions) if positions else []
        self._lock = threading.Lock()
        self._qpos = None                     # 控制线程发布的最新 qpos
        self._pub_t = 0.0
        self._stop = threading.Event()
        self._th = None
        self._err = None
        self.frames = 0                       # 已显示帧数 (诊断用)
        self.ok = False                       # 真的起来了吗

    # ---- 控制线程调用 (必须极便宜: 只在限频点拷贝一次) ----
    def publish(self, qpos, min_interval=0.02):
        if not self.enabled:
            return
        now = time.time()
        if now - self._pub_t < min_interval:      # 50Hz 足够
            return
        self._pub_t = now
        with self._lock:
            if self._qpos is None:
                self._qpos = np.array(qpos, dtype=np.float64)
            else:
                self._qpos[:] = qpos

    def _snapshot(self):
        with self._lock:
            return None if self._qpos is None else self._qpos.copy()

    def stop(self):
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=3.0)

    # ---- 显示线程 ----
    def _run(self):
        try:
            import tkinter as tk
            from PIL import Image, ImageTk
        except Exception as e:                                # noqa: BLE001
            print(f"[相机视图] 起不来 (缺 tkinter/PIL): {e}; 遥操作不受影响")
            return
        try:
            m = load_model_with_cams(self.model_path, self.extra_cams)
            d = mujoco.MjData(m)
            # 相机偏转/位移: 只改内存里的局部量 (相对父 body), 不动 XML
            apply_abs_cams(m, self.abs_cams)          # ★ 绝对覆盖（含 fovy），最先应用
            for cname, deg in self.tilt_deg.items():
                if abs(deg) < 1e-9:
                    continue
                ccid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cname)
                if ccid < 0:
                    print(f"[相机视图] 偏转跳过: 模型里没有相机 '{cname}'")
                    continue
                m.cam_quat[ccid] = _tilt_cam_about_x(m.cam_quat[ccid], deg)
                print(f"[相机视图] '{cname}' 绕自身 x 轴偏 {deg:+.1f}° "
                      f"(局部四元数 {np.round(m.cam_quat[ccid], 4).tolist()})")
            for cname, dv in self.offset.items():
                dv = np.asarray(dv, float)
                if not np.any(dv):
                    continue
                ccid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cname)
                if ccid < 0:
                    print(f"[相机视图] 位移跳过: 模型里没有相机 '{cname}'")
                    continue
                m.cam_pos[ccid] = m.cam_pos[ccid] + dv
                print(f"[相机视图] '{cname}' 局部平移 {dv.tolist()} → "
                      f"pos={np.round(m.cam_pos[ccid], 5).tolist()}")
            W, H = self.size
            from glutil import gl_guard
            with gl_guard():          # GL 上下文创建必须串行 (见 glutil.py)
                r = mujoco.Renderer(m, H, W)
            apply_render_flags(r, self.render_off)
        except Exception as e:                                # noqa: BLE001
            print(f"[相机视图] 模型/渲染器创建失败: {e}; 遥操作不受影响")
            return

        # 解析相机 id, 跳过模型里没有的
        cams = []
        for label, cname in self.cameras:
            cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cname)
            if cid < 0:
                print(f"[相机视图] 模型里没有相机 '{cname}', 跳过")
                continue
            cams.append((label, cname, cid))
        if not cams:
            print("[相机视图] 没有任何可用相机; 关闭")
            r.close()
            return

        try:
            root = tk.Tk()
            root.title(self.title)
            root.withdraw()          # root 本身不显示 (否则桌面上多一个空白窗口)
            wins = []
            for label, cname, cid in cams:
                w = tk.Toplevel(root)
                w.title(f"{label}  [{cname}]  {W}x{H}")
                if self.positions:
                    pos = self.positions[len(wins) % len(self.positions)]
                    if pos:
                        try:
                            w.geometry(pos)
                        except Exception:                     # noqa: BLE001
                            pass
                w.protocol("WM_DELETE_WINDOW", lambda: None)   # 用按钮/关主窗退出
                lbl = tk.Label(w, bg="#111")
                lbl.pack()
                wins.append((w, lbl))
            root.protocol("WM_DELETE_WINDOW", self._stop.set)
            root.update()
            print(f"[相机视图] 已开 {len(cams)} 个独立窗口: "
                  + ", ".join(f"{l}({c})" for l, c, _ in cams)
                  + f"   {W}x{H}@{self.fps:.0f}fps")
            self.ok = True
        except Exception as e:                                # noqa: BLE001
            print(f"[相机视图] 建窗口失败: {e}; 遥操作不受影响")
            try:
                r.close()
            except Exception:                                 # noqa: BLE001
                pass
            return

        period = 1.0 / self.fps
        t_last = 0.0
        try:
            while not self._stop.is_set():
                # 窗口被用户关掉 → 结束
                try:
                    if not wins[0][0].winfo_exists():
                        print("[相机视图] 窗口已关闭")
                        break
                except Exception:                             # noqa: BLE001
                    break
                q = self._snapshot()
                if q is None:
                    root.update()
                    time.sleep(0.02)
                    continue
                if q.shape[0] == d.qpos.shape[0]:
                    d.qpos[:] = q
                else:
                    n = min(q.shape[0], d.qpos.shape[0])
                    d.qpos[:n] = q[:n]
                mujoco.mj_forward(m, d)
                for (w, lbl), (label, cname, cid) in zip(wins, cams):
                    r.update_scene(d, camera=cid)
                    arr = r.render()
                    img = Image.fromarray(arr)
                    ph = ImageTk.PhotoImage(img)
                    lbl.configure(image=ph)
                    lbl.image = ph                            # 防被 GC
                    if self.dump_dir and (self.dump_every <= 0
                                          or self.frames % self.dump_every == 0):
                        try:
                            img.save(os.path.join(
                                self.dump_dir, f"{cname}_{self.frames:06d}.png"))
                        except Exception:                     # noqa: BLE001
                            pass
                root.update()
                self.frames += 1
                # 限频
                now = time.time()
                dt = period - (now - t_last)
                if dt > 0:
                    time.sleep(dt)
                t_last = time.time()
        except Exception:                                     # noqa: BLE001
            print("[相机视图] 显示线程异常退出 (遥操作不受影响):")
            traceback.print_exc()
        finally:
            self.ok = False
            try:
                root.destroy()
            except Exception:                                 # noqa: BLE001
                pass
            try:
                r.close()
            except Exception:                                 # noqa: BLE001
                pass
            print(f"[相机视图] 已停止 (共显示 {self.frames} 帧)")

    def start(self):
        if not self.enabled:
            print("[相机视图] 已关闭 (VR_TELEOP_CAMVIEW=0 / CAMVIEW_SHOW=False)")
            return self
        self._th = threading.Thread(target=self._run, name="camview", daemon=True)
        self._th.start()
        return self
