"""GLFW/OpenGL 上下文创建的全局串行化。

★ 为什么需要这个文件 (实测踩出来的崩溃)
----------------------------------------
这个进程里有**三处**要用 OpenGL:
  1. `mujoco.viewer.launch_passive` —— 主 viewer 窗口 (它自己起线程)
  2. `camview.py` —— 两个相机预览窗口 (独立线程 + 独立 Renderer)
  3. `dataset_recorder.py` —— 采集时 50Hz 渲染两路相机写进 mp4 (另一个独立 Renderer)

三路线程**同时**首次调 `glfw.init()` 时, GLFW 内部的 X11 后端会把错误处理器装两次:

    python: glfw-3.4/src/x11_init.c:1099: _glfwGrabErrorHandlerX11:
        Assertion `_glfw.x11.errorHandler == NULL' failed.
    Fatal Python error: Aborted      ← 整个进程直接 abort, 连 traceback 都没有

表现是"有时能跑、有时一启动就死"—— 典型竞态。GLFW 本身明确要求
"所有调用必须在同一个线程里", 所以正确做法是:

  1. 在**任何线程启动之前**, 在前台线程里把 `glfw.init()` 做掉 (之后各线程里的
     `glfw.init()` 就是幂等空调用, 不会再走平台初始化);
  2. 用一把进程级锁把所有 GL 上下文的**创建**串行化 (`create_window` 也碰 X11)。

注意: 这里只锁"创建/初始化", **不锁渲染** —— 渲染各自持有独立上下文, 实测可以并发
(viewer + camview 这样并行跑了几周没问题)。
"""
from __future__ import annotations

import contextlib
import threading

try:
    import glfw as _glfw
except Exception:                                                # noqa: BLE001
    _glfw = None

GL_LOCK = threading.RLock()
_inited = False
_init_err = None


def init_glfw():
    """在前台线程调用一次。幂等; 失败不抛(后面创建 Renderer 时会给出真正的报错)。"""
    global _inited, _init_err
    with GL_LOCK:
        if _inited or _glfw is None:
            return _inited
        try:
            _inited = bool(_glfw.init())
            if not _inited:
                _init_err = "glfw.init() 返回 False (没有可用的显示器/DISPLAY?)"
        except Exception as e:                                   # noqa: BLE001
            _init_err = str(e)
        return _inited


@contextlib.contextmanager
def gl_guard():
    """包住所有 `mujoco.Renderer(...)` 之类的构造, 保证 GL 上下文创建不并发。"""
    with GL_LOCK:
        if not _inited:
            init_glfw()
        yield


def status():
    return {"glfw": _glfw is not None, "inited": _inited, "err": _init_err}
