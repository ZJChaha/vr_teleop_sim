# 06 · 故障排查

按"现象 → 怎么确认 → 怎么修"组织。每条都来自真实踩坑。

---

## A. 头显连不上

### A1. 页面打不开 / 连接被拒

```bash
ss -ltn | grep 8444            # ① 服务在听吗？（应该是 0.0.0.0:8444）
curl -k https://127.0.0.1:8444 -o /dev/null -w '%{http_code}\n'   # ② 本机自己能访问吗
ip -4 addr show | grep inet    # ③ 头显用的那个 IP 还对吗
```

| 检查结果 | 结论 |
|---|---|
| 8444 没在听 | 服务没起 / 起崩了（看终端输出） |
| 本机 200，头显打不开 | 防火墙 或 路由器隔离（见 A3） |
| 地址栏自动变成 `http://` | 8444 **只**提供 TLS，必须 `https://` |
| 昨天能、今天不能 | IP 变了 → 重签证书（A2） |

### A2. 证书报错 / 换网后连不上

自签证书把 **IP 写进了 SAN**，IP 变了浏览器就拒绝。

```bash
./scripts/make_cert.sh          # 重签（会把本机所有 IP 写进去）
# 重启 ./run.sh
```

多网卡/多网段时，把可能用到的 IP 一起加：

```bash
./scripts/make_cert.sh 192.168.1.50 10.0.0.7
```

> 症状对照：`ERR_CERT_COMMON_NAME_INVALID` / "证书对 xxx 无效" = SAN 不含当前 IP。

### A3. 防火墙 / 路由器隔离

```bash
sudo ufw status verbose
sudo ufw allow 8444/tcp          # 只开这一个端口
```

如果 `ufw` 显示 allow 仍然不通，且手机/Pad 也连不上，多半是路由器开了
**AP isolation（客户端隔离）**：设备能上网但互相不可见。
换个路由器 / 手机热点（热点也可能开隔离）。

> 快速二分：在电脑上起一个纯 HTTP 小服务
> （`python3 -m http.server 8080`），头显打开 `http://<IP>:8080`。
> 能打开 → 网络/防火墙没问题，问题在 8444/TLS；打不开 → 网络层。

### A4. WS 连上又断 / 一直"未连接(重试中)"

* 证书没点"继续前往" → 页面在安全上下文外，`wss` 被拦
* 电脑休眠 / WiFi 换网 → 重连逻辑每 3 秒重试，但 socket 可能半死
* 服务在重启空档 → 看终端是否刚启动

---

## B. 操作手感

### B1. "一顿一顿"，像掉帧

**根因通常不是性能，是这三件事之一**：

1. **viewer 同步频率被"降频"**。用墙钟节流实现降频会把周期节拍变成相位漂移：
   ```
   每 5 步无节流  → 99.8Hz，间隔 10.00ms ± 0.09ms   ✅
   每 8 步+节流   → 31.3Hz，间隔 31.7ms ± 1.77ms    ❌ 肉眼就是卡
   ```
   不要用墙钟去"降频"。
2. **任何人为目标加的"墙"**（位移球上限、姿态角上限、抗积分饱和）：
   手越过边界目标就停住，反向一动又跳回来 → 典型的一顿一顿。
   **参考实现里这些全被撤掉了**，手到哪末端到哪。
3. **home 姿态贴着关节限位**：IK 在限位附近解不唯一、跳变。
   用 `check_profile.py` 看各关节余量，或换一个更居中的 home。

### B2. 手腕姿态跟随不对 / 拧手时乱转

* 检查 `teleop.rot_signs`（见 docs/03 第 9 节）—— 三项分别对应 pitch/yaw/roll，
  只有一项反了也会表现成"整体乱"。
* 无限旋转关节（Kinova j1/j3/j5/j7）**不要按 ctrlrange 裁剪**，否则
  解会在 ±π 处跳。

### B3. 手要划很大幅度才动一点

`teleop.scale` 调大（1.2~1.5）。反过来"太灵敏、够不到远处"就调小。

### B4. 网络延迟/抖动

```bash
# 量手柄数据到达间隔（不是循环耗时）
# 日志里 hand_age_ms 就是"手柄数据距今多久"；>50ms 人就能感觉到卡
```

已确认的两个 PC 侧元凶（都会周期性抢网络/CPU）：

```bash
iw dev wlo1 get power_save            # on → 关掉（省电会周期性休眠网卡）
nmcli connection modify <SSID> 802-11-wireless.powersave 2

ps -eo pcpu,args --sort=-pcpu | head  # 有没有后台进程占满一核
systemctl status whoopsie             # Ubuntu 崩溃上报，上传不通时会死转
sudo systemctl mask whoopsie          # 它 Restart=always，stop 会被拉回来
```

---

## C. 抓取

### C1. 夹空 / 停在方块上方

**先量 `tcp_offset`**（docs/03 第 6 节）。`site` 往往在两指垫**外面几厘米**，
直接拿它当抓取中心就是"对准了却抓空"。

### C2. 夹住但一抬就掉

* 夹持力不足：参考实现的 2F-85 `forcerange` 只有 ±5N，抬升时被方块反推张开。
  解法是并联一个力执行器（`grip_boost`），或直接调大 `forcerange`。
  真机 2F-85 的 rFR 0~255 对应指垫 20~235N，仿真里要保证同量级。
* 摩擦系数太低：`<geom friction="1.0 0.01 0.001">`，方块建议 ≥1.0。
* 方块太小：两指垫组高约 ±18mm，物体高过 ~40mm 时掌部会压到顶面，接触力爆炸。

### C3. 抓得住但抬不起来（参考实现的已知现象）

竖直朝下抓取时，方块会被"夹住"但抬升 Δz 只有 ~10~90mm（正常应 ~160mm+）。
根因未完全钉死（疑似指垫接触点太少 + 腕部姿态接近限位）。
**规避**：用斜朝前 45° 的 home 姿态，或改用两指夹持面更大的模型。

### C4. 抓取瞬间方块被弹飞

* 接触求解器收到"瞬移"注入的能量 —— 确认 `DRIVE_MODE="auto"` 且
  `PHYS_TRIGGER_MIN` / `TELEPORT_RAMP` 生效（扣扳机后才切伺服）。
* `timestep` 太大（500Hz = 0.002s 是参考值）。
* 方块初始位置与夹爪/桌面穿模。

---

## D. 数据采集

### D1. 数采只有 37Hz，达不到 50Hz

三路 640×480 渲染吃满预算。按顺序试：

1. **关掉一路的 `record`** —— 数据集两路就够（腕部 + 侧面全局/俯视全局二选一），
   见 docs/04 第 3 节
2. **开 `render_off=("shadow","reflection")`** —— 单帧 6.12ms → 1.81ms，见 docs/04 第 5 节
3. 降分辨率 / 关预览窗口（`preview=False` 或 `VR_TELEOP_CAMVIEW=0`）

### D2. 想往旧数据集继续采，报"配置不一致"

`_check_compatible()` 拦住了 —— 特征集（相机路数/分辨率/关节数/fps）变了。
这是**故意的**（否则会把数据写坏）。换目录：

```bash
VR_TELEOP_DS_ROOT=data/lerobot_v2 ./run.sh
```

### D3. 上次 kill -9 留下的半截视频

启动时写入器会自动清理"编号 ≥ 已记录集数"的残留文件。
如果清理后仍异常，检查 `meta/episodes.jsonl` 与 `data/` 的编号是否对得上。

### D4. 夹爪在数据里一直是常数

`gripper.qpos_index` 填错，或 `qpos_at_open/close` 填反。
跑 `tools/check_profile.py` 的"夹爪方向一致"检查。

---

## E. 启动 / 退出

### E1. 启动就崩，没有 traceback

多半是 **GLFW 在多线程里首次初始化**导致 X11 断言。参考实现的做法是
在**主线程、任何线程启动之前**先 `glfwInit()`（见 `glutil.init_glfw()`），
三路线程再各自建上下文。不要删这一段。

### E2. 端口被占

```bash
ss -ltnp | grep 8444
```

**先绑端口再起仿真线程**是有意为之：否则端口失败后仿真线程已经构造了
`DatasetRecorder`，它的"清理残留集"逻辑会删掉另一个实例刚采的数据。

### E3. Ctrl-C 后卡很久才退出

aiohttp 的优雅关闭会等所有 WebSocket handler 返回；头显没断开时实测白等 **122 秒**。
参考实现：退出时主动 `close()` 所有活动 WS + `shutdown_timeout=1.0`。
如果你自己加了新的长连接，记得也登记到 `_LIVE_WS`。

### E4. 相机窗口黑屏 / 报 GL 错误

* 必须是 **X11**（Wayland 下 tkinter + GL 容易崩）
* 三个线程不能**同时**首次创建 GL 上下文 → 用 `glutil.gl_guard()` 串行化
* `MUJOCO_GL=egl` 在没有 EGL 的机器上会失败，保持默认（glfw）

---

## F. 换机器人之后

症状几乎总在这几处，按顺序查：

```
1. tools/check_profile.py       ← 名字/下标/刻度，一口气全查
2. tools/probe_model.py         ← 模型里到底有什么
3. 单步 IK 试探（不动 viewer）：
      python -c "import server as S; rig=S.Rig(); print(rig.tcp())"
4. 相机对不对：先 MERGE 之前先看 home 姿态渲染
5. 可达区：从"终点"和"整条接近轨迹"两个角度各测一遍
```

---

## G. 把日志交给"另一个 agent"

出问题时收集这三份（同一时间戳）：

```
logs/run_<stamp>.log    文本事件
logs/run_<stamp>.json   列定义 + 配置快照 + 源码指纹
logs/run_<stamp>.bin    250Hz 数值
```

配上 `git rev-parse HEAD` 的输出，另一个 agent 就能完整复原现场。
详见 `docs/07_AGENT_HANDOFF.md`。
