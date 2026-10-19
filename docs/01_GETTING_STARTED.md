# 01 · 安装与第一次连上

## 1. 环境要求

| 项 | 要求 |
|---|---|
| 系统 | Linux（X11）。**Wayland 下 MuJoCo viewer 与 tkinter 预览窗会出问题**，请用 X11 会话 |
| Python | 3.9+（实测 3.11） |
| OpenGL | 3.3+，能用硬件渲染即可（`glxinfo | grep "OpenGL renderer"`） |
| 依赖 | `mujoco`、`numpy`、`aiohttp`、`pillow`（见 requirements.txt） |
| 头显 | Meta Quest，浏览器支持 **WebXR immersive-vr**（Quest 自带浏览器即可） |
| 网络 | 电脑与头显在**同一个局域网**（同一个路由器/WiFi） |

> 为什么必须要 HTTPS：WebXR 只在**安全上下文**（`https://` 或 `localhost`）下可用。
> 所以服务端用自签证书跑 HTTPS，头显里点一次"继续访问"即可，无需买证书。

## 2. 安装

```bash
cd vr-teleop-sim
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt
```

`./run.sh` 会自动使用仓库里的 `.venv`，不用手动 activate。
想用别的解释器：`VR_TELEOP_PY=/path/to/python ./run.sh`。

## 3. 自检（强烈建议先做）

```bash
.venv/bin/python tools/check_profile.py
```

输出全部 ✅ 才继续。它检查：模型能否加载、关节/执行器数量、site / body / 相机名、
home 位姿是否越限、**夹爪开合方向是否与配置一致**（会真的驱动一次夹爪）。

## 4. 证书

第一次运行 `./run.sh` 时，如果目录里没有 `cert.pem` / `key.pem`，会**自动生成**。
也可以手动重签（换网、换 IP 后必须重签）：

```bash
./scripts/make_cert.sh
# 或者把以后可能用到的 IP 一起写进 SAN:
./scripts/make_cert.sh 192.168.1.50 10.0.0.7
```

脚本会把本机**所有** IPv4 写进 subjectAltName，并打印所有可用的
`https://<ip>:8444` 地址。

> 旧证书会备份到 `.cert_backup/`（可用 `VR_TELEOP_BACKUP_DIR` 改），不会覆盖丢失。

## 5. 启动

```bash
./run.sh
```

两个启动提问：

```
要采集数据吗? [y/N]                     ← n: 只遥操作；y: 顺便录 LeRobot 数据集
本次任务描述（写进数据集的语言指令）: 把红色方块放到黄色方块上
```

也可以完全跳过提问（适合脚本/CI）：

```bash
VR_TELEOP_DS=0 ./run.sh                                  # 不采集
VR_TELEOP_DS=1 VR_TELEOP_DS_TASK="把方块放进盘子" ./run.sh  # 采集
```

启动后应当看到：

1. 终端 `[HTTP] 服务已启动: https://<本机IP>:8444`
2. 一个 **MuJoCo viewer** 窗口（机械臂 + 桌子）
3. **三个相机预览窗口**（全局 / 腕部 / 全局俯视）

## 6. 头显连接

1. 查电脑 IP：`ip -4 addr show | grep inet`（选和头显同网段的那个）
2. Quest 浏览器打开 `https://<那个IP>:8444`
3. 证书警告 → **高级 → 继续前往**
4. 页面顶部会显示 `WebXR: 支持` / `WS: 已连接`
5. 点 **进入 VR**，戴上头显

### 连不上？按这个顺序查（30 秒定位）

```bash
ss -ltn | grep 8444                                  # ① 服务在听吗（端口绑定 0.0.0.0）
ping <头显IP>                                        # ② 头显在线吗
curl -k https://<电脑IP>:8444 -o /dev/null -w '%{http_code}\n'   # ③ 本机自己能访问吗
sudo ufw status                                      # ④ 防火墙拦了吗
```

| 现象 | 原因 |
|---|---|
| 地址栏被浏览器改成 `http://` | 8444 **只**提供 TLS，明文 HTTP 会被拒 → 手输 `https://` |
| 一直转圈、WS 未连接 | 证书没点"继续"，或服务在重启空档 |
| 完全打不开、ping 通 | 防火墙 / 路由器"客户端隔离（AP isolation）" |
| 昨天能连、今天不能 | IP 变了 → 重签证书，或直接用新 IP |

## 7. 停止

`Ctrl-C`。程序会主动关掉 WebSocket 并把数据集收尾（**不会**白等头显断开）。

> 不要 `kill -9`。SIGKILL 会留下半截 mp4，下次启动时写入器会自行清理，
> 但当前这一集的数据会丢。

## 8. 环境变量速查

| 变量 | 默认 | 作用 |
|---|---|---|
| `VR_TELEOP_PROFILE` | 空 | 用哪个配置模块（如 `robot_profile_myarm`） |
| `VR_TELEOP_MODEL` | profile 里的 | 覆盖模型路径 |
| `VR_TELEOP_PORT` | 8444 | 换端口（可同时跑两个实例） |
| `VR_TELEOP_DS` | 1 | 0 = 不采集 |
| `VR_TELEOP_DS_ROOT` | `data/lerobot` | 数据集落点 |
| `VR_TELEOP_DS_TASK` | 空 | 跳过启动提问，直接给任务描述 |
| `VR_TELEOP_D2_DROP_FAILED` | 1 | 0 = 失败集也保留 |
| `VR_TELEOP_CAMVIEW` | 1 | 0 = 关相机预览窗（省 GPU） |
| `VR_TELEOP_OBJ_RANDOM` | 1 | 0 = 物块不随机摆放 |
| `VR_TELEOP_LOGDIR` | `logs/` | 运行日志目录 |
