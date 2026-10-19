# 第三方资源与许可

本仓库的 **代码与文档** 为 MIT（见 LICENSE）。`assets/` 下的机器人模型不是本项目的原创，
它们的许可如下。**分发本仓库时请保留本文件。**

## 1. Kinova Gen3 机械臂模型

```
assets/kinova_gen3_robotiq/gen3_robotiq_2f85.xml
assets/kinova_gen3_robotiq/gen3_robotiq_2f85.urdf
assets/kinova_gen3_robotiq/assets/   (MJCF 用的 STL)
assets/kinova_gen3_robotiq/meshes/   (URDF 用的 STL)
```

* 来源：Kinova 官方 `kortex_description`（机械臂 DH/连杆/网格）
  与 MuJoCo Menagerie 的 `kinova_gen3` 模型（MJCF 描述与网格）。
* 许可：MuJoCo Menagerie 采用 **BSD-3-Clause**；Kinova 的 `kortex_description`
  以其随包许可为准（通常为 BSD/Apache 类）。请以各自上游仓库的 LICENSE 为准。
* 本仓库只在**路径与 include** 上做了本地化改动（相对路径），未修改动力学参数。

## 2. Robotiq 2F-85 夹爪模型

```
assets/kinova_gen3_robotiq/*2f85*   中的夹爪部分
```

* 来源：ros-industrial-attic / robotiq 的
  `robotiq_2f_85_gripper_visualization`（DAE → STL），以及
  MuJoCo Menagerie 的 `robotiq_2f85`（四连杆 + tendon 的 MJCF 实现）。
* 许可：随上游（BSD 类）。四连杆机构的 MJCF 实现来自 Menagerie。

## 3. 场景（工作台 / 方块 / 相机）

```
assets/scene_grasp.xml
```

* 本项目原创（几何体 + 相机 + 光照参数），MIT。

## 4. 依赖库

`mujoco`（Apache-2.0 / 商业双许可）、`numpy`（BSD）、`aiohttp`（Apache-2.0）、
`pillow`（HPND）、`pyarrow`（Apache-2.0）、`ffmpeg`（LGPL/GPL，视编译选项）。
它们不被本仓库打包，由用户自行安装。

## 5. 如果你要再分发

* 保留本文件与各上游 LICENSE。
* 若你替换成了自己的机器人模型，把上面第 1/2 节删掉，写上你的来源与许可。
* 不要把厂商的私有网格/标定文件上传到公开仓库（这一点在商业项目里尤其重要）。
