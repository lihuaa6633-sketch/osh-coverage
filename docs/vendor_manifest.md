# 厂商依赖与文档清单

| 项目 | 本次审计版本 | 使用方式 | 许可/限制 |
|---|---|---|---|
| RoboSense `rslidar_sdk` | `v1.5.20`, `8b4b4b7ff910799260347821084c59e1c73d50d5` | `vendor/robosense.repos` 外部导入 | BSD-3-Clause |
| RoboSense `rslidar_msg` | `v1.5.9`, `fe8a95cb242bd294cc3d5e3422f2093fb49a56ee` | `vendor/robosense.repos` 外部导入 | BSD-3-Clause |
| Airy 产品手册 | 本地 PDF；手册包含 96 线 Airy、ROS/ROS2 与协议说明 | 仅提取配置和接口约束 | 厂商文档，不复制到仓库 |
| Woosh ROS 2 接口文档 | 2026，50 页 | 桥接层接口依据，实机以 `ros2 interface show` 为准 | 厂商文档，不复制到仓库 |
| OSH100 Ranger Pro 用户指南 | 本地 DOCX | 平台能力、盲区和安全边界依据 | 厂商文档，不复制到仓库 |
| Woosh robot agent/messages | Humble/Jazzy/Foxy 对应厂商安装包 | 主机本地安装 | 专有软件，不提交 |

## 已核对的 Airy 参数

- 型号：`RSAIRY`，96 线，10 Hz，水平 360°，垂直 0–90°。
- 默认设备/主机地址：`192.168.1.200` / `192.168.1.102/24`。
- MSOP/DIFOP/IMU：UDP `6699` / `7788` / `6688`。
- 点云存在约每 10 帧一帧、约 32° 的设计性缺口；不能把缺口直接解释为空闲空间。
- ROS 2 使用 `rslidar_sdk` + `rslidar_msg`；Airy 的 IMU 解析需要编译选项 `ENABLE_IMU_DATA_PARSE=ON`。

## 已核对的 Woosh 参数

- agent 默认底盘 IP：`169.254.128.2`，建议命名空间 `/woosh_robot`。
- 位姿：`/woosh_robot/robot/PoseSpeed`，角度为弧度。
- 导航 action：`/woosh_robot/ros/MoveBase`；逐点模式 `K_ONE_BY_ONE=1`，执行 `K_EXECUTE=1`。
- action 结果/反馈使用 `woosh_ros_msgs/Feedback`；只有 `state.value == K_ROS_SUCCESS (1)` 才能发送下一路径分块。

## 已核对的 OSH100 平台约束

- 尺寸约 `699 x 503 x 307 mm`，自重约 90 kg，平台额定负载 100 kg。
- 底盘额定电池电压 48 V；扩展电源口输出 48–54.7 V，不能直接给 Airy 供电。
- 防护等级 IP21，只用于室内；环境 5–40 ℃、10–95% RH、无冷凝和腐蚀/爆炸性气体。
- 原二维雷达扫描面离地约 180 mm，无法探测扫描面上方或下方的物体；Airy 替换方案必须验证三维盲区而不能假定问题自动消失。
- 手册额定最大速度 1.5 m/s，但本项目首次实车验证仍限制在不高于 0.20 m/s。
