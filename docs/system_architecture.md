# 系统架构与数据流

```text
Airy UDP点云/IMU
      │ rslidar_sdk
      ▼
/rslidar_points ──► KISS-ICP ──► 激光里程计
      │                                │
      └──────────────► RTAB-Map ◄──────┘ + /rslidar_imu_data
                              │ /rtabmap/cloud_map
                              ▼
                     map_projector_node
                              │ /map (OccupancyGrid, airy_map)
                              ▼
                    coverage_planner_node
                              │ /coverage/path
                              ▼
                         woosh_bridge
               SE(2): airy_map → woosh_map
                              │ /woosh_robot/ros/MoveBase
                              ▼
                         OSH100底盘
                              │ PoseSpeed
               SE(2): woosh_map → airy_map
                              ▼
                    coverage_monitor_node
                              │
                 covered/residual/status
                              │
                 coverage_supervisor_node
                    任务结束且覆盖率不足时
                              └────► /coverage/path 补扫
```

## 坐标系不变量

- 覆盖规划、覆盖掩膜和残余区全部使用 `airy_map`。
- 原厂 `MoveBase` 使用底盘内部的 `woosh_map`。
- `slam_to_woosh_{x,y,yaw}` 表示 `airy_map` 点到 `woosh_map` 点的刚体变换。
- 路径下发使用正变换；`PoseSpeed` 回传使用逆变换。不得只变换其中一条数据链。
- 配准参数来自不少于20组低速、近似同步的配对位姿，并用未参与拟合的轨迹验证。

## 规划安全边界

Masked Double DQN只能选择几何规划器已经生成的分区和正/反扫掠方案。障碍膨胀、车体碰撞检查、A*连接和原厂避障均不受网络输出控制。当网络无有效动作或模型尺寸不匹配时，系统回退到确定性可行选择。

## ROS话题与动作

| 名称 | 类型 | 方向 |
|---|---|---|
| `/map` | `nav_msgs/OccupancyGrid` | 输入 |
| `/coverage/roi` | `geometry_msgs/PolygonStamped` | 可选输入 |
| `/coverage/start_pose` | `geometry_msgs/PoseStamped` | 可选输入 |
| `/coverage/path` | `nav_msgs/Path` | 规划输出 |
| `/coverage/reachable_mask` | `nav_msgs/OccupancyGrid` | 规划输出 |
| `/coverage/actual_pose` | `geometry_msgs/PoseStamped` | 桥接输出 |
| `/coverage/covered_mask` | `nav_msgs/OccupancyGrid` | 监测输出 |
| `/coverage/residual_mask` | `nav_msgs/OccupancyGrid` | 监测输出 |
| `/coverage/supervisor_status` | `std_msgs/String` | 补漏状态 |
| `/woosh_robot/ros/MoveBase` | `woosh_ros_msgs/action/MoveBase` | 底盘执行 |
| `/woosh_robot/robot/PoseSpeed` | `woosh_robot_msgs/msg/PoseSpeed` | 底盘反馈 |
