# Woosh与Airy ROS 2接口审计

在写入实车参数前保存全部输出和时间戳，形成论文“平台接口标定”原始记录。

## 1. 软件版本

```bash
uname -a
printenv ROS_DISTRO
ros2 pkg prefix woosh_robot_msgs
ros2 pkg prefix woosh_ros_msgs
ros2 pkg prefix rslidar_sdk
ros2 doctor --report
```

## 2. 消息真实定义

PDF可能与已安装版本不同，以下输出才是代码适配依据：

```bash
ros2 interface show woosh_robot_msgs/msg/PoseSpeed --all-comments
ros2 interface show woosh_robot_msgs/msg/RobotState --all-comments
ros2 interface show woosh_robot_msgs/msg/OperationState --all-comments
ros2 interface show woosh_robot_msgs/msg/AbnormalCodes --all-comments
ros2 interface show woosh_ros_msgs/action/MoveBase --all-comments
ros2 interface show woosh_ros_msgs/action/StepControl --all-comments
```

若字段与 `woosh_bridge_node.py` 不一致，先修改桥接层并增加录包回放测试，不修改算法核心。

## 3. 频率、QoS和时延

```bash
ros2 topic hz /woosh_robot/robot/PoseSpeed
ros2 topic info -v /woosh_robot/robot/PoseSpeed
ros2 topic hz /rslidar_points
ros2 topic hz /rslidar_imu_data
ros2 action info /woosh_robot/ros/MoveBase
```

记录静止、0.2 m/s直行、0.2 m/s横移和原地旋转数据。若 `PoseSpeed` 无消息时间戳，在桥接接收时打时间戳，并把网络时延作为地图配准误差来源。

## 4. 坐标约定

分别发送很短的正X、正Y和正角速度命令，核对：

- `PoseSpeed.pose.x/y/theta` 正方向
- `MoveBase` 航点的角度单位和范围
- `linear_y` 或横移步骤的正方向
- Airy点云的X/Y/Z与 `base_link` 的静态变换
- KISS-ICP和RTAB-Map实际发布的里程计与地图话题名称

## 5. 动作生命周期

用两个航点验证接受、反馈、成功、失败、暂停、继续、取消和急停。再依次尝试50、100、200个航点，确定底盘可稳定接受的分块大小；默认配置200只是起始值，不是实测结论。

