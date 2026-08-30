# OSH100 Ranger Pro 动态全覆盖研究工作空间

这是面向硕士论文的可复现实验代码库。核心算法不依赖ROS，可在普通Python环境中完成地图、几何规划、动态补漏、地图配准和Masked DQN/Double DQN实验；ROS 2包负责连接Airy、RTAB-Map和悟时底盘。

## 当前实现状态

| 模块 | 状态 | 说明 |
|---|---|---|
| 栅格地图、障碍膨胀、连通域、A* | 已实现并测试 | 纯NumPy |
| 栅格BCD式分区、双方向扫掠 | 已实现并测试 | 支持水平/垂直候选方向 |
| 全向固定航向横移与传统掉头对照 | 已实现并测试 | 几何安全由规划器保证 |
| 实际轨迹覆盖掩膜、残余区、两次重试 | 已实现并测试 | 不按计划路径虚报覆盖 |
| SE(2)双地图RANSAC配准 | 已实现并测试 | 提供CSV命令行工具 |
| Masked DQN/Double DQN、课程训练 | 已实现并测试 | 纯NumPy，离线训练 |
| ROS 2地图、路径、覆盖掩膜和补漏监督节点 | 已实现，待Ubuntu验证 | 当前电脑没有ROS 2 |
| Woosh MoveBase/PoseSpeed桥 | 已实现，待专有消息包验证 | 含双向坐标变换和路径分块 |
| Airy/KISS-ICP/RTAB-Map配置 | 已提供模板 | 需按实机话题核对 |
| 实车安全、定位精度和动态障碍实验 | 未执行 | 必须在隔离实验区完成 |

## 目录

```text
osh-coverage/
├── src/osh_coverage_core/   # 无ROS依赖的算法包
├── src/osh_coverage_ros/    # ROS 2 Humble节点与配置
├── scripts/                 # 几何、RL评测脚本
├── tests/                   # 标准库unittest
└── docs/                    # 上机、实验和论文执行说明
```

## 当前电脑离线验证

将核心包加入 `PYTHONPATH` 后运行：

```powershell
$env:PYTHONPATH='D:\Research\osh-coverage\src\osh_coverage_core'
python -m unittest discover -s 'D:\Research\osh-coverage\tests' -v
python -c "from osh_coverage_core.cli import demo_main; demo_main(['--scene','plate_shop'])"
python 'D:\Research\osh-coverage\scripts\evaluate_geometry.py'
```

训练模型时显式把结果放到D盘成果目录：

```powershell
python -c "from osh_coverage_core.cli import train_main; train_main(['--episodes','2000','--output','D:\codex_data\artifacts\osh_coverage\ddqn_seed7.npz'])"
python 'D:\Research\osh-coverage\scripts\evaluate_rl.py' 'D:\codex_data\artifacts\osh_coverage\ddqn_seed7.npz' --problems 100
```

## Ubuntu 22.04 / ROS 2 Humble构建

1. 安装ROS 2 Humble、NumPy以及悟时提供的 `woosh_*_msgs` 和agent。
2. 把 `rslidar_sdk`、`rslidar_msg`、KISS-ICP、RTAB-Map依赖放入同一工作空间或安装到系统。
3. Airy驱动编译时开启IMU解析和坐标变换选项。

```bash
cd ~/osh-coverage
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --cmake-args -DENABLE_IMU_DATA_PARSE=ON -DENABLE_TRANSFORM=ON
source install/setup.bash
```

先只启动规划与监测，不连接底盘：

```bash
ros2 launch osh_coverage_ros coverage_system.launch.py use_woosh_bridge:=false
```

完成接口审计、地图配准和低速安全测试后再启用：

```bash
ros2 launch osh_coverage_ros coverage_system.launch.py use_woosh_bridge:=true
```

详细顺序见 [硬件集成](docs/hardware_integration.md)、[ROS接口审计](docs/ros2_interface_audit.md)、[实验协议](docs/experiment_protocol.md)。

## 重要限制

- 点云投影节点在指定矩形实验边界内把未命中单元视为空闲；实车前必须输入经过测量的边界，不能直接把自动推断边界用于无人监督运行。
- 当前几何分解针对车间常见直角栅格，候选扫掠方向为世界坐标X/Y；任意角度旋转分解属于后续扩展。
- ROS节点已经过Python语法检查，但未在当前Windows环境编译，也未连接专有Woosh消息包。
- 底盘IP21，只允许在干燥、平整、无油水粉尘、隔离且有人监护的实验区测试。
