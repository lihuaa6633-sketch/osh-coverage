# OSH100 Ranger Pro 动态全覆盖研究工作空间

面向船舶超大幅面薄板加工车间和高铁机车蒙皮车间的全覆盖扫描研究。核心算法不依赖 ROS；ROS 2 包连接 Airy 三维激光雷达、SLAM 和 Woosh OSH100 Ranger Pro 全向底盘。

## 当前实现状态

| 模块 | 状态 | 说明 |
|---|---|---|
| 栅格地图、障碍膨胀、连通域、A* | 已实现并测试 | 纯 NumPy |
| 栅格 BCD 式分区、双方向扫掠 | 已实现并测试 | 当前候选方向为世界坐标 X/Y |
| 全向固定航向横移与传统掉头对照 | 已实现并测试 | 几何安全由规划器保证 |
| 实际轨迹覆盖、残余区和补扫 | 已实现并测试 | 覆盖率不按计划路径虚报 |
| SE(2) 双地图 RANSAC 配准 | 已实现并测试 | Airy 地图与 Woosh 地图对齐 |
| Masked DQN/Double DQN | 已实现并测试 | 只在几何可行候选中决策 |
| 多帧球心拟合 | 已整理并验证现有可执行程序 | PCL/C++，用于单球质量控制 |
| 多球靶 6-DoF 点云绝对定位 | 已整理并测试 | NumPy/Python，4 项测试通过 |
| ROS 2 规划、监测和补扫节点 | Ubuntu 22.04 / Humble 构建测试通过 | 两包共 14 项 colcon 测试通过 |
| Woosh MoveBase/PoseSpeed 桥 | 已按 2026 接口文档核对 | 默认 dry-run；纯协议测试通过，专有消息实机仍需验证 |
| Airy/rslidar 配置 | 已按手册和本地 v1.5.20 驱动核对 | 6699/7788/6688，需实机核对 IP/帧 |

## 目录

```text
osh-coverage/
├── .github/workflows/              # 纯 Python 持续集成
├── .vscode/                        # Ubuntu/ROS 开发任务
├── docs/                           # 架构、集成、实验和来源审计
├── scripts/                        # 评测与工作空间准备脚本
├── src/
│   ├── osh_coverage_core/          # 无 ROS 依赖的规划核心
│   │   └── test/                   # colcon 可发现的规划核心测试
│   ├── osh_coverage_ros/           # ROS 2 Humble 适配层
│   └── vendor/                     # 外部导入，Git 忽略
├── tools/
│   ├── multi_frame_sphere_fit/     # 多帧单球拟合
│   └── sphere_target_registration/ # 多球靶点云绝对定位
└── vendor/robosense.repos          # 固定版本的 Airy 驱动来源
```

## Ubuntu 离线测试

```bash
cd ~/ws/osh-coverage
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-dev.txt

PYTHONPATH=src/osh_coverage_core \
  python3 -m unittest discover -s src/osh_coverage_core/test -v
PYTHONPATH=tools/sphere_target_registration/src \
  python3 -m unittest discover -s tools/sphere_target_registration/tests -v
PYTHONPATH=src/osh_coverage_core:src/osh_coverage_ros \
  python3 -m pytest -q src/osh_coverage_ros/test
```

运行几何示例：

```bash
PYTHONPATH=src/osh_coverage_core \
  python3 -c "from osh_coverage_core.cli import demo_main; demo_main(['--scene','plate_shop'])"
PYTHONPATH=src/osh_coverage_core python3 scripts/evaluate_geometry.py
```

训练模型与实验结果应写入仓库外：

```bash
mkdir -p ~/data/osh-coverage/models
PYTHONPATH=src/osh_coverage_core python3 -c \
  "from osh_coverage_core.cli import train_main; train_main(['--episodes','2000','--output','$HOME/data/osh-coverage/models/ddqn_seed7.npz'])"
```

## Ubuntu 22.04 / ROS 2 Humble

首次准备外部依赖：

```bash
cd ~/ws/osh-coverage
./scripts/bootstrap_ubuntu.sh
```

构建与测试：

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --cmake-args \
  -DENABLE_IMU_DATA_PARSE=ON -DENABLE_TRANSFORM=ON
source install/setup.bash
colcon test
colcon test-result --verbose
```

先启动规划与监测，不连接底盘：

```bash
ros2 launch osh_coverage_ros coverage_system.launch.py use_woosh_bridge:=false
```

完成接口审计后，先以 dry-run 启动桥接层：

```bash
ros2 launch osh_coverage_ros coverage_system.launch.py \
  use_woosh_bridge:=true woosh_dry_run:=true
```

只有在专有消息定义、坐标变换、急停、取消和低速短路径全部验证后，才允许显式设置 `woosh_dry_run:=false`。

## 文档入口

- [Ubuntu + VSCode 工作空间](docs/ubuntu_vscode.md)
- [原始材料整理结论](docs/source_inventory.md)
- [厂商版本与已核对接口](docs/vendor_manifest.md)
- [球靶标定与点云绝对定位](docs/calibration_tools.md)
- [系统架构](docs/system_architecture.md)
- [硬件集成](docs/hardware_integration.md)
- [ROS 2 接口审计](docs/ros2_interface_audit.md)
- [实验协议](docs/experiment_protocol.md)

## 安全限制

- Airy 每约 10 帧有一帧可能出现约 32° 的设计性点云缺口；缺口不能直接解释为空闲空间。
- 点云投影必须使用人工测量的实验边界；自动推断边界不得用于无人监督实车运行。
- 规划器不是功能安全系统；原厂避障、急停、围栏和现场监护不可关闭。
- 当前扫掠候选仅为世界坐标 X/Y，非轴对齐车间需要后续扩展主方向估计和任意角度扫掠。
- 实车只允许在干燥、平整、隔离且有人监护的区域，从架空/dry-run 和不高于 0.20 m/s 的短路径开始。
