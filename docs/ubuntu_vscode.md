# Ubuntu 22.04 + VSCode 工作空间

## 推荐布局

```text
~/ws/osh-coverage/                 # 本仓库和 colcon 工作空间根目录
├── src/
│   ├── osh_coverage_core/
│   ├── osh_coverage_ros/
│   └── vendor/                    # bootstrap 脚本导入，Git 忽略
├── tools/
├── build/                         # Git 忽略
├── install/                       # Git 忽略
└── log/                           # Git 忽略

~/data/osh-coverage/
├── bags/
├── pointclouds/
├── models/
└── results/
```

用 VSCode 打开仓库根目录。仓库已提供扩展建议、Python 路径、单元测试和 ROS 构建任务。

## 首次准备

安装 Ubuntu 22.04、ROS 2 Humble desktop、`python3-vcstool`、`python3-rosdep`、`python3-colcon-common-extensions`、PCL 和 Woosh Humble agent/messages。随后执行：

```bash
cd ~/ws/osh-coverage
./scripts/bootstrap_ubuntu.sh
```

## 构建与测试

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --cmake-args \
  -DENABLE_IMU_DATA_PARSE=ON -DENABLE_TRANSFORM=ON
source install/setup.bash
colcon test
colcon test-result --verbose
```

每个新终端都需要 source ROS 和工作空间；不要把 `source install/setup.bash` 写进全局 shell 配置，避免多个工作空间互相污染。
