# 贡献与实验变更规则

## 分支

从最新 `main` 创建短生命周期分支：`feat/...`、`fix/...`、`test/...`、`docs/...` 或 `chore/...`。不要在 `main` 上进行实车临时修改。

## 提交前验证

```bash
PYTHONPATH=src/osh_coverage_core \
  python3 -m unittest discover -s src/osh_coverage_core/test -v
PYTHONPATH=tools/sphere_target_registration/src \
  python3 -m unittest discover -s tools/sphere_target_registration/tests -v
PYTHONPATH=src/osh_coverage_core:src/osh_coverage_ros \
  python3 -m pytest -q src/osh_coverage_ros/test
python3 -m compileall -q src scripts tools/sphere_target_registration/src
```

安装 ROS 2 Humble 后还必须运行：

```bash
colcon build --symlink-install --cmake-args \
  -DENABLE_IMU_DATA_PARSE=ON -DENABLE_TRANSFORM=ON
colcon test
colcon test-result --verbose
```

## 数据与厂商软件

- 不提交 `build/`、`install/`、`log/`、rosbag、PCD/PLY、模型权重或实验结果。
- RoboSense 源码通过 `vendor/robosense.repos` 获取；不要把其本地 Git 仓库复制进本仓库。
- Woosh agent 和专有消息包只在获得授权的主机安装，不进入 Git 历史。
- 小型、脱敏、可再分发的回归样本才可放入测试目录。

## 实车 Pull Request

涉及 Airy 或 Woosh 的 PR 必须记录：Ubuntu/ROS 版本、硬件/固件和文档版本、实际话题与消息定义、验证数据、最高测试速度、场地和急停监护条件。
