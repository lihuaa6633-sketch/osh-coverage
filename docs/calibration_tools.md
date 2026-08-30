# Airy 球靶标定与点云绝对定位

## 工具边界

1. `multi_frame_sphere_fit`：输入一个球靶的多帧 PCD，输出稳定球心和质量指标。
2. `sphere_target_registration`：输入三个或更多已对应球心及其世界坐标，输出完整点云的 6-DoF 变换。

球靶采集期间必须静止。若雷达移动，每帧必须先通过可信位姿去畸变并变换到同一 `map` 坐标系。只对球面做 ICP 不可观，因为球体具有旋转对称性。

## 多帧单球拟合

```bash
cmake -S tools/multi_frame_sphere_fit \
  -B tools/multi_frame_sphere_fit/build
cmake --build tools/multi_frame_sphere_fit/build -j

tools/multi_frame_sphere_fit/build/multi_frame_sphere_fit frames \
  --roi 0.86 -0.28 0.45 0.15 \
  --fixed-radius 0.070 \
  --output ~/data/osh-coverage/results/sphere_01
```

不要提交 `frames` 或结果 PCD。验收拟合半径、RMSE、内点数、支持帧比例、方向覆盖和前后半段球心漂移。

## 多球世界坐标配准

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e tools/sphere_target_registration
sphere-target-register run path/to/config.json
```

至少三个球心不共线；工程上推荐四个以上。报告的方向固定为：

```text
registered_xyz = rotation @ scanned_xyz + translation
```

输出变换发布为 ROS TF 前，应确认现有 TF 树中没有重复父节点，并用未参与拟合的检查点验证。
