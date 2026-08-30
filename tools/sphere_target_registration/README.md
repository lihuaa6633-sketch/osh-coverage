# 三球靶点云绝对定位与坐标修正

## 这个过程叫什么

比较准确的名称是：

- **基于球形控制点的点云刚体配准**（sphere-target landmark registration）；
- 若指定球心属于厂区、测量或世界坐标系，也称 **点云绝对定向** 或 **点云地理配准**（absolute orientation / georeferencing）；
- 数学上是一个 **无尺度的三维 Helmert 变换**，即 6 自由度刚体变换。

它通常不叫“雷达内参标定”。只有在用这些球靶反求雷达相对机器人、相机或其他传感器的安装位姿时，才属于外参标定。

设扫描点云中拟合出的球心为 `c_i`，人为指定的同名球心为 `C_i`，程序求解：

```text
C_i = R * c_i + t
```

其中 `R` 是旋转矩阵，`t` 是平移向量。随后同一个变换应用到环境点云中的每个点。

## 适用前提

1. 输入必须是真正的三维点云，三个球靶与环境点必须处于同一个累计扫描坐标系。
2. 三个球心不能共线。建议把球靶布成面积尽可能大的三角形，不要集中在一个小角落。
3. 球靶在扫描期间不能移动，`T1/T2/T3` 与指定坐标必须一一对应。
4. 点云单位、指定坐标和球半径单位必须相同，建议全部使用米。
5. 移动扫描必须先做时间同步、运动补偿和 SLAM 累计。该工具修正整体刚体偏差，不能消除点云内部的运动畸变或 SLAM 非刚性漂移。

当前相邻目录中的 `scanner_bridge.py` 发布的是二维 `sensor_msgs/LaserScan`。固定高度的二维扫描只能得到球的截圆，不能稳定拟合三维球心。应使用 Airy 输出的三维点云（通常为 `sensor_msgs/PointCloud2`），或先利用可信位姿对多层扫描去畸变并累计成三维点云。

三个球是理论最小配置；工程上推荐放置 4 个以上。包支持任意 `N >= 3`，多余控制点可以形成冗余并让残差更有诊断意义。

## 安装

在装有 Python 3.9+ 的电脑或 ROS 2 Ubuntu 主机上执行：

```bash
cd sphere_target_registration
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

Windows PowerShell 激活命令为：

```powershell
.venv\Scripts\Activate.ps1
```

唯一运行依赖是 NumPy。

## 输入数据

支持以下点云格式：

- ASCII PCD；
- ASCII PLY；
- XYZ/TXT/PTS 空格分隔数值表；
- CSV，可带表头；
- NPY 二维数值数组。

PCD/PLY/CSV 中的额外数值列会原样保留，只修改 `x/y/z`。二进制 PCD/PLY 可先用 CloudCompare 或 PCL 转成 ASCII；大规模生产点云建议在外部用 PCL/Open3D 读取，再直接调用本包的 Python API，以避免 ASCII 文件体积过大。

每个球靶有两种取点方式：

- `points`：提前在 CloudCompare/PCL 中把球靶粗裁成一个单独文件；
- `roi`：在未修正的扫描坐标系中给出粗略轴对齐包围盒，由程序从环境点云裁取。

ROI 可以包含少量支架或背景，RANSAC 会排除离群点，但框不应大到让球面点只占很小比例。两种方式可在同一配置中混用。

## 配置与运行

复制并修改 [examples/config.example.json](examples/config.example.json)。所有相对路径都相对于配置文件所在目录。

关键参数：

- `expected_radius`：球的真实半径，不是直径；
- `radius_tolerance`：允许的半径偏差；
- `distance_threshold`：点到拟合球面的内点距离，一般取单点测距噪声的 2 到 4 倍；
- `maximum_pair_distance_error`：扫描球心间距与指定球心间距允许的最大差异，用于发现靶号错误、球移动和严重点云形变；
- `min_inlier_ratio`：粗裁点中至少有多少比例属于球面。

执行完整流程：

```bash
sphere-target-register run examples/config.example.json
```

先单独检查某个球的拟合质量：

```bash
sphere-target-register fit data/target_1.pcd \
  --radius 0.0725 \
  --radius-tolerance 0.006 \
  --distance-threshold 0.003
```

也可以不安装，直接运行：

```bash
PYTHONPATH=src python3 -m sphere_target_registration run examples/config.example.json
```

## 输出与判读

程序生成：

- 修正后的完整环境点云；
- JSON 报告，包括每个球的拟合球心、半径、内点率、球面 RMSE；
- 扫描坐标系到指定坐标系的旋转、平移、四元数和 4x4 齐次矩阵；
- 球心配准残差及所有球心对的距离差。

报告中的变换方向固定为：

```text
registered_xyz = rotation @ scanned_xyz + translation
```

对于 ROS TF，这个矩阵正好表示“父坐标系为指定/世界坐标系、子坐标系为原扫描坐标系”时的变换。报告的 `quaternion_xyzw` 顺序与 ROS 一致。发布前仍应根据自己的 TF 树确认没有同时发布另一条冲突的父子关系。

结果验收至少检查：

- 拟合半径接近真实半径；
- 单球 `rmse` 与雷达噪声水平一致；
- `inlier_ratio` 不过低；
- `center_errors` 和 `pair_distance_errors` 小于项目允许的定位误差；
- 修正后在 CloudCompare/RViz 中复核全部球心，而不是只看三个控制点附近。

## Python API

```python
from sphere_target_registration import fit_sphere_ransac, estimate_rigid_transform

fit = fit_sphere_ransac(
    target_points_xyz,
    expected_radius=0.0725,
    radius_tolerance=0.006,
    distance_threshold=0.003,
)

transform = estimate_rigid_transform(
    measured_centers_xyz,
    reference_centers_xyz,
    maximum_pair_distance_error=0.015,
)
registered_points = transform.apply(environment_points_xyz)
```

## 测试

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

测试数据包含球面遮挡、测距噪声和随机离群点，并验证从输入文件到修正点云及报告的完整流程。
