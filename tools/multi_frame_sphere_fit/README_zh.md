# 多帧 PCD 合并与 14 cm 靶标球球心拟合

这套程序把原来的单帧 `sphere_ransac.cpp` 改成以下完整流程：

1. 按文件名顺序读取目录中的多帧 PCD。
2. 删除每帧的 NaN 点。
3. 如雷达发生移动，用每帧位姿把点变换到统一坐标系。
4. 在统一坐标系内逐帧截取球附近的紧 ROI。
5. 合并所有 ROI 点。
6. 用中值体素去重并抑制多帧重复测量噪声。
7. 用宽松球面 RANSAC 找到初始球心。
8. 固定真实半径为 0.070 m，用 Huber 鲁棒损失联合精化球心。
9. 用严格阈值重新划分内点，检查各帧支持数、残差、点面覆盖和前后半段球心漂移。

## 一、最重要的前提：坐标系必须统一

| 采集情况 | 能否直接拼接 | 程序用法 |
|---|---:|---|
| 雷达不动、球不动，每帧都是雷达坐标系 | 可以 | 不传 `--poses` |
| 雷达移动，但 SDK 已把每帧输出到同一个 map/odom 坐标系 | 可以 | 不传 `--poses` |
| 雷达移动，每帧仍是各自的雷达坐标系 | 不可以直接相加 | 提供 `--poses poses.txt` |
| 球在采集期间移动 | 不适合拼接 | 重新采集静止数据或逐帧拟合 |

不要只拿球面做 ICP 配准。球具有旋转对称性，而且可见点通常只是局部球冠，配准容易漂移。移动雷达时，优先使用标定后的雷达位姿；如果必须 ICP，应使用球外的稳定静态场景来求帧间变换，再把相同变换应用到球点。

## 二、准备数据

把本次采集的原始帧单独放在一个目录中，例如：

```text
frames/
├── frame_0001.pcd
├── frame_0002.pcd
├── frame_0003.pcd
└── ...
```

建议固定雷达和球，采集 20～50 帧。输入坐标单位必须是米。目录中只放需要参与计算的原始帧，不要混入以前生成的 `merged_*.pcd` 或 `sphere_*.pcd`。

先在 CloudCompare 或 PCL Viewer 中估计球的大致中心。14 cm 球半径是 0.07 m，推荐 ROI 半径先用 0.12～0.18 m；只要能完整包住球即可，不要继续用 0.5 m 的大范围，否则球点占比太低，RANSAC 很难抽中四个真实球面点。

## 三、编译

Ubuntu 20.04 可先确认已安装 PCL 开发包和 CMake，然后执行：

```bash
cd multi_frame_sphere_fit
mkdir build
cd build
cmake ..
make -j4
cd ..
```

生成的程序为：

```text
build/multi_frame_sphere_fit
```

## 四、雷达和球都不动时的推荐命令

下面以粗略球心 `(0.86, -0.28, 0.45)`、ROI 半径 `0.15 m` 为例：

```bash
./build/multi_frame_sphere_fit frames \
  --roi 0.86 -0.28 0.45 0.15 \
  --voxel 0.004 \
  --ransac-threshold 0.008 \
  --final-threshold 0.004 \
  --radius-min 0.065 \
  --radius-max 0.075 \
  --fixed-radius 0.070 \
  --output sphere_result
```

参数分为两层：

- 粗检测：半径 65～75 mm、球面距离阈值 ±8 mm，目的是先稳定找到正确的球。
- 最终结果：半径固定为 70 mm，球面内点阈值恢复为 ±4 mm，与原程序的最终严格判据一致。

如果点云质量很好，也可以把粗检测改回原程序的严格参数：

```bash
--ransac-threshold 0.004 --radius-min 0.068 --radius-max 0.072
```

如果最终内点明显过少，而传感器单点距离噪声确实超过 4 mm，可单独把 `--final-threshold` 调到 `0.006`；不要因为想增加内点就同时无限放宽半径范围。

## 五、雷达移动时的位姿文件

`poses.txt` 每行格式为：

```text
frame_0001.pcd tx ty tz qx qy qz qw
frame_0002.pcd tx ty tz qx qy qz qw
```

程序采用：

```text
p_common = T_common_lidar × p_lidar
```

其中平移单位为米，四元数顺序为 `qx qy qz qw`。每一个 PCD 都必须有对应位姿，文件名按 basename 匹配。运行时增加：

```bash
--poses poses.txt
```

此时 `--roi` 给出的粗球心以及最终输出球心都属于 common/map 坐标系。

## 六、输出文件

| 文件 | 含义 |
|---|---|
| `merged_roi_raw.pcd` | 对齐、裁剪后直接合并的全部点 |
| `merged_voxel_median.pcd` | 4 mm 中值体素去重后的拟合点 |
| `sphere_inliers.pcd` | 固定 70 mm 半径且最终径向残差不超过阈值的原始合并点 |
| `sphere_outliers.pcd` | ROI 中其余点 |
| `sphere_fit_colored.pcd` | 绿色为内点，灰色为外点，红色十字为球心 |
| `sphere_center.pcd` | 单独保存的最终球心点 |
| `sphere_report.txt` | 球心、粗拟合半径、误差、各帧支持数和稳定性检查 |

## 七、怎样判断结果可靠

至少同时检查以下项目：

1. `sphere_fit_colored.pcd` 中绿色点必须落在真实靶标球上，不能主要落在支架、墙面或其他圆弧上。
2. `coarse_radius_m` 应接近 0.070 m，不能长期贴着 0.065 或 0.075 的边界。
3. `frames_supporting` 不应只有少数帧。采集 20～50 帧时，建议至少 60% 的帧都有球面内点。
4. `rmse_m` 和 `p95_absolute_residual_m` 越小越好；最终阈值为 4 mm 时，通常希望 RMSE 在 2～3 mm 以内。
5. `half_center_delta_m` 是前半段与后半段分别拟合的球心差。静止采集时建议小于 0.002～0.003 m；若明显更大，应优先检查雷达晃动、球移动、温漂或坐标变换误差。
6. `direction_min_max_ratio` 太小，说明点主要集中在一条扫描线或很窄的球冠上。程序在小于 0.01 时会警告；增加完全重复的帧只能降噪，不能补足缺失的观察方向。

程序默认把“至少 60% 的帧支持结果”作为质量门槛，也可以用 `--min-frames n` 手动覆盖。若拟合结果文件已经生成，但程序返回码为 `2`，表示计算完成但内点数或支持帧数未通过质量门槛，应查看报告和彩色点云，而不是直接采用球心。

## 八、与原单帧程序相比，核心改动

原流程：

```text
单个 PCD → 删除 NaN → 球面 RANSAC → 输出球心
```

新流程：

```text
多帧 PCD
→ 统一坐标系
→ 每帧紧 ROI
→ 合并
→ 中值体素去重
→ 宽松球面 RANSAC 定位
→ 固定 R=0.070 m 鲁棒联合精化球心
→ 严格 ±4 mm 内点复核
→ 多帧支持与前后半段稳定性验收
```

如果原始 PCD 还能保存 `ring/channel`、方位角和时间戳，下一步可以按“线号 + 方位角”对重复射线取距离中值；这会比只用 XYZ 进行空间体素中值更准确。本程序保持与原 `pcl::PointXYZ` 数据兼容，因此只使用 XYZ。
