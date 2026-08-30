# 原始材料整理结论

## 已纳入仓库

| 原目录 | 结论 | 仓库位置 |
|---|---|---|
| `multi_frame_sphere_fit` | 保留源代码、CMake 和说明；排除构建产物、PCD 和结果 | `tools/multi_frame_sphere_fit` |
| `sphere_target_registration` | 功能明确且测试通过，完整纳入 | `tools/sphere_target_registration` |
| `suteng` | 不复制厂商 Git 仓库；记录精确提交并外部导入 | `vendor/robosense.repos` |
| Airy 本地配置 | 与手册和本地驱动改动核对后保留 RSAIRY、6699/7788/6688 | `src/osh_coverage_ros/config/airy_rslidar_config.yaml` |

## 不纳入仓库

- `suteng/src/build`、`install`、`log`：可重建产物。
- `multi_frame_sphere_fit/build`、样例 PCD 和 `sphere_result`：本地测试和实验数据。
- Woosh agent 安装包及专有消息包：受厂商许可约束。
- `input` 中的简历：与项目无关且包含个人信息。
- 厂商 PDF/DOCX 原件：当前只记录版本和技术结论，不公开复制。

## 功能判定

`sphere_target_registration` 是“基于球形控制点的点云刚体配准/绝对定向”：先在 Airy 三维累计点云中拟合同名球心，再求解 `world = R * scan + t`。它不能修复时间不同步、运动畸变或 SLAM 非刚性漂移。
