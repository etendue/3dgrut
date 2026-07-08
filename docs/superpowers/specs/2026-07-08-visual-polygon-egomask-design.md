# 视觉多边形静态 ego mask 设计（b6a9，10 相机）—— P0.3 执行级设计

- 日期：2026-07-08
- 状态：**设计已获批**（大g 2026-07-08 拍板：视觉多边形 / 做 10 台 / 鱼眼黑圈一起盖 / 并集补强 / 替换现有 itar）
- 定位：v5 Phase C 前置项 **P0.3** 的执行级设计。**取代** [`2026-07-08-expand-cameras-campaign-design.md`](2026-07-08-expand-cameras-campaign-design.md) §3 P0.3 原「sseg egocar 派生」路线——生成源改为「视觉多边形手绘」，存储目标（egomask itar）与下游接线（P0.2 Task 1/2）不变。
- 依赖：P0.2 Task 1（`EgomaskAuxReader` + `resolve_ego_valid_mask`，已合 `4a9f2e6`）已能读 egomask itar；本设计产出「干净的 12→10 台 egomask itar」喂给它。

## 1. 背景与决策链

- **ego mask 双层故障**（P0.1 诊断）：itar 里 4/6 相机有真 mask 但从未接进训练（SDK 内嵌 mask 全零）；front_wide/back_rear_wide itar 全黑。
- **可视化新发现**（本会话）：现有 4 台非空 itar mask 盖住了车头/车身，但 **left_wide/right_wide 的后视镜漏网**（黑色镜体未被覆盖）。
- **大g 核心洞察**：ego mask 全程静态不变（自车结构静止、后视镜不折叠）→ **一相机一张静态 mask、每帧复用** 比逐帧 sseg 派生更省、更可控，且能显式盖住 sseg 漏掉的后视镜。
- **方法拍板**：用 Claude 视觉标注多边形 ROI（Claude 已在诊断中用视觉识别出后视镜位置）。

## 2. 目标与范围

为 b6a9 的 **10 台**相机各生成一张静态 ego mask（自车结构 + 鱼眼黑圈），写回 egomask itar。**跳过 2 台纯远景相机**（front_tele_30fov / front_standard_55fov，实测自车完全不入镜）。

12 台自车结构入镜判断（实测首帧目检）：

| 相机 | 自车结构入镜 | 处置 | mask 来源 |
|---|---|---|---|
| camera_front_tele_30fov | 无（纯远景） | **跳过** | —（不写条目） |
| camera_front_standard_55fov | 无（纯远景） | **跳过** | —（不写条目） |
| camera_front_wide_120fov | 右下角极少 | 做 | 纯视觉（现 itar 全黑） |
| camera_front_fisheye | 镜头黑圈为主 | 做 | 纯视觉（黑圈+少量） |
| camera_cross_left_120fov | 下缘防护杆/车把 明显 | 做 | **并集补强** |
| camera_cross_right_120fov | 右下少量 | 做 | **并集补强** |
| camera_left_wide_90fov | 左后视镜 + 车身 大块 | 做 | **并集补强**（补后视镜） |
| camera_right_wide_90fov | 右后视镜 + 车身 大块 | 做 | **并集补强**（补后视镜） |
| camera_rear_left_70fov | 左下/右下白色车体 | 做 | 纯视觉（无 itar 条目） |
| camera_rear_right_70fov | 右下白色车体 + 后视镜 | 做 | 纯视觉（无 itar 条目） |
| camera_back_rear_wide_90fov | 下缘车尾少量 | 做 | 纯视觉（现 itar 全黑） |
| camera_back_rear_fisheye | 上缘车顶大块 + 黑圈 | 做 | 纯视觉（无 itar 条目） |

- **并集补强（4 台）**：cross_left / cross_right / left_wide / right_wide —— 现有 itar mask（`EgomaskAuxReader.read_static_mask` 读得到、像素级贴合车身）**∪** Claude 补的漏项多边形（后视镜等）。
- **纯视觉（6 台）**：front_wide / back_rear_wide（现 itar 全黑）+ front_fisheye / back_rear_fisheye / rear_left / rear_right（无 itar 条目）—— 全部由视觉多边形生成。

## 3. 生成方法：视觉多边形 ROI

- 每相机：Claude 看一张**带坐标网格的参考图**（原图 + 每 100px 网格线 + 边缘刻度）→ 目测自车结构多边形顶点（图像像素坐标）。
- 每个独立结构（车头 / 左镜 / 右镜 / 防护杆 / 鱼眼黑圈）各一个多边形；同相机多多边形**取并集**。
- 鱼眼黑圈用「圆环外区」表达（圆心 + 半径 → 圆外全 True），或多边形近似。
- 栅格化：`PIL.ImageDraw.polygon` 逐多边形填充到 `(1080, 1920)` bool → 并集 → 该相机静态 mask。
- **保守覆盖，宁大勿漏**（顶点适度外扩；ego 区域盖大不影响，漏则污染重建）。

## 4. 标注闭环（迭代）

参考图（网格）→ Claude 给顶点 → 栅格化 + 叠图 → Claude 自检调顶点 → 10 台叠图汇总 → **大g 目检确认（含并集补强的后视镜验证）** → 才写正式 itar。

- 顶点数据落 per-camera 定义（JSON/dict：`{camera_id: [polygon1_pts, polygon2_pts, ...], fisheye_circle: (cx,cy,r)}`），入 git（是标注数据、不是二进制），便于复跑/微调。
- 叠图目检以**最终（`resolve` dilation 后）valid 区域**为准（见 §6 dilation 注记）。

## 5. 存储与接入

**替换现有 egomask itar**（大g 拍板）：新建一个完整 egomask itar，10 台各写静态 mask，2 台跳过（不写条目），整体替换旧的。

- itar 内部结构沿 nre-tools 原样（Task 1 probe 确认）：`aux/egomask/<camera_id>/<timestamp>` = 0-D `|S<n>` PNG bytes。每相机写 **1 帧**静态 mask（timestamp key 用 `"0"`；reader 取并集 = 该单帧）。
- **write-once 纪律**（itar 不可 in-place）：新 itar 先写临时名 → 旧 itar `mv` 到 `aux_backup/` → 临时名改回正名。`discover_aux_path` 同目录多个 `*.aux.egomask.zarr.itar` 会 ValueError，故替换后目录内只保留 1 个正名 itar。
- 跳过的 2 台（front_tele/front_standard）不写条目 → `reader.has_camera` False → `resolve_ego_valid_mask` 走 branch 3 全 True（不 mask，符合「不入镜」）。
- **下游零改动**：P0.2 Task 1 `EgomaskAuxReader` 读新 itar、Task 2 接线读 `resolve_ego_valid_mask`，均不改。
- itar 写模式复用 `scripts/merge_lidar_aux.py` 经验（`create_dataset` + PNG 编码 bytes 写 0-D）。

## 6. dilation 交互注记

`resolve_ego_valid_mask` 对 itar mask 默认再 `binary_dilation(iterations=30)` 后取反。静态多边形 mask 已保守覆盖，叠加 30 次膨胀会额外外扩 buffer：

- 验收以**最终 dilation 后 valid 区域**为准（叠图渲染 `resolve` 输出，而非裸多边形）。
- 若某相机总覆盖过量吃掉有效路面 → 该相机标注收紧（不改全局 `n_camera_mask_dilation_iterations`，保持 PAI 线字节等价）。

## 7. 验收标准

1. **叠图目检（主）**：每台自车结构（含后视镜/防护杆/车体/鱼眼黑圈）被覆盖；无大块误盖有效路面；4 台并集补强的后视镜确被补上（大g 视觉验证）。
2. **itar 回读**：`diag_egomask_itar.py` 复扫 → 10 台 nonzero>0、2 台无条目；`EgomaskAuxReader` 读新 itar，10 台 `has_camera` True。
3. **字节等价不受影响**：PAI 线（无 egomask itar）行为不变（本设计只动 b6a9 clip 目录的数据文件，不改代码判定路径）。

## 8. 测试策略

- **轻量单测**（Mac，纯逻辑）：多边形/圆环栅格化正确性（给定顶点/圆参数 → mask 像素精确）；itar 写 → `EgomaskAuxReader` 读回一致（round-trip，含并集补强合成）。
- **视觉质量不可单测** → 靠 §7 目检。
- 派生脚本纯 numpy/PIL/scipy + itar 读写，Mac 可跑逻辑单测；itar 实写与叠图在 inceptio（原图/现有 itar 在那）。

## 9. 与 Task/plan 关系

- 本设计 = P0.3 执行级；**取代** expand-cameras spec §3 P0.3 的 sseg 派生方法（存储/下游不变）。
- 产物（新 egomask itar）→ P0.2 Task 2 接线后训练生效 → P0.4 R4e 重锚用它。
- Phase C 阶梯扩相机（C2 rear_left/front_standard 等）时，新参训相机的 mask 沿同一视觉多边形流程补（本 spec 已覆盖 rear_left；front_standard 判定不入镜、不需要）。

## 10. 出界项（本设计不做）

- 逐帧动态 mask（本设计明确只做静态）；sseg egocar 自动派生（被视觉多边形取代）；其它 clip 的 ego mask（b6a9 专项）；Task 2 datasetNcore 接线本身（独立任务）。
