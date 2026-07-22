# Road ownership 修复计划（从 multi-camera 清晰度任务剥离）

## 目标与范围

本任务只解决多层模型中 background 侵占 road 成像责任的问题；6cam 清晰度、训练预算与跨相机一致性另立任务。基线为冻结的 6cam checkpoint，R0 产物位于 `/home/inceptio/work/output/mcro_r0_ownership/`。

冻结门槛以 `configs/eval/mcro_ownership_guards.json` 为准：background-on-road alpha 至少下降 50%，road alpha P10 不低于 0.37，sky-on-road 不高于 0.001，front-wide full/road-crop KPI 不回退。任何 GPU 训练启动前请求用户确认；代码、本地测试和只读诊断自动推进。

## 已保留的前置结果

- [x] checkpoint 分层统计、front-wide crop/radial 报告、per-camera telemetry。
- [x] 通用 layer filter、alpha/road-mask/sky-contrib 导出与 ownership evaluator。
- [x] R0 1cam/6cam JSON、四联图、lateral 3m/6m 样例与冻结 guards。
- [x] 基线 checkpoint 路径、hash、帧集合和评测口径文档。

## 修复阶段

- [x] B1 semantic-disjoint init：road/sidewalk LiDAR 不再进入 background init；缺标签点保留并记录统计。
- [x] B2 hard background exclusion：训练期对 road slab/投影命中的 background 做 density 回收或 footprint 收缩；保护高于路面的物体。
- [x] B3 road responsibility：定期 road-only forward，在 eroded road mask 内监督 alpha/RGB；position/rotation/scale 断梯度，只允许 density/appearance 补洞；支持 LiDAR RGB 与更高初始 opacity 的条件 ablation。
- [ ] B5 5s 单变量实验：先跑同预算 R0-5s 画质锚，再跑 R1=B1，R2=R1+B2，R3=R2+B3；若 R3 alpha 仍不足，再依次试 R3O（初始 opacity）和 R3C（LiDAR RGB）。每臂单独确认；画质对 R0-5s，ownership 绝对门对冻结 20s R0。
- [ ] B4 条件 sky gate：仅当新 arm 的 `sky_on_road_energy > 0.001` 时实现/启用；R0 为 0，不默认引入。
- [ ] B6 正式 20s/30k：只有 5s arm 通过定量门且视觉检查无车辆/路缘/标牌损失，才请求确认启动。

## 自适应判因

- R1 未降低 background alpha：检查语义 init 统计与训练后 background 回占速度，不直接叠加后续机制。
- R2 background 达标但 road P10 下降：B2 有效而 road 接不住，进入 R3；若 full KPI 下降，降低 B2 频率/收缩强度而不放宽 guards。
- R3 road alpha 仍不足：先用 loss telemetry 判断梯度与开销，再单变量试 R3O、R3C；禁止放宽 road scale 上限补洞。
- ownership 达标但 full/road KPI 失败：调整责任 loss 权重/频率，保持 B1/B2 归属约束；不以把责任还给 background 的方式过画质门。
- sky 超门才进入 B4；无证据不增加训练/推理 mismatch。
