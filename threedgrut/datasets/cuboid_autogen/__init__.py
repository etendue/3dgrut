# SPDX-License-Identifier: Apache-2.0
"""cuboid_autogen — 离线纯-LiDAR 聚类自动生成车辆 cuboid 轨迹 → NCore V4 shard。

模块边界（与 SDK 隔离，沿用 tracks_loader 的 Mac-可测先例）：
- labels / cluster / track / bev_metric : 纯 numpy，Mac 可 import + 单测。
- lidar_source / v4_writer              : 依赖 NCore SDK，仅 inceptio。
"""
