# SPDX-License-Identifier: Apache-2.0
"""T3.3.a unit tests for road_init.init_road_layer.

Pure-CPU mock tests: feed synthetic flat-Z road LiDAR + ego trajectory and
verify the BEV-grid + KNN-Z init returns parameters consistent with
LayerSpec(road).scale_prior=(0.1, 0.1, 0.001).

T3.3.b lands the implementation; these tests are the behavioral contract.
"""

from __future__ import annotations

import math

import torch
from omegaconf import OmegaConf

from threedgrut.layers.layer_spec import LayerSpec
from threedgrut.layers.registry import specs_from_config
from threedgrut.layers.road_init import init_road_layer


def _flat_ground_lidar(n: int = 200, x_range: float = 50.0, y_range: float = 50.0) -> torch.Tensor:
    """100×N synthetic 'flat ground' LiDAR points with Z=0."""
    pts = torch.zeros(n, 3)
    pts[:, 0] = (torch.rand(n) * 2 - 1) * x_range
    pts[:, 1] = (torch.rand(n) * 2 - 1) * y_range
    pts[:, 2] = 0.0
    return pts


def _ego_trajectory(n: int = 20, length: float = 80.0) -> torch.Tensor:
    """Straight line ego traj along +X axis at Z=1.5 (driver height)."""
    traj = torch.zeros(n, 3)
    traj[:, 0] = torch.linspace(-length / 2, length / 2, n)
    traj[:, 1] = 0.0
    traj[:, 2] = 1.5
    return traj


def test_road_init_z_lock():
    """T3.3.a: mock 100 flat ground points Z=0 → init 后所有 Z 误差 < 0.05 m."""
    road_pts = _flat_ground_lidar(n=100)
    traj = _ego_trajectory(n=10, length=20.0)
    positions, _, _, _, _ = init_road_layer(road_pts, traj, cut_range=5.0, resolution=0.5, max_n=2_000)
    assert positions.shape[0] > 0, "should produce some BEV grid points"
    z_err = positions[:, 2].abs().max().item()
    assert z_err < 0.05, f"Z lock failed: max |Z| = {z_err}"


def test_road_init_scale_flat():
    """T3.3.a: scales.exp()[:, 2] < 0.005 (路面薄盘约束)."""
    road_pts = _flat_ground_lidar(n=50)
    traj = _ego_trajectory(n=5, length=10.0)
    _, _, scales, _, _ = init_road_layer(road_pts, traj, cut_range=2.0, resolution=0.5, max_n=200)
    assert scales.shape[0] > 0
    sz_max = scales[:, 2].exp().max().item()
    assert sz_max < 0.005, f"flat Z scale violated: max exp(sz) = {sz_max}"
    # XY scale 应该是 ~0.1 m（scale_prior 默认）
    sxy = scales[:, :2].exp()
    assert (0.05 < sxy).all() and (
        sxy < 0.2
    ).all(), f"XY scale out of range: min={sxy.min().item()}, max={sxy.max().item()}"


def test_road_init_handles_empty_lidar():
    """T3.3.a: 空 LiDAR 输入 → 不 crash，返回 shape=(0, ...) tensors."""
    road_pts = torch.zeros(0, 3)
    traj = _ego_trajectory(n=10, length=20.0)
    positions, rotations, scales, densities, colors = init_road_layer(
        road_pts, traj, cut_range=5.0, resolution=0.5, max_n=200
    )
    assert positions.shape == (0, 3)
    assert rotations.shape == (0, 4)
    assert scales.shape == (0, 3)
    assert densities.shape == (0, 1)
    assert colors.shape == (0, 3)


def test_road_init_respects_max_n():
    """T3.3.a: 当 BEV 候选格点数 > max_n 时，输出截到 max_n."""
    road_pts = _flat_ground_lidar(n=500, x_range=100.0, y_range=100.0)
    traj = _ego_trajectory(n=20, length=200.0)
    # 100×100m + cut_range=20 → 140×140m BEV at res=0.5m → 78400 grid points
    positions, _, _, _, _ = init_road_layer(road_pts, traj, cut_range=20.0, resolution=0.5, max_n=1_000)
    assert positions.shape[0] <= 1_000, f"max_n cap violated: got {positions.shape[0]} > 1_000"


def test_road_init_rotations_are_identity_quat():
    """T3.3.a: 默认 rotation = identity quat (1, 0, 0, 0) (wxyz convention)."""
    road_pts = _flat_ground_lidar(n=20)
    traj = _ego_trajectory(n=5, length=5.0)
    _, rotations, _, _, _ = init_road_layer(road_pts, traj, cut_range=2.0, resolution=1.0, max_n=100)
    if rotations.shape[0] > 0:
        assert torch.allclose(rotations[:, 0], torch.ones(rotations.shape[0]))
        assert torch.allclose(rotations[:, 1:], torch.zeros(rotations.shape[0], 3))


def test_road_init_z_follows_uneven_terrain():
    """T3.3.a: 当 LiDAR 点有 Z 梯度（坡道），grid Z 应跟随最近邻路面点而非锁 0."""
    # 100 ramp points: X 从 -50 到 50, Z = 0.1 * X
    n = 100
    road_pts = torch.zeros(n, 3)
    road_pts[:, 0] = torch.linspace(-50, 50, n)
    road_pts[:, 2] = 0.1 * road_pts[:, 0]  # slope: 10% grade
    traj = _ego_trajectory(n=10, length=80.0)
    positions, _, _, _, _ = init_road_layer(road_pts, traj, cut_range=2.0, resolution=2.0, max_n=2_000)
    # 在 X≈0 附近的点 Z 应该 ≈0; 在 X≈40 附近 Z ≈4
    near_zero = positions[positions[:, 0].abs() < 2.0]
    near_forty = positions[(positions[:, 0] - 40.0).abs() < 2.0]
    if near_zero.shape[0] > 0 and near_forty.shape[0] > 0:
        assert near_zero[:, 2].abs().mean().item() < 0.5
        assert (near_forty[:, 2].mean().item() - 4.0) < 1.0


# ---------------------------------------------------------------------------
# E3.2.5 ① — KNN-median Z snap (init 提质：从「最近单点」改局部 KNN 中值降噪).
# 攻 spec §5「init 带噪」失败因，对齐 recon-studio 起伏 8mm 致密毯。
# 默认 knn_k=1 保留 legacy「最近单点」行为（off baseline 字节等价）；
# on run 显式传 knn_k=5 启用中值滤离群。
# ---------------------------------------------------------------------------


def _dense_flat_with_spikes() -> torch.Tensor:
    """致密平地 Z=0 网格 + 确定性离群尖刺（每第 10 点 +0.3m）.

    25×25=625 致密点，~62 个离群（10%）。x-major flatten 下 ``[::10]`` 的离群
    在空间上对角稀疏分布，任一 BEV 格点的 5 近邻里离群是少数 → 中值可滤除，
    而最近单点（k=1）会被尖刺污染。
    """
    span, step = 12.0, 1.0
    xs = torch.arange(-span, span + step, step)
    ys = torch.arange(-span, span + step, step)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    pts = torch.zeros(gx.numel(), 3)
    pts[:, 0] = gx.flatten()
    pts[:, 1] = gy.flatten()
    pts[:, 2] = 0.0
    pts[::10, 2] = 0.3  # deterministic outlier spikes
    return pts


def test_road_init_knn_median_rejects_outliers():
    """E3.2.5①: knn_k=5 中值滤除 +0.3m 离群尖刺，grid Z 贴合真平面 Z=0."""
    road_pts = _dense_flat_with_spikes()
    traj = _ego_trajectory(n=10, length=20.0)
    positions, _, _, _, _ = init_road_layer(road_pts, traj, cut_range=2.0, resolution=1.0, max_n=5_000, knn_k=5)
    assert positions.shape[0] > 0
    z_abs = positions[:, 2].abs()
    assert z_abs.max().item() < 0.05, f"KNN median failed to reject spikes: max|Z|={z_abs.max().item()}"
    rms = (z_abs**2).mean().sqrt().item()
    assert rms < 0.02, f"KNN median residual RMS too high: {rms}"


def test_road_init_knn_k1_is_legacy_nearest():
    """E3.2.5①: knn_k=1 退化为 legacy 最近单点（守不变量）——会被离群污染.

    这是 knn_k=5 的对照锚：同一带噪输入下 k=1 max|Z|→尖刺值，凸显中值价值。
    """
    road_pts = _dense_flat_with_spikes()
    traj = _ego_trajectory(n=10, length=20.0)
    positions, _, _, _, _ = init_road_layer(road_pts, traj, cut_range=2.0, resolution=1.0, max_n=5_000, knn_k=1)
    assert positions.shape[0] > 0
    # legacy 行为：grid 会吸到离群尖刺 → max|Z| 接近 0.3
    assert (
        positions[:, 2].abs().max().item() > 0.1
    ), "knn_k=1 should reproduce legacy nearest-single-point (spike-polluted)"


def test_road_init_knn_median_preserves_slope():
    """E3.2.5①: KNN 中值不抹平真实坡度（只杀离群，不杀信号）."""
    n = 200
    road_pts = torch.zeros(n, 3)
    road_pts[:, 0] = torch.linspace(-50, 50, n)
    road_pts[:, 1] = (torch.arange(n) % 5 - 2).float() * 0.5  # 横向展开给 KNN 2D 邻域
    road_pts[:, 2] = 0.1 * road_pts[:, 0]  # 10% grade
    traj = _ego_trajectory(n=10, length=80.0)
    positions, _, _, _, _ = init_road_layer(road_pts, traj, cut_range=2.0, resolution=2.0, max_n=5_000, knn_k=5)
    near_zero = positions[positions[:, 0].abs() < 2.0]
    near_forty = positions[(positions[:, 0] - 40.0).abs() < 2.0]
    if near_zero.shape[0] > 0 and near_forty.shape[0] > 0:
        assert near_zero[:, 2].abs().mean().item() < 0.5
        assert abs(near_forty[:, 2].mean().item() - 4.0) < 1.0


# ---------------------------------------------------------------------------
# E3.2.5 ①-2 — config wiring: road_init_knn_k rides on LayerSpec so the CLI
# form ++layers.overrides.road.road_init_knn_k=5 reaches init_road_layer via
# trainer (same registry-override path as max_n_particles). Default 1 keeps the
# off baseline (multilayer.yaml, no override) byte-identical.
# ---------------------------------------------------------------------------


def test_layerspec_road_init_knn_k_default_1():
    """E3.2.5①: road_init_knn_k defaults 1 = legacy nearest-single-point."""
    s = LayerSpec(name="road", layer_id=1, max_n_particles=200_000)
    assert s.road_init_knn_k == 1


def test_registry_routes_road_init_knn_k():
    """E3.2.5①: ++layers.overrides.road.road_init_knn_k=5 lands on road spec only."""
    conf = OmegaConf.create(
        {
            "layers": {
                "enabled": ["background", "road"],
                "overrides": {"road": {"road_init_knn_k": 5}},
            }
        }
    )
    specs = specs_from_config(conf)
    road = next(s for s in specs if s.name == "road")
    bg = next(s for s in specs if s.name == "background")
    assert road.road_init_knn_k == 5
    assert bg.road_init_knn_k == 1  # only road overridden
