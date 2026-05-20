# SPDX-License-Identifier: Apache-2.0
"""T6F.1 unit tests for ego mask → Batch.mask wiring (Stage 6-fix).

Stage 1-6 bug: NCoreDataset 在 __init__ 加载了 ego car mask 并 dilate 缓存到
`sequence_cameras_frame_valid_pixels_masks`，但训练分支 __getitem__ 从未读取，
也未把任何 mask 字段塞进 batch_dict → get_gpu_batch_with_intrinsics 没有把
val 分支已有的 "valid" 字段拷贝进 Batch.mask → trainer.get_losses 的
`if mask is not None` 永远跳过 → ego 车身像素参与 L1 / layered_l1。

T6F.1 修复：
  - 训练分支注入 `batch_dict["valid"]`（2D bool [H, W]，downsample 走
    cv2 INTER_NEAREST 对齐 val 分支）。
  - `get_gpu_batch_with_intrinsics` 把 "valid" 转 `Batch.mask` reshape
    [1, H, W, 1] float32 GPU。

直接 import datasetNcore 需要 ncore SDK + cv2 + imageio 等，故本套测试走
纯 mock 路径：验证 (a) Batch.mask 形状契约，(b) datasetNcore 的 reshape /
dtype 转换逻辑等价于纯函数，(c) ego mask 接通后 layered_l1 自动从三区
partition 剔除 ego 像素。端到端集成 verification 在 T6F.3 A800 跑。
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from threedgrut.datasets.protocols import Batch
from threedgrut.model.layered_loss import compute_layered_l1_loss


# --- (a) Batch.mask 形状契约 -----------------------------------------------

def _make_minimal_batch(mask: torch.Tensor | None) -> Batch:
    """构造跑通 Batch.__post_init__ 的最小桩，rays/pose 用 1x4x4x3 / 1x4x4."""
    B, H, W = 1, 4, 4
    return Batch(
        rays_ori=torch.zeros(B, H, W, 3),
        rays_dir=torch.zeros(B, H, W, 3),
        T_to_world=torch.eye(4).unsqueeze(0),
        rgb_gt=torch.zeros(B, H, W, 3),
        mask=mask,
    )


def test_batch_mask_field_4d_shape_passes_post_init():
    """T6F.1 接口契约：mask shape [1, H, W, 1] 通过 protocols.Batch.__post_init__."""
    mask = torch.ones(1, 4, 4, 1, dtype=torch.float32)
    batch = _make_minimal_batch(mask)
    assert batch.mask is not None
    assert batch.mask.shape == (1, 4, 4, 1)
    assert batch.mask.dtype == torch.float32


def test_batch_mask_field_3d_shape_rejected_by_post_init():
    """T6F.1 防御：Batch.__post_init__ 要求 mask.ndim == 4（防止误传 3D [B,H,W]）.

    确认 get_gpu_batch_with_intrinsics 的 `reshape(1, h, w, 1)` 不可省略.
    """
    mask_3d = torch.ones(1, 4, 4, dtype=torch.float32)
    with pytest.raises(AssertionError, match="mask must be a 3D tensor"):
        _make_minimal_batch(mask_3d)


# --- (b) reshape / dtype 等价契约 (与 datasetNcore.get_gpu_batch_with_intrinsics 同源) ---

def _ego_reshape_logic(valid, h, w):
    """提取 datasetNcore.get_gpu_batch_with_intrinsics 中的 T6F.1 reshape 段（CPU 版）：

        if not isinstance(valid, torch.Tensor):
            valid = torch.from_numpy(valid)
        mask = valid.to(device, non_blocking=True).float()
        batch_dict["mask"] = mask.reshape(1, h, w, 1)
    """
    if not isinstance(valid, torch.Tensor):
        valid = torch.from_numpy(valid)
    return valid.float().reshape(1, h, w, 1)


def test_valid_2d_reshape_to_4d_mask():
    """T6F.1 训练分支：valid 是 2D numpy bool [H, W]（cv2 resize 输出）→ [1, H, W, 1] float32.

    sum 守恒；dtype 转 float32；通过 Batch.__post_init__.
    """
    H, W = 6, 8
    rng = np.random.default_rng(0)
    valid_2d = rng.integers(0, 2, size=(H, W), dtype=np.int8).astype(bool)
    expected_sum = int(valid_2d.sum())

    mask = _ego_reshape_logic(valid_2d, H, W)

    assert mask.shape == (1, H, W, 1)
    assert mask.dtype == torch.float32
    assert mask.sum().item() == expected_sum
    # Batch.__post_init__ 接受
    batch = _make_minimal_batch(mask[:, :4, :4, :])  # 同 _make_minimal_batch (4x4)
    assert batch.mask is not None


def test_valid_1d_reshape_to_4d_mask():
    """T6F.1 验证分支：valid 是 1D numpy bool [H*W]（按 pixel order 采样）→ [1, H, W, 1] float32.

    数值与 2D 等价（同样数据 reshape 不变）.
    """
    H, W = 6, 8
    rng = np.random.default_rng(0)
    valid_2d = rng.integers(0, 2, size=(H, W), dtype=np.int8).astype(bool)
    valid_1d = valid_2d.flatten()
    assert valid_1d.shape == (H * W,)

    mask_from_1d = _ego_reshape_logic(valid_1d, H, W)
    mask_from_2d = _ego_reshape_logic(valid_2d, H, W)

    assert mask_from_1d.shape == (1, H, W, 1)
    assert torch.equal(mask_from_1d, mask_from_2d), (
        "1D 和 2D valid 经 reshape 后应数值完全一致（reshape 是 row-major 顺序无关）"
    )


# --- (c) ego mask 接通 layered_l1 后剔除 ego 像素 ---------------------------

def test_layered_l1_with_ego_mask_excludes_ego_pixels():
    """T6F.1 集成契约：当 valid_mask（ego 接通）传入 compute_layered_l1_loss，
    ego=False 区域的像素误差不进入 loss.

    构造：pred 与 gt 在 ego=False 区相差 0.5、ego=True 区相差 0.0 →
      - valid_mask=None：loss = mean |diff| = 0.5 * 50% = 0.25
      - valid_mask=ego  ：loss ≈ 0（ego 区被剔除，其他区误差也 0）
    """
    H, W = 8, 8
    rgb_gt = torch.zeros(1, H, W, 3)
    rgb_pred = torch.zeros(1, H, W, 3)
    # 在下半图（"ego 车身区"）造 0.5 误差
    rgb_pred[:, H // 2:, :, :] = 0.5

    ego_valid = torch.ones(1, H, W, dtype=torch.float32)
    ego_valid[:, H // 2:, :] = 0.0  # ego 区 invalid

    loss_no_mask = compute_layered_l1_loss(rgb_pred, rgb_gt, valid_mask=None)
    loss_with_mask = compute_layered_l1_loss(rgb_pred, rgb_gt, valid_mask=ego_valid)

    # 无 mask: 半张图 |diff|=0.5 → mean = 0.25
    assert loss_no_mask.item() == pytest.approx(0.25, abs=1e-6)
    # 有 mask: 误差被完全 mask 掉 → ≈ 0
    assert loss_with_mask.item() == pytest.approx(0.0, abs=1e-6)


def test_layered_l1_valid_mask_none_is_v1_byte_identical():
    """T6F.1 byte-identical 回归：valid_mask=None 路径与 v1 .mean() 完全一致.

    确保 NCore 之外的 dataset（NeRF / Colmap）/ 无 ego mask 数据集不受 T6F.1 影响.
    """
    H, W = 8, 8
    rng = torch.Generator().manual_seed(42)
    rgb_pred = torch.rand(1, H, W, 3, generator=rng)
    rgb_gt = torch.rand(1, H, W, 3, generator=rng)

    loss = compute_layered_l1_loss(rgb_pred, rgb_gt, valid_mask=None)
    expected = (rgb_pred - rgb_gt).abs().mean(dim=-1).mean()
    assert loss.item() == pytest.approx(expected.item(), abs=1e-7)


def test_layered_l1_accepts_4d_valid_mask_with_image_infos():
    """T6F.1 回归（A800 5k smoke 暴露的 bug 2026-05-20）：

    Batch.mask 形状是 [B, H, W, 1] (protocols 4D 契约 + RGB broadcast 需要)，
    而 image_infos 的 sky/road/dyn 是 [B, H, W] (T3.1.b). 老版本
    `compute_layered_l1_loss` line 70 `bg = valid * (1-road) * ...` 会 broadcast
    错配 (1920 vs 1080). 修复：函数顶部 squeeze 4D valid_mask 最后维到 3D.

    本测试以 [B, H, W, 1] valid_mask + 完整 image_infos 调用，验证不再爆.
    """
    B, H, W = 1, 8, 16
    rgb_pred = torch.rand(B, H, W, 3)
    rgb_gt = torch.rand(B, H, W, 3)
    image_infos = {
        "sky_mask": torch.zeros(B, H, W),
        "road_mask": torch.zeros(B, H, W),
        "dyn_mask_sseg": torch.zeros(B, H, W),
    }
    # 注意：[B, H, W, 1] 而不是 [B, H, W]，模拟真实 Batch.mask
    valid_mask_4d = torch.ones(B, H, W, 1, dtype=torch.float32)
    # 应当不抛 RuntimeError("size of tensor a (W) must match tensor b (H) at dim 2")
    loss = compute_layered_l1_loss(
        rgb_pred, rgb_gt, image_infos=image_infos, valid_mask=valid_mask_4d
    )
    assert loss.dim() == 0  # scalar
    assert torch.isfinite(loss)


def test_layered_l1_4d_and_3d_valid_mask_numerically_equivalent():
    """T6F.1 回归：4D [B,H,W,1] 和 3D [B,H,W] valid_mask 应数值等价 (squeeze 一致)."""
    B, H, W = 1, 8, 16
    torch.manual_seed(7)
    rgb_pred = torch.rand(B, H, W, 3)
    rgb_gt = torch.rand(B, H, W, 3)
    image_infos = {
        "sky_mask": (torch.rand(B, H, W) > 0.7).float(),
        "road_mask": (torch.rand(B, H, W) > 0.6).float(),
        "dyn_mask_sseg": (torch.rand(B, H, W) > 0.8).float(),
    }
    valid_3d = (torch.rand(B, H, W) > 0.3).float()
    valid_4d = valid_3d.unsqueeze(-1)

    loss_3d = compute_layered_l1_loss(rgb_pred, rgb_gt, image_infos=image_infos, valid_mask=valid_3d)
    loss_4d = compute_layered_l1_loss(rgb_pred, rgb_gt, image_infos=image_infos, valid_mask=valid_4d)
    assert loss_3d.item() == pytest.approx(loss_4d.item(), abs=1e-7)


def test_layered_l1_all_valid_mask_equivalent_to_no_mask():
    """T6F.1 退化测：valid_mask 全 1 应等价于 valid_mask=None（数值上同 .mean()）.

    对于 NCore 上 ego mask 不存在的相机（如 LiDAR-only 单 lidar 模式 / 老 clip），
    NCoreDataset 应回退到不塞 "valid" → mask=None，行为与全 1 valid 等价.
    """
    H, W = 8, 8
    rng = torch.Generator().manual_seed(7)
    rgb_pred = torch.rand(1, H, W, 3, generator=rng)
    rgb_gt = torch.rand(1, H, W, 3, generator=rng)
    valid_all = torch.ones(1, H, W, dtype=torch.float32)

    loss_none = compute_layered_l1_loss(rgb_pred, rgb_gt, valid_mask=None)
    loss_all = compute_layered_l1_loss(rgb_pred, rgb_gt, valid_mask=valid_all)
    # fallback 分支：(l1 * valid).sum() / valid.sum() == l1.sum() / numel == l1.mean()
    assert loss_none.item() == pytest.approx(loss_all.item(), abs=1e-6)
