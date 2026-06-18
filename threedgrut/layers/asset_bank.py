# SPDX-License-Identifier: Apache-2.0
"""E2.8 — AH 资产库查询：按 (label_class, cuboid dims) 从 bank 选最近资产。

区别 e25_inject.match_assets_by_size（greedy bijection，每资产用一次）：bank
查询**不消耗**资产——同一库资产可被多个 track 复用（编辑场景不要 identity）。
fallback ladder：① 同 class L2 最近（level 0）② 跨 class 全局最近 + WARN（level
1）③ on_miss='skip' 时抛 BankMiss 让调用方保留 recon（level 跳过，不 silent）。
"""
from __future__ import annotations

import logging

import torch

from threedgrut.layers.warmstart_metadata import AssetSpec

logger = logging.getLogger(__name__)


class BankMiss(Exception):
    """bank 无可用资产且 on_miss='skip' —— 调用方应保留该 track recon。"""


def _l2(a, b) -> float:
    ta = torch.as_tensor(a, dtype=torch.float32)
    tb = torch.as_tensor(b, dtype=torch.float32)
    return float(torch.linalg.norm(ta - tb))


def query_bank(
    bundle: dict[str, AssetSpec],
    label_class: str,
    dims,
    *,
    on_miss: str = "global",
) -> tuple[str, int]:
    """选最匹配的 ``asset_hash``。返回 ``(asset_hash, fallback_level)``。

    fallback_level: 0=同 class 最近 · 1=跨 class 全局最近（WARN）。
    ``on_miss``: 'global'（默认，空 class 退全局）或 'skip'（空 class 抛 BankMiss）。
    平手按 ``asset_hash`` 字典序（deterministic）。
    """
    same = [s for s in bundle.values() if s.label_class == label_class]
    if same:
        best = min(same, key=lambda s: (_l2(s.cuboids_dims, dims), s.asset_hash))
        return best.asset_hash, 0
    if on_miss == "skip" or not bundle:
        raise BankMiss(
            f"no asset for class {label_class!r} (on_miss={on_miss}, "
            f"bank classes={sorted({s.label_class for s in bundle.values()})})"
        )
    best = min(bundle.values(), key=lambda s: (_l2(s.cuboids_dims, dims), s.asset_hash))
    logger.warning(
        "asset_bank: class %r absent; cross-class fallback → %r (%s)",
        label_class, best.asset_hash, best.label_class,
    )
    return best.asset_hash, 1
