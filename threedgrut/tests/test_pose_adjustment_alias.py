# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""V3 pose_adjustment yaml alias verification.

The user-facing interface is ``trainer.pose_adjustment.{enabled, lambda_t,
lambda_r}`` (default disabled — byte-identical with v2 baseline). The
implementation key ``trainer.learnable_pose.*`` is kept as the internal
source of truth (ckpt schema uses ``learnable_pose_state``, so renaming
would break backward compat).

The wiring is OmegaConf interpolation in ``configs/base_gs.yaml``:

    learnable_pose:
      enabled: ${trainer.pose_adjustment.enabled}
      lambda_temporal_smooth_trans: ${trainer.pose_adjustment.lambda_t}
      lambda_temporal_smooth_rot:   ${trainer.pose_adjustment.lambda_r}

These tests pin the four-way contract so a future yaml refactor that
breaks any of the routes is caught immediately:

  1. Default config → both ``pose_adjustment.enabled`` AND the resolved
     ``learnable_pose.enabled`` are False (and lambdas 0.0).
  2. ``trainer.pose_adjustment.enabled=true`` CLI override → resolved
     ``learnable_pose.enabled`` becomes True (interpolation propagates).
  3. ``trainer.pose_adjustment.lambda_t=1e-2`` CLI override → resolved
     ``learnable_pose.lambda_temporal_smooth_trans`` becomes 1e-2.
  4. Backward compat: direct legacy ``trainer.learnable_pose.enabled=true``
     CLI override still works (replaces the interpolation with a literal).
"""
import os

import pytest
from hydra import compose, initialize_config_dir

_CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "configs")
)


def _compose(overrides=()):
    """Compose a tiny multilayer-friendly conf. ``ncore_3dgut_mcmc`` carries
    the full base_gs.yaml chain so the pose_adjustment / learnable_pose
    blocks are populated.
    """
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        return compose(
            config_name="apps/ncore_3dgut_mcmc",
            overrides=list(overrides),
        )


def test_pose_adjustment_default_off():
    """Untouched config → user-facing alias + internal key both disabled
    + lambdas both 0.0. This is the v2 byte-identical guarantee."""
    cfg = _compose()
    assert cfg.trainer.pose_adjustment.enabled is False, \
        "pose_adjustment.enabled MUST default to False"
    assert float(cfg.trainer.pose_adjustment.lambda_t) == 0.0
    assert float(cfg.trainer.pose_adjustment.lambda_r) == 0.0
    # Internal learnable_pose follows via interpolation.
    assert cfg.trainer.learnable_pose.enabled is False
    assert float(cfg.trainer.learnable_pose.lambda_temporal_smooth_trans) == 0.0
    assert float(cfg.trainer.learnable_pose.lambda_temporal_smooth_rot) == 0.0


def test_pose_adjustment_enabled_forwards_to_learnable_pose():
    """Flip the user-facing knob → resolved learnable_pose.enabled flips
    too (OmegaConf interpolation propagates)."""
    cfg = _compose(overrides=["trainer.pose_adjustment.enabled=true"])
    assert cfg.trainer.pose_adjustment.enabled is True
    assert cfg.trainer.learnable_pose.enabled is True, \
        "interpolation broken: pose_adjustment.enabled=true did not propagate to learnable_pose.enabled"


def test_pose_adjustment_lambdas_forward_to_learnable_pose():
    """User-facing lambda_t / lambda_r → resolved
    learnable_pose.lambda_temporal_smooth_{trans,rot}."""
    cfg = _compose(overrides=[
        "trainer.pose_adjustment.enabled=true",
        "trainer.pose_adjustment.lambda_t=1.0e-2",
        "trainer.pose_adjustment.lambda_r=1.0e-1",
    ])
    assert float(cfg.trainer.pose_adjustment.lambda_t) == pytest.approx(1.0e-2)
    assert float(cfg.trainer.pose_adjustment.lambda_r) == pytest.approx(1.0e-1)
    assert float(cfg.trainer.learnable_pose.lambda_temporal_smooth_trans) == pytest.approx(1.0e-2)
    assert float(cfg.trainer.learnable_pose.lambda_temporal_smooth_rot) == pytest.approx(1.0e-1)


def test_legacy_learnable_pose_override_still_works():
    """Backward compat: someone with an old script still doing
    ``trainer.learnable_pose.enabled=true`` must still get a learnable
    run. OmegaConf replaces the ``${...}`` interpolation with the literal
    True when the leaf is directly overridden.
    """
    cfg = _compose(overrides=[
        "trainer.learnable_pose.enabled=true",
        "trainer.learnable_pose.lambda_temporal_smooth_trans=5.0e-3",
    ])
    assert cfg.trainer.learnable_pose.enabled is True, \
        "legacy CLI override trainer.learnable_pose.enabled=true broke"
    assert float(cfg.trainer.learnable_pose.lambda_temporal_smooth_trans) == pytest.approx(5.0e-3)
    # pose_adjustment alias still says False because we didn't touch it —
    # legacy override takes precedence locally without rewriting the alias.
    assert cfg.trainer.pose_adjustment.enabled is False


def test_advanced_internal_knobs_still_overridable():
    """Advanced fields (lr_*, freeze_until_iter, pose_prior_*) have no
    user-facing alias on purpose — verify they're still CLI-reachable
    on ``trainer.learnable_pose.*``."""
    cfg = _compose(overrides=[
        "trainer.pose_adjustment.enabled=true",
        "trainer.learnable_pose.freeze_until_iter=10000",
        "trainer.learnable_pose.lr_rotation=2.0e-5",
    ])
    assert cfg.trainer.learnable_pose.enabled is True   # propagated from alias
    assert int(cfg.trainer.learnable_pose.freeze_until_iter) == 10000
    assert float(cfg.trainer.learnable_pose.lr_rotation) == pytest.approx(2.0e-5)


def test_multilayer_poseopt_yaml_turns_pose_adjustment_on():
    """End-to-end: the published app yaml ``apps/ncore_3dgut_mcmc_multilayer_poseopt``
    must turn the feature on via the new user-facing alias, with the
    DriveStudio-magnitude lambdas hardcoded."""
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        cfg = compose(config_name="apps/ncore_3dgut_mcmc_multilayer_poseopt")
    assert cfg.trainer.pose_adjustment.enabled is True
    assert float(cfg.trainer.pose_adjustment.lambda_t) == pytest.approx(1.0e-2)
    assert float(cfg.trainer.pose_adjustment.lambda_r) == pytest.approx(1.0e-1)
    # Internal route is consistent.
    assert cfg.trainer.learnable_pose.enabled is True
    assert float(cfg.trainer.learnable_pose.lambda_temporal_smooth_trans) == pytest.approx(1.0e-2)
    assert float(cfg.trainer.learnable_pose.lambda_temporal_smooth_rot) == pytest.approx(1.0e-1)
