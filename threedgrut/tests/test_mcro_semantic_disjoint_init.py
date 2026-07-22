import numpy as np
import pytest
from omegaconf import OmegaConf

from threedgrut.datasets.ncore_semantic import filter_init_points_by_semantics


def test_semantic_disjoint_filters_road_after_dynamic_and_keeps_unknown():
    xyz = np.arange(18, dtype=np.float32).reshape(6, 3)
    color = np.arange(18, dtype=np.uint8).reshape(6, 3)
    labels = np.array([0, 1, 2, 255, 13, 8], dtype=np.uint8)
    dynamic = np.array([0, 0, 0, 0, 1, 0], dtype=np.uint8)

    kept_xyz, kept_color, stats = filter_init_points_by_semantics(
        xyz,
        color,
        labels=labels,
        dynamic_flags=dynamic,
        non_dynamic_points_only=True,
        exclude_semantic_class_ids=frozenset({0, 1}),
    )

    np.testing.assert_array_equal(kept_xyz, xyz[[2, 3, 5]])
    np.testing.assert_array_equal(kept_color, color[[2, 3, 5]])
    assert stats == {
        "n_input_points": 6,
        "n_dynamic_removed": 1,
        "n_bg_points": 3,
        "n_road_points": 2,
        "n_excluded_road_class": 2,
        "n_unknown_kept": 1,
        "n_intersection": 0,
    }


def test_semantic_disjoint_missing_labels_keeps_all_nondynamic_as_unknown():
    xyz = np.arange(12, dtype=np.float32).reshape(4, 3)
    kept_xyz, kept_color, stats = filter_init_points_by_semantics(
        xyz,
        None,
        labels=None,
        dynamic_flags=np.array([0, 1, 0, 0]),
        non_dynamic_points_only=True,
        exclude_semantic_class_ids=frozenset({0, 1}),
    )
    np.testing.assert_array_equal(kept_xyz, xyz[[0, 2, 3]])
    assert kept_color is None
    assert stats["n_unknown_kept"] == 3
    assert stats["n_excluded_road_class"] == 0
    assert stats["n_intersection"] == 0


def test_semantic_disjoint_rejects_misaligned_labels():
    with pytest.raises(ValueError, match="align 1:1"):
        filter_init_points_by_semantics(
            np.zeros((3, 3), dtype=np.float32),
            None,
            labels=np.zeros(2, dtype=np.uint8),
            dynamic_flags=None,
            non_dynamic_points_only=True,
            exclude_semantic_class_ids=frozenset({0, 1}),
        )


def test_b1_config_default_is_off():
    conf = OmegaConf.load("configs/base_gs.yaml")
    assert conf.layers.semantic_disjoint_init is False
