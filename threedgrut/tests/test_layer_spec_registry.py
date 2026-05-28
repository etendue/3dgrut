# SPDX-License-Identifier: Apache-2.0
"""Pure-Python unit tests for LayerSpec dataclass + layers/registry.

These tests deliberately avoid importing torch / MixtureOfGaussians / hydra
compose so they can run on a developer laptop without CUDA. The heavier
container/checkpoint contract tests live in test_layered_gaussians.py and
require the A800 environment.

Coverage:
  T1.2 part 1 (LayerSpec extended fields)  -- test_layer_spec_*
  T1.2 part 2 (registry STANDARD_LAYERS)   -- test_registry_*
  T1.2 part 3 (specs_from_config factory)  -- test_specs_from_config_*
"""
from dataclasses import FrozenInstanceError

import pytest
from omegaconf import OmegaConf

from threedgrut.layers.layer_spec import LayerSpec


# --------------------------------------------------------------- T1.2 part 1
def test_layer_spec_frozen_immutable():
    """LayerSpec 是 frozen dataclass：运行时改字段抛 FrozenInstanceError"""
    spec = LayerSpec(name="background", layer_id=0, max_n_particles=100)
    with pytest.raises(FrozenInstanceError):
        spec.name = "road"


def test_layer_spec_full_field_defaults():
    """T1.2 字段：未显式传时使用默认值，传时透传"""
    minimal = LayerSpec(name="background", layer_id=0, max_n_particles=100)
    assert minimal.scale_prior == (0.1, 0.1, 0.1)
    assert minimal.scale_lr_mult == 1.0
    assert minimal.mask_field is None
    assert minimal.is_particle_layer is True
    assert minimal.density_init == 0.1

    explicit = LayerSpec(
        name="road", layer_id=1, max_n_particles=200_000,
        scale_prior=(0.1, 0.1, 0.001), scale_lr_mult=0.2,
        mask_field="road_mask", is_particle_layer=True, density_init=0.05,
    )
    assert explicit.scale_prior == (0.1, 0.1, 0.001)
    assert explicit.scale_lr_mult == 0.2
    assert explicit.mask_field == "road_mask"
    assert explicit.density_init == 0.05


# --------------------------------------------------------------- T1.2 part 2
def test_registry_standard_layers_complete():
    """STANDARD_LAYERS 含 5 个 v2 标准层"""
    from threedgrut.layers.registry import STANDARD_LAYERS
    assert set(STANDARD_LAYERS.keys()) == {
        "background", "road", "dynamic_rigids",
        "dynamic_deformables", "sky_envmap",
    }


def test_registry_specs_have_unique_ids():
    """每个 layer 的 layer_id 唯一"""
    from threedgrut.layers.registry import STANDARD_LAYERS
    ids = [s.layer_id for s in STANDARD_LAYERS.values()]
    assert len(ids) == len(set(ids))


def test_registry_particle_flags_correct():
    """sky_envmap 和 dynamic_deformables 不是粒子层"""
    from threedgrut.layers.registry import STANDARD_LAYERS
    assert STANDARD_LAYERS["background"].is_particle_layer is True
    assert STANDARD_LAYERS["road"].is_particle_layer is True
    assert STANDARD_LAYERS["dynamic_rigids"].is_particle_layer is True
    assert STANDARD_LAYERS["sky_envmap"].is_particle_layer is False
    assert STANDARD_LAYERS["dynamic_deformables"].is_particle_layer is False


def test_registry_road_layer_has_flat_scale_prior():
    """Road 层 scale_prior 第三维必须远小于前两维（thin-disc Z-lock 约定）"""
    from threedgrut.layers.registry import STANDARD_LAYERS
    sx, sy, sz = STANDARD_LAYERS["road"].scale_prior
    assert sz < sx
    assert sz < sy
    assert STANDARD_LAYERS["road"].mask_field == "road_mask"


# --------------------------------------------------------------- T1.2 part 3
def test_specs_from_config_filters_enabled():
    """specs_from_config 只返回 enabled list 中的 layer，保持顺序"""
    from threedgrut.layers.registry import specs_from_config
    conf = OmegaConf.create({"layers": {"enabled": ["background", "road"]}})
    specs = specs_from_config(conf)
    assert [s.name for s in specs] == ["background", "road"]


def test_specs_from_config_unknown_layer_raises():
    """enabled 含未知 layer name → 明确 ValueError 提示可选名"""
    from threedgrut.layers.registry import specs_from_config
    conf = OmegaConf.create({"layers": {"enabled": ["background", "bogus"]}})
    with pytest.raises(ValueError, match="bogus"):
        specs_from_config(conf)


def test_specs_from_config_defaults_to_single_background():
    """conf 无 layers 字段 → fallback 单 background"""
    from threedgrut.layers.registry import specs_from_config
    conf = OmegaConf.create({})
    specs = specs_from_config(conf)
    assert len(specs) == 1
    assert specs[0].name == "background"


# --------------------------------------------------------------- T8/B3
def test_specs_from_config_overrides_max_n_particles():
    """T8/B3: ``layers.overrides.<name>.max_n_particles`` applied via replace."""
    from threedgrut.layers.registry import specs_from_config
    conf = OmegaConf.create({
        "layers": {
            "enabled": ["background", "dynamic_rigids"],
            "overrides": {"dynamic_rigids": {"max_n_particles": 700_000}},
        }
    })
    specs = specs_from_config(conf)
    by_name = {s.name: s for s in specs}
    assert by_name["dynamic_rigids"].max_n_particles == 700_000
    # background is untouched
    assert by_name["background"].max_n_particles == 600_000


def test_specs_from_config_overrides_unknown_field_raises():
    from threedgrut.layers.registry import specs_from_config
    conf = OmegaConf.create({
        "layers": {
            "enabled": ["dynamic_rigids"],
            "overrides": {"dynamic_rigids": {"max_n_particles_typo": 700_000}},
        }
    })
    with pytest.raises(ValueError, match="max_n_particles_typo"):
        specs_from_config(conf)


# --------------------------------------------------------------- V3-L5/L8/L9
def test_v3_extra_keys_routed_to_extra_dict():
    """V3-L5/L8/L9: ``symmetric_axis`` / ``optimize_track_albedo`` /
    ``optimize_track_scale`` / ``track_warmup_steps`` etc. are NOT
    LayerSpec dataclass fields — they must land in ``spec.extra`` instead
    of raising."""
    from threedgrut.layers.registry import specs_from_config
    conf = OmegaConf.create({
        "layers": {
            "enabled": ["dynamic_rigids"],
            "overrides": {
                "dynamic_rigids": {
                    "max_n_particles": 700_000,
                    "symmetric_axis": "Y",
                    "optimize_track_albedo": True,
                    "optimize_track_scale": True,
                    "track_warmup_steps": 500,
                    "track_albedo_lr": 1.0e-5,
                    "track_scale_lr": 1.0e-5,
                },
            },
        }
    })
    specs = specs_from_config(conf)
    s = {sp.name: sp for sp in specs}["dynamic_rigids"]
    assert s.max_n_particles == 700_000  # dataclass field still routed correctly
    assert s.extra.get("symmetric_axis") == "Y"
    assert s.extra.get("optimize_track_albedo") is True
    assert s.extra.get("optimize_track_scale") is True
    assert s.extra.get("track_warmup_steps") == 500
    assert s.extra.get("track_albedo_lr") == 1.0e-5
    assert s.extra.get("track_scale_lr") == 1.0e-5


def test_v3_extra_keys_default_off_when_absent():
    """No V3 overrides present → spec.extra is empty (or registry default
    only); the OFF code paths in dynamic_rigid_init / layered_model take
    precedence."""
    from threedgrut.layers.registry import specs_from_config
    conf = OmegaConf.create({
        "layers": {"enabled": ["dynamic_rigids"]},
    })
    specs = specs_from_config(conf)
    s = {sp.name: sp for sp in specs}["dynamic_rigids"]
    # No registry default extras on dynamic_rigids (sky_envmap is the only
    # one with built-in extras).
    assert s.extra.get("symmetric_axis") is None
    assert not s.extra.get("optimize_track_albedo", False)
    assert not s.extra.get("optimize_track_scale", False)


def test_v3_extra_keys_unknown_still_raises():
    """A typo on a V3-style extra key must still raise (no silent no-op)."""
    from threedgrut.layers.registry import specs_from_config
    conf = OmegaConf.create({
        "layers": {
            "enabled": ["dynamic_rigids"],
            "overrides": {
                "dynamic_rigids": {"symmetric_axis_typo": "Y"},
            },
        }
    })
    with pytest.raises(ValueError, match="symmetric_axis_typo"):
        specs_from_config(conf)


def test_v3_extra_keys_preserves_registry_default_extras():
    """sky_envmap has registry default ``extra={"backend": "cubemap",
    "resolution": 128}``. Overriding one extra key MUST NOT drop the others."""
    from threedgrut.layers.registry import specs_from_config
    # We piggy-back on dynamic_rigids — sky_envmap has no V3 extra keys but
    # we still want to make sure the merge logic preserves existing extras.
    # Use a fresh dynamic_rigids extra default by patching registry.
    from threedgrut.layers import registry as reg
    # Snapshot + monkey-patch: pretend dyn_rigids ships a baked-in extra.
    orig = reg.STANDARD_LAYERS["dynamic_rigids"]
    try:
        reg.STANDARD_LAYERS["dynamic_rigids"] = type(orig)(
            **{**{f.name: getattr(orig, f.name)
                  for f in __import__("dataclasses").fields(orig)},
               "extra": {"_baked_in": "preserved"}},
        )
        conf = OmegaConf.create({
            "layers": {
                "enabled": ["dynamic_rigids"],
                "overrides": {
                    "dynamic_rigids": {"symmetric_axis": "Y"},
                },
            },
        })
        specs = specs_from_config(conf)
        s = specs[0]
        assert s.extra.get("_baked_in") == "preserved"
        assert s.extra.get("symmetric_axis") == "Y"
    finally:
        reg.STANDARD_LAYERS["dynamic_rigids"] = orig


def test_specs_from_config_no_overrides_section_no_change():
    """``layers.overrides`` absent → specs are pristine STANDARD_LAYERS entries."""
    from threedgrut.layers.registry import STANDARD_LAYERS, specs_from_config
    conf = OmegaConf.create({"layers": {"enabled": ["dynamic_rigids"]}})
    specs = specs_from_config(conf)
    assert specs[0] is STANDARD_LAYERS["dynamic_rigids"]
