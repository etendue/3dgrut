# SPDX-License-Identifier: Apache-2.0
"""Regression: Renderer.from_checkpoint must apply its ``path`` override onto
the ckpt-embedded conf BEFORE deriving ``object_name = Path(conf.path).stem``.

E2.8 packed/edited ckpts (built by ``build_native_ckpt``) leave ``conf.path``
as OmegaConf MISSING — the dataset path is supplied only at render time via
``--path``. Before this fix, ``from_checkpoint`` read ``conf.path`` for the run
name before the override was applied, so ``python render.py --checkpoint
packed_ckpt.pt --path manifest.json`` crashed with MissingMandatoryValue even
though ``--path`` was given. Pins the override helper used by from_checkpoint.
"""

from omegaconf import OmegaConf

from threedgrut.render import _override_conf_path


def test_override_applies_path_over_missing_value():
    # MISSING (`???`) path, struct-locked — mirrors a packed_ckpt conf.
    conf = OmegaConf.create({"path": "???", "experiment_name": "e28"})
    OmegaConf.set_struct(conf, True)
    _override_conf_path(conf, "/data/9ae151dc/clip.json")
    assert conf.path == "/data/9ae151dc/clip.json"  # no MissingMandatoryValue


def test_override_is_noop_on_empty_path():
    conf = OmegaConf.create({"path": "/orig.json"})
    _override_conf_path(conf, "")
    assert conf.path == "/orig.json"
