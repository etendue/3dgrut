# SPDX-License-Identifier: Apache-2.0
"""Pure mocked tests for the read-only NCore clip readiness gate."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.ncore_data_readiness import (
    V4_MULTILAYER_PROFILE_CONTRACT,
    V4_REQUIRED_AUX_TYPES,
    validate_ncore_data_readiness,
    validate_v4_multilayer_dataset_contract,
    validate_v4_multilayer_profile_contract,
)


CAMERAS = ("camera_a", "camera_b")
ROOT = Path(__file__).resolve().parents[2]


def _validate(manifest, camera_ids=CAMERAS, *, required_aux=V4_REQUIRED_AUX_TYPES, **kwargs):
    """Invoke the generic gate with this test module's explicit v4 profile."""

    return validate_ncore_data_readiness(
        manifest,
        camera_ids,
        required_aux=required_aux,
        **kwargs,
    )


class _Array:
    def __init__(self, value=1, *, corrupt: bool = False, shape=(2,)):
        self.shape = shape
        self.value = value
        self.corrupt = corrupt

    def __getitem__(self, index):
        if self.corrupt:
            raise OSError("corrupt array payload")
        if index is Ellipsis and self.shape:
            return np.full(self.shape, self.value)
        return self.value


class _Group:
    def __init__(self, children=None):
        self.children = children or {}

    def group_keys(self):
        return [key for key, value in self.children.items() if isinstance(value, _Group)]

    def array_keys(self):
        return [key for key, value in self.children.items() if isinstance(value, _Array)]

    def __getitem__(self, key):
        value = self
        for part in str(key).split("/"):
            value = value.children[part]
        return value


class _BrokenCanonicalEgomaskGroup(_Group):
    def __getitem__(self, key):
        if key == "aux/egomask":
            raise OSError("egomask index is corrupt")
        return super().__getitem__(key)


class _Sensor:
    def __init__(self, count: int):
        self.frames_timestamps_us = np.arange(count * 2, dtype=np.int64).reshape(count, 2)

    def get_frame_image_array(self, _index):
        return np.zeros((3, 4, 3), dtype=np.uint8)


class _Loader:
    def __init__(self, counts, *, point_counts=(2, 2), fail_point_index=None):
        self.counts = counts
        self.lidar_ids = ["lidar_top"]
        self.point_clouds_ids = []
        self.point_counts = tuple(point_counts)
        self.fail_point_index = fail_point_index

    def get_camera_sensor(self, camera_id):
        if camera_id not in self.counts:
            raise KeyError(camera_id)
        return _Sensor(self.counts[camera_id])

    def get_point_clouds_source(self, source_id):
        if source_id != "lidar_top":
            raise KeyError(source_id)
        point_counts = self.point_counts
        fail_point_index = self.fail_point_index

        class _PointCloudSource:
            pc_timestamps_us = np.arange(len(point_counts), dtype=np.int64)

            def get_pc(self, index):
                if index == fail_point_index:
                    raise OSError("point cloud payload is corrupt")
                return type("PointCloud", (), {"xyz": np.zeros((point_counts[index], 3), dtype=np.float32)})()

        return _PointCloudSource()


def _png_bytes(width: int = 4, height: int = 3) -> bytes:
    stream = io.BytesIO()
    Image.new("L", (width, height), color=1).save(stream, format="PNG")
    return stream.getvalue()


def _frame_group(count: int, *, corrupt: bool = False, keys=None, value=1, shape=(2,)) -> _Group:
    frame_keys = tuple(str(index) for index in range(count)) if keys is None else tuple(str(key) for key in keys)
    return _Group(
        {
            key: _Array(value=value, corrupt=corrupt and index == 0, shape=shape)
            for index, key in enumerate(frame_keys)
        }
    )


def _nested(path: str, value: _Group) -> _Group:
    root = value
    for part in reversed(path.split("/")):
        root = _Group({part: root})
    return root


def _copy_v4_profile_minimum(target: Path, *, include_sidecar: bool = True) -> tuple[Path, Path]:
    contract = V4_MULTILAYER_PROFILE_CONTRACT
    names = ["config", "runtime_artifact"]
    if include_sidecar:
        names.append("provenance_sidecar")
    for name in names:
        relative = Path(contract[name]["path"])
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    return (
        target / contract["config"]["path"],
        target / contract["runtime_artifact"]["path"],
    )


def _fixture(tmp_path: Path, *, raw_counts=None, sseg_counts=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    raw_counts = raw_counts or {camera_id: 2 for camera_id in CAMERAS}
    sseg_counts = sseg_counts or dict(raw_counts)
    component_names = ("clip.ncore4-camera_a.zarr.itar", "clip.ncore4-meta.zarr.itar")
    manifest = tmp_path / "clip.json"
    manifest.write_text(
        json.dumps(
            {
                "sequence_id": "clip",
                "version": 4,
                "sequence_timestamp_interval_us": {"start": 0, "stop": 10},
                "component_stores": [
                    {"path": name, "md5": "unused", "components": {}} for name in component_names
                ],
            }
        ),
        encoding="utf-8",
    )
    roots = {}
    for name in component_names:
        (tmp_path / name).touch()
        roots[name] = _nested("raw", _frame_group(1))

    aux_names = {
        "sseg": "clip.aux.sseg.zarr.itar",
        "egomask": "clip.aux.egomask.zarr.itar",
        "lidar-sseg": "clip.aux.lidar-sseg.zarr.itar",
        "lidar-camvis": "clip.aux.lidar-camvis.zarr.itar",
    }
    for name in aux_names.values():
        (tmp_path / name).touch()
    roots[aux_names["sseg"]] = _nested(
        "aux/semantic_segmentation",
        _Group(
            {
                camera_id: _frame_group(
                    sseg_counts.get(camera_id, 0),
                    keys=(2 * index + 1 for index in range(sseg_counts.get(camera_id, 0))),
                    value=_png_bytes(),
                    shape=(),
                )
                for camera_id in sseg_counts
            }
        ),
    )
    roots[aux_names["egomask"]] = _nested(
        "aux/egomask",
        _Group({camera_id: _frame_group(1, value=_png_bytes(), shape=()) for camera_id in CAMERAS}),
    )
    roots[aux_names["lidar-sseg"]] = _nested(
        "aux/lidar_semantic_segmentation",
        _Group({"lidar_top": _frame_group(2, value=_png_bytes(2, 1), shape=())}),
    )
    roots[aux_names["lidar-camvis"]] = _nested(
        "aux/lidar_camera_visibility", _Group({"lidar_top": _frame_group(2, shape=(2, 2))})
    )

    def opener(path: Path):
        return roots[path.name]

    return manifest, roots, aux_names, opener, lambda _path: _Loader(raw_counts)


def test_v4_profile_contract_accepts_only_frozen_config_artifact_and_sidecar() -> None:
    contract = V4_MULTILAYER_PROFILE_CONTRACT
    resolved = validate_v4_multilayer_profile_contract(
        ROOT,
        ROOT / contract["config"]["path"],
        ROOT / contract["runtime_artifact"]["path"],
    )
    assert resolved["provenance_sidecar"] == (ROOT / contract["provenance_sidecar"]["path"]).resolve()


@pytest.mark.parametrize("wrong_input", ("config", "artifact"))
def test_v4_profile_contract_rejects_legacy_v3_inputs(wrong_input: str) -> None:
    contract = V4_MULTILAYER_PROFILE_CONTRACT
    config = ROOT / contract["config"]["path"]
    artifact = ROOT / contract["runtime_artifact"]["path"]
    if wrong_input == "config":
        config = ROOT / "configs/apps/ncore_3dgut_mcmc_multilayer_inceptio_7cam.yaml"
    else:
        artifact = ROOT / "scripts/pin_ftheta_b6a9_7cam_params.json"
    with pytest.raises(ValueError, match=rf"v4-multilayer .*{wrong_input}.*path mismatch"):
        validate_v4_multilayer_profile_contract(ROOT, config, artifact)


def test_v4_profile_contract_rejects_missing_or_stale_sidecar(tmp_path: Path) -> None:
    config, artifact = _copy_v4_profile_minimum(tmp_path, include_sidecar=False)
    with pytest.raises(ValueError, match=r"profile file is missing or unreadable:.*provenance"):
        validate_v4_multilayer_profile_contract(tmp_path, config, artifact)

    sidecar_relative = Path(V4_MULTILAYER_PROFILE_CONTRACT["provenance_sidecar"]["path"])
    sidecar = tmp_path / sidecar_relative
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"provenance_sidecar SHA-256 mismatch"):
        validate_v4_multilayer_profile_contract(tmp_path, config, artifact)


def test_v4_dataset_contract_requires_frozen_sha_clip_and_fourteen_stores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "clip.json"
    value = {
        "sequence_id": V4_MULTILAYER_PROFILE_CONTRACT["clip_id"],
        "component_stores": [{"path": f"store-{index}.itar"} for index in range(14)],
    }
    manifest.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setitem(
        V4_MULTILAYER_PROFILE_CONTRACT,
        "manifest_sha256",
        hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )
    assert len(validate_v4_multilayer_dataset_contract(manifest)["component_stores"]) == 14

    value["component_stores"].pop()
    manifest.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setitem(
        V4_MULTILAYER_PROFILE_CONTRACT,
        "manifest_sha256",
        hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match=r"exactly 14 component stores"):
        validate_v4_multilayer_dataset_contract(manifest)


def test_happy_mocked_schema_accepts_static_egomask_counts(tmp_path: Path) -> None:
    manifest, _, _, opener, loader = _fixture(tmp_path)
    report = _validate(
        manifest, CAMERAS, store_opener=opener, sequence_loader_factory=loader
    )
    assert report["component_store_count"] == 2
    assert report["raw_camera_frame_counts"] == {"camera_a": 2, "camera_b": 2}
    assert report["sseg_frame_counts"] == report["raw_camera_frame_counts"]
    assert report["egomask_static_array_counts"] == {"camera_a": 1, "camera_b": 1}


def test_happy_dict_form_component_stores(tmp_path: Path) -> None:
    manifest, _, _, opener, loader = _fixture(tmp_path)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    paths = [entry["path"] for entry in value["component_stores"]]
    value["component_stores"] = {"camera": paths[0], "metadata": {"path": paths[1]}}
    manifest.write_text(json.dumps(value), encoding="utf-8")
    report = _validate(
        manifest, CAMERAS, store_opener=opener, sequence_loader_factory=loader
    )
    assert report["component_store_count"] == 2


def test_missing_manifest_component_fails_with_exact_store(tmp_path: Path) -> None:
    manifest, _, _, opener, loader = _fixture(tmp_path)
    missing = tmp_path / "clip.ncore4-camera_a.zarr.itar"
    missing.unlink()
    with pytest.raises(ValueError, match=r"component store\[0\] missing:.*camera_a"):
        _validate(manifest, CAMERAS, store_opener=opener, sequence_loader_factory=loader)


def test_corrupt_manifest_component_payload_fails_closed(tmp_path: Path) -> None:
    manifest, roots, _, opener, loader = _fixture(tmp_path)
    roots["clip.ncore4-camera_a.zarr.itar"] = _nested("raw", _frame_group(1, corrupt=True))
    with pytest.raises(ValueError, match=r"component store\[0\].*representative array is unreadable"):
        _validate(manifest, CAMERAS, store_opener=opener, sequence_loader_factory=loader)


@pytest.mark.parametrize("aux_type", ("sseg", "egomask", "lidar-sseg", "lidar-camvis"))
def test_each_required_aux_store_is_mandatory(tmp_path: Path, aux_type: str) -> None:
    manifest, _, aux_names, opener, loader = _fixture(tmp_path)
    (tmp_path / aux_names[aux_type]).unlink()
    with pytest.raises(ValueError, match=rf"required aux\.{aux_type} store missing"):
        _validate(manifest, CAMERAS, store_opener=opener, sequence_loader_factory=loader)


def test_required_aux_is_a_mandatory_keyword_only_profile(tmp_path: Path) -> None:
    manifest, _, _, opener, loader = _fixture(tmp_path)
    with pytest.raises(TypeError, match="required_aux"):
        validate_ncore_data_readiness(
            manifest,
            CAMERAS,
            store_opener=opener,
            sequence_loader_factory=loader,
        )


def test_required_aux_profile_can_be_scoped_explicitly(tmp_path: Path) -> None:
    manifest, _, aux_names, opener, loader = _fixture(tmp_path)
    for aux_type in ("egomask", "lidar-sseg", "lidar-camvis"):
        (tmp_path / aux_names[aux_type]).unlink()
    report = _validate(
        manifest,
        CAMERAS,
        required_aux=("sseg",),
        store_opener=opener,
        sequence_loader_factory=loader,
    )
    assert set(report["aux_paths"]) == {"sseg"}
    assert report["egomask_static_array_counts"] == {}
    assert report["lidar_aux_frame_counts"] == {}


def test_missing_active_raw_camera_coverage_fails(tmp_path: Path) -> None:
    manifest, _, _, opener, _ = _fixture(tmp_path)
    loader = lambda _path: _Loader({"camera_a": 2})
    with pytest.raises(ValueError, match=r"active raw camera camera_b is unavailable"):
        _validate(manifest, CAMERAS, store_opener=opener, sequence_loader_factory=loader)


def test_zero_active_raw_camera_coverage_fails(tmp_path: Path) -> None:
    manifest, _, _, opener, _ = _fixture(tmp_path)
    loader = lambda _path: _Loader({"camera_a": 2, "camera_b": 0})
    with pytest.raises(ValueError, match=r"active raw camera camera_b has zero frames"):
        _validate(manifest, CAMERAS, store_opener=opener, sequence_loader_factory=loader)


def test_raw_camera_resolution_unavailable_fails_closed(tmp_path: Path) -> None:
    manifest, _, _, opener, _ = _fixture(tmp_path)
    loader_value = _Loader({camera_id: 2 for camera_id in CAMERAS})

    def get_camera_sensor(camera_id):
        if camera_id not in CAMERAS:
            raise KeyError(camera_id)
        return type(
            "ResolutionlessSensor",
            (),
            {"frames_timestamps_us": np.arange(4, dtype=np.int64).reshape(2, 2)},
        )()

    loader_value.get_camera_sensor = get_camera_sensor
    with pytest.raises(ValueError, match=r"camera_a exposes neither a representative image nor resolution"):
        _validate(
            manifest,
            CAMERAS,
            store_opener=opener,
            sequence_loader_factory=lambda _path: loader_value,
        )


def test_missing_sseg_camera_and_count_mismatch_fail_precisely(tmp_path: Path) -> None:
    manifest, _, _, opener, loader = _fixture(tmp_path, sseg_counts={"camera_a": 2})
    with pytest.raises(ValueError, match=r"aux\.sseg missing active camera camera_b"):
        _validate(manifest, CAMERAS, store_opener=opener, sequence_loader_factory=loader)

    manifest, _, _, opener, loader = _fixture(
        tmp_path / "mismatch", sseg_counts={"camera_a": 2, "camera_b": 1}
    )
    with pytest.raises(ValueError, match=r"raw/sseg frame-count mismatch for camera_b: raw=2 sseg=1"):
        _validate(manifest, CAMERAS, store_opener=opener, sequence_loader_factory=loader)


def test_equal_count_different_raw_sseg_timestamp_keys_fail(tmp_path: Path) -> None:
    manifest, roots, aux_names, opener, loader = _fixture(tmp_path)
    sseg_group = roots[aux_names["sseg"]]["aux/semantic_segmentation"]
    sseg_group.children["camera_b"] = _frame_group(
        2, keys=(1, 999), value=_png_bytes(), shape=()
    )
    with pytest.raises(
        ValueError,
        match=r"raw/sseg timestamp-key mismatch for camera_b:.*missing=.*3.*unexpected=.*999",
    ):
        _validate(manifest, CAMERAS, store_opener=opener, sequence_loader_factory=loader)


def test_equal_count_different_lidar_timestamp_keys_fail(tmp_path: Path) -> None:
    manifest, roots, aux_names, opener, loader = _fixture(tmp_path)
    camvis_group = roots[aux_names["lidar-camvis"]]["aux/lidar_camera_visibility"]
    camvis_group.children["lidar_top"] = _frame_group(2, keys=(0, 999))
    with pytest.raises(ValueError, match=r"lidar aux timestamp-key mismatch for lidar_top"):
        _validate(manifest, CAMERAS, store_opener=opener, sequence_loader_factory=loader)


def test_both_lidar_aux_shifted_together_still_fail_against_raw_source(tmp_path: Path) -> None:
    manifest, roots, aux_names, opener, loader = _fixture(tmp_path)
    roots[aux_names["lidar-sseg"]]["aux/lidar_semantic_segmentation"].children["lidar_top"] = _frame_group(
        2, keys=(10, 11), value=_png_bytes(2, 1), shape=()
    )
    roots[aux_names["lidar-camvis"]]["aux/lidar_camera_visibility"].children["lidar_top"] = _frame_group(
        2, keys=(10, 11), shape=(2, 2)
    )
    with pytest.raises(ValueError, match=r"raw/lidar aux timestamp-key mismatch for lidar_top"):
        _validate(manifest, CAMERAS, store_opener=opener, sequence_loader_factory=loader)


@pytest.mark.parametrize("failure", ("corrupt_png", "camvis_length"))
def test_lidar_payload_decode_and_point_count_gate(tmp_path: Path, failure: str) -> None:
    manifest, roots, aux_names, opener, loader = _fixture(tmp_path)
    if failure == "corrupt_png":
        group = roots[aux_names["lidar-sseg"]]["aux/lidar_semantic_segmentation/lidar_top"]
        group.children["0"] = _Array(value=b"not a png", shape=())
        error = r"aux\.lidar-sseg/lidar_top/0 PNG payload is unreadable"
    else:
        group = roots[aux_names["lidar-camvis"]]["aux/lidar_camera_visibility/lidar_top"]
        group.children["0"] = _Array(value=1, shape=(3, 2))
        error = r"lidar point-count mismatch.*raw=2 lidar-sseg=2 lidar-camvis=3"
    with pytest.raises(ValueError, match=error):
        _validate(manifest, CAMERAS, store_opener=opener, sequence_loader_factory=loader)


def test_lidar_later_frame_count_mismatch_fails_closed(tmp_path: Path) -> None:
    manifest, roots, aux_names, opener, loader = _fixture(tmp_path)
    sseg_group = roots[aux_names["lidar-sseg"]]["aux/lidar_semantic_segmentation/lidar_top"]
    camvis_group = roots[aux_names["lidar-camvis"]]["aux/lidar_camera_visibility/lidar_top"]
    sseg_group.children["1"] = _Array(value=_png_bytes(3, 1), shape=())
    camvis_group.children["1"] = _Array(value=1, shape=(3, 2))
    with pytest.raises(
        ValueError,
        match=r"lidar point-count mismatch for lidar_top/1: raw=2 lidar-sseg=3 lidar-camvis=3",
    ):
        _validate(manifest, CAMERAS, store_opener=opener, sequence_loader_factory=loader)


def test_lidar_later_raw_point_cloud_unreadable_fails_closed(tmp_path: Path) -> None:
    manifest, _, _, opener, _ = _fixture(tmp_path)
    loader = lambda _path: _Loader({camera_id: 2 for camera_id in CAMERAS}, fail_point_index=1)
    with pytest.raises(ValueError, match=r"raw point cloud lidar_top/1 is unreadable"):
        _validate(manifest, CAMERAS, store_opener=opener, sequence_loader_factory=loader)


@pytest.mark.parametrize("aux_type", ("sseg", "egomask"))
@pytest.mark.parametrize("failure", ("corrupt_png", "wrong_shape"))
def test_camera_aux_representative_png_is_decoded_and_resolution_checked(
    tmp_path: Path, aux_type: str, failure: str
) -> None:
    manifest, roots, aux_names, opener, loader = _fixture(tmp_path)
    group_path = "aux/semantic_segmentation" if aux_type == "sseg" else "aux/egomask"
    group = roots[aux_names[aux_type]][group_path]
    value = b"nonempty but not a png" if failure == "corrupt_png" else _png_bytes(5, 3)
    first_key = group.children["camera_a"].array_keys()[0]
    group.children["camera_a"].children[first_key] = _Array(value=value, shape=())
    error = (
        rf"aux\.{aux_type}.*camera_a.*PNG payload is unreadable"
        if failure == "corrupt_png"
        else rf"aux\.{aux_type} resolution mismatch for camera_a"
    )
    with pytest.raises(ValueError, match=error):
        _validate(manifest, CAMERAS, store_opener=opener, sequence_loader_factory=loader)


@pytest.mark.parametrize("shape", ((3, 4), (3, 4, 3)))
def test_plain_egomask_hw_and_hwc_arrays_use_leading_image_dimensions(tmp_path: Path, shape) -> None:
    manifest, roots, aux_names, opener, loader = _fixture(tmp_path)
    group = roots[aux_names["egomask"]]["aux/egomask"]
    group.children["camera_a"].children["0"] = _Array(value=1, shape=shape)
    report = _validate(
        manifest, CAMERAS, store_opener=opener, sequence_loader_factory=loader
    )
    assert report["egomask_resolutions"]["camera_a"] == (4, 3)


@pytest.mark.parametrize(
    ("aux_type", "canonical_group"),
    (
        ("lidar-sseg", "lidar_semantic_segmentation"),
        ("lidar-camvis", "lidar_camera_visibility"),
    ),
)
@pytest.mark.parametrize("failure", ("empty", "corrupt"))
def test_lidar_aux_validates_canonical_payload_not_decoy_root_array(
    tmp_path: Path, aux_type: str, canonical_group: str, failure: str
) -> None:
    manifest, roots, aux_names, opener, loader = _fixture(tmp_path)
    target = _Group({}) if failure == "empty" else _frame_group(1, corrupt=True)
    roots[aux_names[aux_type]] = _Group(
        {
            "decoy": _Array(),
            "aux": _Group({canonical_group: _Group({"lidar_top": target})}),
        }
    )
    error = "has zero frames" if failure == "empty" else "representative array is unreadable"
    with pytest.raises(ValueError, match=rf"aux\.{aux_type}/lidar_top.*{error}"):
        _validate(manifest, CAMERAS, store_opener=opener, sequence_loader_factory=loader)


def test_egomask_canonical_io_error_does_not_fall_back(tmp_path: Path) -> None:
    manifest, roots, aux_names, opener, loader = _fixture(tmp_path)
    roots[aux_names["egomask"]] = _BrokenCanonicalEgomaskGroup(
        {
            "decoy": _Array(),
            "aux": _Group({"egomask": _Group({camera_id: _frame_group(1) for camera_id in CAMERAS})}),
        }
    )
    with pytest.raises(ValueError, match=r"canonical group 'aux/egomask' is unreadable.*index is corrupt"):
        _validate(manifest, CAMERAS, store_opener=opener, sequence_loader_factory=loader)
