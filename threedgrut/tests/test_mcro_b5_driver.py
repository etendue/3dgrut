import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "scripts/drivers/mcro_b5_ownership_ab.sh"


def test_driver_bash_syntax():
    subprocess.run(["bash", "-n", str(DRIVER)], check=True)


def test_driver_has_frozen_inceptio_and_eval_contract():
    text = DRIVER.read_text()
    for required in (
        "n_iterations=5000",
        "dataset.train.duration_sec=5.0",
        "dataset.val.duration_sec=5.0",
        "trainer.use_lidar_depth=false",
        "trainer.use_depth_prior=false",
        "dataset.load_lidar_depth_map=false",
        "dataset.load_depth_prior=false",
        "num_workers=10",
        "--novel-view",
        "mcro_layer_ownership_eval",
        "mcro_ownership_guard",
        "--quality-baseline",
    ):
        assert required in text


def test_driver_arms_are_strictly_cumulative_and_single_launch():
    text = DRIVER.read_text()
    assert "ARM_OVERRIDES=()" in text
    assert "ARM_OVERRIDES+=(layers.semantic_disjoint_init=true)" in text
    assert "ARM_OVERRIDES+=(layers.bg_road_exclusion.enabled=true)" in text
    assert "ARM_OVERRIDES+=(loss.road_responsibility.enabled=true)" in text
    assert "road_init_initial_opacity=0.7" in text
    assert "road_init_use_lidar_color=true" in text
    assert "MCRO_5S_BASELINE_DONE" in text
    # One train.py call means one user-confirmed arm per driver invocation.
    assert len(re.findall(r'\"\$PYTHON_BIN\" train\.py', text)) == 1
