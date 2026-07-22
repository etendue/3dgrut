import copy

import pytest

from scripts.drivers.mcro_ownership_guard import evaluate_guards


def _guards():
    return {
        "comparison": {
            "background_in_front_of_road_alpha_reduction_fraction_min": 0.5,
            "road_interior_alpha_p10_min": 0.37,
            "sky_on_road_energy_max": 0.001,
            "full_cc_psnr_masked_drop_db_max": 0.3,
            "road_crop_psnr_drop_db_max": 0.0,
            "road_crop_lpips_increase_max": 0.0,
        },
        "r0_reference": {
            "six_cam": {
                "bg_on_road_alpha_mean": 0.99,
                "bg_in_front_of_road_alpha_mean": 0.02,
                "front_cc_psnr_masked": 22.7,
                "front_road_crop_psnr": 27.1,
                "front_road_crop_lpips": 0.26,
            }
        },
    }


def _ownership():
    return {
        "summary": {
            "bg_on_road_alpha_mean": 0.49,
            "bg_in_front_of_road_alpha_mean": 0.009,
            "road_coverage_p10": 0.40,
            "sky_on_road_energy": 0.0005,
        }
    }


def _metrics():
    return {
        "mean_cc_psnr_masked": 22.5,
        "mean_road_crop_psnr": 27.2,
        "mean_road_crop_lpips": 0.25,
    }


def test_all_frozen_guards_pass():
    report = evaluate_guards(_ownership(), _metrics(), _guards())
    assert report["passed"] is True
    assert report["n_passed"] == report["n_checks"] == 6


def test_each_bad_metric_fails_its_guard():
    cases = [
        ("ownership", "bg_in_front_of_road_alpha_mean", 0.018),
        ("ownership", "road_coverage_p10", 0.2),
        ("ownership", "sky_on_road_energy", 0.1),
        ("metrics", "mean_cc_psnr_masked", 20.0),
        ("metrics", "mean_road_crop_psnr", 26.0),
        ("metrics", "mean_road_crop_lpips", 0.5),
    ]
    for source, key, value in cases:
        own, metrics = copy.deepcopy(_ownership()), copy.deepcopy(_metrics())
        (own["summary"] if source == "ownership" else metrics)[key] = value
        assert evaluate_guards(own, metrics, _guards())["passed"] is False


def test_missing_required_metric_fails_loudly():
    metrics = _metrics()
    metrics.pop("mean_road_crop_psnr")
    with pytest.raises(ValueError, match="mean_road_crop_psnr"):
        evaluate_guards(_ownership(), metrics, _guards())


def test_raw_background_alpha_is_diagnostic_not_a_guard():
    ownership = _ownership()
    ownership["summary"]["bg_on_road_alpha_mean"] = 1.0
    report = evaluate_guards(ownership, _metrics(), _guards())
    assert report["passed"] is True


def test_missing_depth_aware_background_metric_fails_loudly():
    ownership = _ownership()
    ownership["summary"].pop("bg_in_front_of_road_alpha_mean")
    with pytest.raises(ValueError, match="bg_in_front_of_road_alpha_mean"):
        evaluate_guards(ownership, _metrics(), _guards())


def test_same_budget_quality_baseline_overrides_30k_quality_reference_only():
    same_budget = {
        "mean_cc_psnr_masked": 19.0,
        "mean_road_crop_psnr": 23.0,
        "mean_road_crop_lpips": 0.4,
    }
    metrics = {
        "mean_cc_psnr_masked": 18.8,
        "mean_road_crop_psnr": 23.0,
        "mean_road_crop_lpips": 0.4,
    }
    report = evaluate_guards(_ownership(), metrics, _guards(), same_budget)
    assert report["passed"] is True
    # Ownership still compares to the frozen 20s R0, not the 5s baseline.
    bg = next(c for c in report["checks"] if c["name"] == "background_in_front_of_road_alpha")
    assert bg["limit"] == pytest.approx(0.01)
