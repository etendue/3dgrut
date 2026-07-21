import importlib.util
from pathlib import Path
import pytest
import torch

P=Path(__file__).resolve().parents[2]/"scripts/drivers/mcro_layer_ownership_eval.py"
def m():
 s=importlib.util.spec_from_file_location("own",P); x=importlib.util.module_from_spec(s); s.loader.exec_module(x); return x
def test_ownership_metrics_and_erosion():
 f=m().compute_ownership_metrics
 bg=torch.tensor([[[[.1],[.2],[.3]],[[.4],[.5],[.6]],[[.7],[.8],[.9]]]])
 road=torch.ones_like(bg)*.8; rgb=torch.zeros(1,3,3,3); sky=torch.zeros_like(rgb); mask=torch.ones_like(bg,dtype=torch.bool)
 r=f(bg,road,rgb,sky,mask,erosion_px=1)
 assert r["n_valid_px"]==1 and r["bg_on_road_alpha_mean"]==pytest.approx(.5) and r["sky_on_road_energy"]==0
def test_empty_road_mask_is_nan():
 f=m().compute_ownership_metrics; z=torch.zeros(1,2,2,1); r=f(z,z,z.repeat(1,1,1,3),z.repeat(1,1,1,3),torch.zeros_like(z,dtype=torch.bool),erosion_px=0)
 assert r["n_valid_px"]==0 and r["road_coverage_p10"]!=r["road_coverage_p10"]
