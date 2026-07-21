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

def test_ownership_dir_summary(tmp_path):
 from PIL import Image
 f=m().summarize_ownership_dirs
 for name in ("bg", "road", "sky"):
  (tmp_path/name).mkdir()
 for name, value in (("bg", 128), ("road", 204)):
  Image.new("L", (5,5), value).save(tmp_path/name/"00000_alpha.png")
 Image.new("L", (5,5), 255).save(tmp_path/"bg"/"00000_roadmask.png")
 Image.new("RGB", (5,5), (10,20,30)).save(tmp_path/"sky"/"00000_sky.png")
 report=f(tmp_path/"bg",tmp_path/"road",tmp_path/"sky")
 assert report["summary"]["n_frames"]==1 and report["summary"]["n_valid_px_total"]==9
