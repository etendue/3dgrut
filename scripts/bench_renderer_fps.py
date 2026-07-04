"""Headless FPS benchmark for Engine3DGRUT (3DGUT / 3DGRT).

Usage:
    python scripts/bench_renderer_fps.py \
        --gs_object /path/to/ckpt_last.pt \
        --renderer 3dgut \
        [--resolution 1024] [--warmup 5] [--n_frames 50]

Measures pure render FPS at the given resolution, no viser server needed.
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kaolin.render.camera import Camera

from threedgrut_playground.engine import Engine3DGRUT


def make_camera(resolution: int, device: str = "cuda") -> Camera:
    """Identity-pose pinhole camera, 60-degree FOV, square resolution."""
    view_matrix = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)
    fov_y = math.radians(60.0)
    return Camera.from_args(
        view_matrix=view_matrix,
        fov=fov_y,
        width=resolution,
        height=resolution,
        near=0.1,
        far=1000.0,
        dtype=torch.float32,
        device=device,
    )


def main():
    parser = argparse.ArgumentParser(description="Headless render FPS benchmark")
    parser.add_argument("--gs_object", required=True, help="Path to checkpoint .pt")
    parser.add_argument("--renderer", default="3dgut", choices=["3dgut", "3dgrt"])
    parser.add_argument("--default_gs_config", default="apps/colmap_3dgrt.yaml")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--n_frames", type=int, default=50)
    args = parser.parse_args()

    print(f"[bench] checkpoint : {args.gs_object}")
    print(f"[bench] renderer   : {args.renderer}")
    print(f"[bench] resolution : {args.resolution}x{args.resolution}")

    assets = os.path.join(os.path.dirname(__file__), "../threedgrut_playground/assets")
    engine = Engine3DGRUT(
        gs_object=args.gs_object,
        mesh_assets_folder=assets,
        default_config=args.default_gs_config,
        renderer=args.renderer,
    )
    engine.use_spp = False
    engine.use_depth_of_field = False
    engine.use_optix_denoiser = False

    camera = make_camera(args.resolution)

    print(f"[bench] warmup ({args.warmup} frames)…")
    for _ in range(args.warmup):
        with torch.no_grad():
            engine.render_pass(camera, is_first_pass=True)
    torch.cuda.synchronize()

    print(f"[bench] benchmarking {args.n_frames} frames…")
    times = []
    for i in range(args.n_frames):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            engine.render_pass(camera, is_first_pass=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        if (i + 1) % 10 == 0:
            print(f"  frame {i+1:3d}/{args.n_frames}  {1/(t1-t0):.1f} fps")

    t = np.array(times)
    print(f"\n{'='*52}")
    print(f"  renderer  : {args.renderer}")
    print(f"  resolution: {args.resolution}x{args.resolution}")
    print(f"  mean FPS  : {1.0/t.mean():.1f}")
    print(f"  median FPS: {1.0/np.median(t):.1f}")
    print(f"  p5  FPS   : {1.0/np.percentile(t, 95):.1f}   (slow tail)")
    print(f"  p95 FPS   : {1.0/np.percentile(t, 5):.1f}   (fast burst)")
    print(f"  mean ms   : {t.mean()*1000:.1f} ms/frame")
    print(f"  min  ms   : {t.min()*1000:.1f} ms")
    print(f"  max  ms   : {t.max()*1000:.1f} ms")
    print(f"{'='*52}")


if __name__ == "__main__":
    main()
