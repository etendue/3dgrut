#!/usr/bin/env python3
"""轻量 viser 点云查看器 —— 快速预览 reconstruction-studio 的 3DGS 产物。

这是「方案②：点云快速预览」的实现：不做真高斯渲染，只把每个基元的中心点
用其颜色画出来，用于秒开确认坐标系 / 场景结构是否正确，作为真高斯版的前置验证。

支持输入：
  - .ply  带 red/green/blue (uint8)            -> 直接用      (如 *_color_*.ply)
  - .ply  带 f_dc_0..2     (SH degree0 3DGS)   -> rgb=f_dc*C0+0.5 (如 *_gs_*.ply)
  - .usd  UsdGeomPoints    (points+displayColor)-> 直接用      (如 *_color_*.usd)

用法：
  python view_pointcloud.py --input test_set_30000_color_Background.ply --port 8091
  python view_pointcloud.py --input xxx_gs_Background.ply --max_points 1500000
"""
import argparse
import os
import time

import numpy as np

C0 = 0.28209479177387814  # SH degree-0 基函数常数 (rgb = f_dc * C0 + 0.5)


def load_ply(path):
    from plyfile import PlyData
    print(f"[load] reading PLY {path} ...", flush=True)
    ply = PlyData.read(path)
    v = ply["vertex"].data
    names = v.dtype.names
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    if all(c in names for c in ("red", "green", "blue")):
        rgb = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.uint8)
        src = "uint8 red/green/blue"
    elif all(c in names for c in ("f_dc_0", "f_dc_1", "f_dc_2")):
        fdc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float32)
        rgb = (np.clip(fdc * C0 + 0.5, 0, 1) * 255).astype(np.uint8)
        src = "f_dc -> RGB (SH deg0)"
    else:
        rgb = np.full((len(xyz), 3), 180, np.uint8)
        src = "no color (gray)"
    return xyz, rgb, src


def load_usd(path):
    from pxr import Usd, UsdGeom
    print(f"[load] opening USD {path} ...", flush=True)
    stage = Usd.Stage.Open(path)
    prim = next((p for p in stage.Traverse() if p.GetTypeName() == "Points"), None)
    if prim is None:
        raise RuntimeError("no UsdGeomPoints prim found in USD")
    xyz = np.array(UsdGeom.Points(prim).GetPointsAttr().Get(), np.float32)
    dc = prim.GetAttribute("primvars:displayColor").Get()
    if dc is not None and len(dc) == len(xyz):
        rgb = (np.clip(np.array(dc, np.float32), 0, 1) * 255).astype(np.uint8)
        src = "primvars:displayColor"
    elif dc is not None and len(dc) >= 1:
        c = (np.clip(np.array(dc[0], np.float32), 0, 1) * 255).astype(np.uint8)
        rgb = np.tile(c, (len(xyz), 1))
        src = "primvars:displayColor (const)"
    else:
        rgb = np.full((len(xyz), 3), 180, np.uint8)
        src = "no color (gray)"
    return xyz, rgb, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to .ply / .usd")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--max_points", type=int, default=0, help="0=全部；否则随机降采样到该上限")
    ap.add_argument("--point_size", type=float, default=0.0, help="0=按 bbox 自动估算")
    ap.add_argument("--up", default="+z", help="初始 up 方向：+z/-z/+y/-y/+x/-x")
    args = ap.parse_args()

    t0 = time.time()
    ext = os.path.splitext(args.input)[1].lower()
    xyz, rgb, src = (load_usd if ext.startswith(".usd") else load_ply)(args.input)
    n0 = len(xyz)
    print(f"[load] N={n0:,}  color_src={src}  ({time.time() - t0:.1f}s)", flush=True)

    if args.max_points and n0 > args.max_points:
        idx = np.random.default_rng(0).choice(n0, args.max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
        print(f"[subsample] {n0:,} -> {len(xyz):,}", flush=True)

    bmin, bmax = xyz.min(0), xyz.max(0)
    diag = float(np.linalg.norm(bmax - bmin))
    print(f"[bbox] min={np.round(bmin, 2)} max={np.round(bmax, 2)} diag={diag:.2f}", flush=True)
    psize = args.point_size if args.point_size > 0 else max(diag / 3000.0, 1e-4)
    print(f"[point_size] {'manual' if args.point_size > 0 else 'auto'}={psize:.5f}", flush=True)

    import viser

    server = viser.ViserServer(port=args.port)
    try:
        server.scene.set_up_direction(args.up)
    except Exception as e:  # noqa: BLE001
        print("set_up_direction failed:", e, flush=True)

    state = {"h": None}

    def draw(ps, shape):
        if state["h"] is not None:
            state["h"].remove()
        state["h"] = server.scene.add_point_cloud(
            "/pointcloud", points=xyz, colors=rgb, point_size=ps, point_shape=shape
        )

    draw(psize, "circle")
    server.scene.add_frame("/world", axes_length=diag * 0.05, axes_radius=max(diag * 0.002, 1e-4))

    with server.gui.add_folder("controls"):
        g_ps = server.gui.add_slider(
            "point size", min=psize * 0.1, max=psize * 8, step=psize * 0.05, initial_value=psize
        )
        g_sh = server.gui.add_dropdown(
            "shape", options=["circle", "square", "rounded", "diamond", "sparkle"], initial_value="circle"
        )
        g_up = server.gui.add_dropdown(
            "up axis", options=["+z", "-z", "+y", "-y", "+x", "-x"], initial_value=args.up
        )
        server.gui.add_text("points", initial_value=f"{len(xyz):,}", disabled=True)
        server.gui.add_text("bbox diag", initial_value=f"{diag:.2f}", disabled=True)
        server.gui.add_text("color src", initial_value=src, disabled=True)

    @g_ps.on_update
    def _(_):
        draw(g_ps.value, g_sh.value)

    @g_sh.on_update
    def _(_):
        draw(g_ps.value, g_sh.value)

    @g_up.on_update
    def _(_):
        try:
            server.scene.set_up_direction(g_up.value)
        except Exception as e:  # noqa: BLE001
            print("up err:", e, flush=True)

    print(f"[viser] READY on :{args.port}  ->  http://<host-ip>:{args.port}", flush=True)
    while True:
        time.sleep(2.0)


if __name__ == "__main__":
    main()
