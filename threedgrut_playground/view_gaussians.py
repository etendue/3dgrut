#!/usr/bin/env python3
"""轻量 viser 真高斯查看器 —— 方案①。

读 3DGS PLY (reconstruction-studio 约定: raw 值, 本数据为 SH degree0),
转成 viser `add_gaussian_splats` 需要的 centers/covariances/rgbs/opacities,
在浏览器用 viser 自带 WebGL 渲染真高斯椭球。
不依赖 gsplat / OptiX / CUDA 渲染器 —— 纯前端渲染,只需 viser+numpy+plyfile。

激活约定 (对齐 utils/gaussian_utils.py 的 save_to_ply, 存的是 pre-activation 值):
  位置     x,y,z
  颜色     f_dc_0..2     -> rgb = f_dc*C0 + 0.5        (SH degree0, 无视角高光)
  不透明度 opacity (raw) -> sigmoid
  尺度     scale_0..2(log)-> exp                        (各轴标准差)
  旋转     rot_0..3 (wxyz, raw) -> 归一化四元数 -> R
  协方差   Σ = R · diag(s^2) · R^T

用法:
  python view_gaussians.py --input test_set_30000_gs_Background.ply --port 8092
  # 浏览器卡顿时降采样(按不透明度加权,保留视觉贡献大的):
  python view_gaussians.py --input ..._gs_...ply --max_gaussians 1500000
  # 整体椭球过糊调小 / 过碎调大:
  python view_gaussians.py --input ..._gs_...ply --splat_scale 0.8
"""
import argparse
import os
import time

import numpy as np

C0 = 0.28209479177387814


def sigmoid(x):
    # 数值稳定版,避免 exp 溢出
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def quat_to_rotmat(q):  # q (N,4) wxyz
    q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3), np.float32)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def load_gs_ply(path):
    from plyfile import PlyData
    print(f"[load] reading {path} ...", flush=True)
    v = PlyData.read(path)["vertex"].data
    names = v.dtype.names
    need = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
            "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    missing = [n for n in need if n not in names]
    if missing:
        raise RuntimeError(f"PLY 缺少 3DGS 字段 {missing} —— 这不是高斯 ply?")
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    fdc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float32)
    op = np.asarray(v["opacity"], np.float32)
    scale = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1).astype(np.float32)
    quat = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float32)
    has_rest = any(n.startswith("f_rest_") for n in names)
    return xyz, fdc, op, scale, quat, has_rest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--max_gaussians", type=int, default=0, help="0=全部;否则按不透明度加权降采样到该上限")
    ap.add_argument("--splat_scale", type=float, default=1.0, help="整体椭球缩放(过糊调小/过碎调大)")
    ap.add_argument("--max_scale", type=float, default=0.0, help="clamp 椭球各轴上限(米);0=自动按 float16 安全值(~250)防溢出")
    ap.add_argument("--min_opacity", type=float, default=0.0, help=">0 时过滤 sigmoid(opacity) 低于此值的高斯")
    ap.add_argument("--up", default="+z")
    args = ap.parse_args()

    t0 = time.time()
    xyz, fdc, op_raw, scale_raw, quat, has_rest = load_gs_ply(args.input)
    n0 = len(xyz)
    print(f"[load] N={n0:,}  has_f_rest={has_rest}  ({time.time() - t0:.1f}s)", flush=True)
    if has_rest:
        print("[warn] 检测到高阶 SH,但 viser 只用 DC 近似(无视角相关高光)", flush=True)

    # 激活
    rgb = np.clip(fdc * C0 + 0.5, 0.0, 1.0).astype(np.float32)        # (N,3) [0,1]
    op = sigmoid(op_raw).reshape(-1, 1)                               # (N,1)
    s = np.exp(np.clip(scale_raw, -20, 20)).astype(np.float32)        # (N,3) 标准差

    # (可选) 过滤近透明高斯(减少雾感与排序开销)
    if args.min_opacity > 0:
        keep = op.reshape(-1) >= args.min_opacity
        xyz, rgb, op, s, quat = xyz[keep], rgb[keep], op[keep], s[keep], quat[keep]
        print(f"[filter] min_opacity={args.min_opacity}: kept {int(keep.sum()):,}/{n0}", flush=True)

    # clamp 椭球上限: 必须 < ~256, 否则 viser 把协方差压成 float16 会溢出(max 65504, 即 scale^2);
    #                 也顺带抑制巨型高斯糊屏。默认 250(仅动极端高斯, 最忠实)。
    cap = min(args.max_scale, 250.0) if args.max_scale > 0 else 250.0
    n_big = int((s > cap).any(axis=1).sum())
    if n_big:
        s = np.minimum(s, cap)
    print(f"[clamp] max_scale={cap:.1f}m  affected {n_big:,} gaussians", flush=True)

    # (可选) 降采样(opacity 加权,保留视觉贡献大的高斯)
    cur = len(xyz)
    if args.max_gaussians and cur > args.max_gaussians:
        w = op.reshape(-1).astype(np.float64)
        w = w / w.sum()
        idx = np.random.default_rng(0).choice(cur, args.max_gaussians, replace=False, p=w)
        xyz, rgb, op, s, quat = xyz[idx], rgb[idx], op[idx], s[idx], quat[idx]
        print(f"[subsample] {cur:,} -> {len(xyz):,} (opacity-weighted)", flush=True)

    # 协方差 Σ = R diag(s^2) R^T = (R*s) (R*s)^T
    t1 = time.time()
    R = quat_to_rotmat(quat)
    RS = R * s[:, None, :]
    cov = (RS @ RS.transpose(0, 2, 1)).astype(np.float32)             # (N,3,3)
    print(f"[cov] {cov.shape} built in {time.time() - t1:.1f}s", flush=True)

    bmin, bmax = xyz.min(0), xyz.max(0)
    diag = float(np.linalg.norm(bmax - bmin))
    print(f"[bbox] min={np.round(bmin, 2)} max={np.round(bmax, 2)} diag={diag:.2f}", flush=True)
    print(f"[stat] scale(exp)   min={s.min():.4f} med={np.median(s):.4f} max={s.max():.4f}", flush=True)
    print(f"[stat] opacity(sig) min={op.min():.3f} med={np.median(op):.3f} max={op.max():.3f}", flush=True)

    import viser

    server = viser.ViserServer(port=args.port)
    try:
        server.scene.set_up_direction(args.up)
    except Exception as e:  # noqa: BLE001
        print("set_up_direction failed:", e, flush=True)

    server.scene.add_gaussian_splats(
        "/splats", centers=xyz, covariances=cov, rgbs=rgb, opacities=op, scale=args.splat_scale
    )

    with server.gui.add_folder("controls"):
        g_up = server.gui.add_dropdown("up axis", options=["+z", "-z", "+y", "-y", "+x", "-x"], initial_value=args.up)
        server.gui.add_text("gaussians", initial_value=f"{len(xyz):,}", disabled=True)
        server.gui.add_text("bbox diag", initial_value=f"{diag:.2f}", disabled=True)
        server.gui.add_text("SH", initial_value="degree0 (DC)" if not has_rest else "f_rest->DC approx", disabled=True)
        server.gui.add_text("splat_scale", initial_value=f"{args.splat_scale}", disabled=True)

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
