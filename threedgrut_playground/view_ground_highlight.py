#!/usr/bin/env python3
"""在 viser 里单独高亮 reconstruction-studio Background 层中的「ground/road」高斯。

ground 高斯的指纹(对齐 vanilla.py 主流程的 ground-label 退化):
  - 旋转是单位四元数 [1,0,0,0](平躺、法线竖直朝上)
  - 最薄轴 < 1mm(z 厚度被强制压到 ~1e-6 → 水平 disk)
本脚本据此把 ground 染成醒目色(默认品红),其余环境灰显,
用于直观查看「被钉死的路面那一层」在 Background 里的覆盖范围与形态。

显示模式:
  --mode points  (默认) 点云:任何视角下路面都立刻可见、流畅,最适合看覆盖范围
  --mode splats        真高斯:能看到 disk 盘形态(薄盘边视角较隐,建议俯视)

用法:
  python view_ground_highlight.py --input test_set_30000_gs_Background.ply --port 8093
  python view_ground_highlight.py --input ..._gs_...ply --mode splats
"""
import argparse
import os
import time

import numpy as np

C0 = 0.28209479177387814
COLOR_TABLE = {
    "magenta": (1.0, 0.0, 1.0),
    "red": (1.0, 0.0, 0.0),
    "lime": (0.0, 1.0, 0.0),
    "cyan": (0.0, 1.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
}


def sigmoid(x):
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def quat_to_rotmat(q):
    q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3), np.float32)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def load_gs_ply(path):
    from plyfile import PlyData
    print(f"[load] reading {path} ...", flush=True)
    v = PlyData.read(path)["vertex"].data
    names = v.dtype.names
    need = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
            "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    if any(n not in names for n in need):
        raise RuntimeError("PLY 缺少 3DGS 字段,这不是高斯 ply?")
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    fdc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float32)
    op = np.asarray(v["opacity"], np.float32)
    scale = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1).astype(np.float32)
    quat = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float32)
    return xyz, fdc, op, scale, quat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--mode", choices=["points", "splats"], default="points")
    ap.add_argument("--max_env", type=int, default=0, help=">0 时把环境高斯随机降采样到该上限(ground 永远全保留)")
    ap.add_argument("--up", default="+z")
    args = ap.parse_args()

    t0 = time.time()
    xyz, fdc, op_raw, scale_raw, quat = load_gs_ply(args.input)
    n0 = len(xyz)

    # ground 指纹:单位旋转 + 最薄轴 < 1mm
    qn = quat / np.clip(np.linalg.norm(quat, axis=1, keepdims=True), 1e-12, None)
    identity = (np.abs(qn[:, 0]) > 0.999) & (np.abs(qn[:, 1:]).max(1) < 0.03)
    s = np.exp(np.clip(scale_raw, -30, 30)).astype(np.float32)
    flat = s.min(1) < 1e-3
    gmask = identity & flat
    print(f"[ground] {int(gmask.sum()):,} / {n0:,} ({100 * gmask.mean():.1f}%) gaussians match ground fingerprint", flush=True)

    rgb = np.clip(fdc * C0 + 0.5, 0.0, 1.0).astype(np.float32)
    op = sigmoid(op_raw).reshape(-1, 1)
    s = np.minimum(s, 250.0)  # float16 安全

    # 分组(ground 全保留;env 可降采样)
    idx_g = np.where(gmask)[0]
    idx_e = np.where(~gmask)[0]
    if args.max_env and len(idx_e) > args.max_env:
        idx_e = np.random.default_rng(0).choice(idx_e, args.max_env, replace=False)
        print(f"[subsample] env {int((~gmask).sum()):,} -> {len(idx_e):,}", flush=True)

    bmin, bmax = xyz.min(0), xyz.max(0)
    diag = float(np.linalg.norm(bmax - bmin))
    psize = max(diag / 3000.0, 1e-4)
    print(f"[bbox] diag={diag:.2f}  point_size={psize:.4f}  mode={args.mode}", flush=True)

    cov = None
    if args.mode == "splats":
        t1 = time.time()
        R = quat_to_rotmat(quat)
        RS = R * s[:, None, :]
        cov = (RS @ RS.transpose(0, 2, 1)).astype(np.float32)
        print(f"[cov] built in {time.time() - t1:.1f}s", flush=True)

    def sub(a, idx):
        return a[idx]

    def resolve_color(base, choice, n):
        if choice == "original":
            return base
        if choice == "gray":
            lum = (base * np.array([0.299, 0.587, 0.114], np.float32)).sum(1, keepdims=True)
            return np.repeat(np.clip(lum * 0.55, 0, 1), 3, axis=1).astype(np.float32)
        c = np.array(COLOR_TABLE.get(choice, (1, 0, 1)), np.float32)
        return np.tile(c, (n, 1))

    print(f"[load] total {time.time() - t0:.1f}s; launching viser on :{args.port}", flush=True)
    import viser

    server = viser.ViserServer(port=args.port)
    try:
        server.scene.set_up_direction(args.up)
    except Exception as e:  # noqa: BLE001
        print("set_up_direction failed:", e, flush=True)

    state = {"ground": None, "env": None}

    def draw(group):
        idx = idx_g if group == "ground" else idx_e
        choice = g_gcol.value if group == "ground" else g_ecol.value
        col = resolve_color(sub(rgb, idx), choice, len(idx))
        if state[group] is not None:
            state[group].remove()
        if args.mode == "splats":
            h = server.scene.add_gaussian_splats(
                f"/{group}", centers=sub(xyz, idx), covariances=sub(cov, idx),
                rgbs=col, opacities=sub(op, idx))
        else:
            ps = psize * (1.6 if group == "ground" else 1.0)
            h = server.scene.add_point_cloud(
                f"/{group}", points=sub(xyz, idx), colors=(col * 255).astype(np.uint8),
                point_size=ps, point_shape="circle")
        h.visible = g_showg.value if group == "ground" else g_showe.value
        state[group] = h

    with server.gui.add_folder("ground highlight"):
        g_showg = server.gui.add_checkbox("show ground", True)
        g_showe = server.gui.add_checkbox("show environment", True)
        g_gcol = server.gui.add_dropdown("ground color", options=[*COLOR_TABLE, "original"], initial_value="magenta")
        g_ecol = server.gui.add_dropdown("environment", options=["gray", "original"], initial_value="gray")
        g_up = server.gui.add_dropdown("up axis", options=["+z", "-z", "+y", "-y", "+x", "-x"], initial_value=args.up)
        server.gui.add_text("ground gaussians", initial_value=f"{len(idx_g):,}", disabled=True)
        server.gui.add_text("environment", initial_value=f"{len(idx_e):,}", disabled=True)

    draw("env")
    draw("ground")

    @g_showg.on_update
    def _(_):
        if state["ground"] is not None:
            state["ground"].visible = g_showg.value

    @g_showe.on_update
    def _(_):
        if state["env"] is not None:
            state["env"].visible = g_showe.value

    @g_gcol.on_update
    def _(_):
        draw("ground")

    @g_ecol.on_update
    def _(_):
        draw("env")

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
