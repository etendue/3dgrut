"""E2.1: side-by-side raw|fixed montage for a few frames per mode."""
import argparse, json, os
import torch, torchvision


def _load(path):
    return torchvision.io.read_image(path).float().div(255.0)[:3]


def montage(raw_dir, fixed_dir, mode, out_path, n=4):
    with open(os.path.join(raw_dir, mode, "frames_map.json")) as f:
        fmap = json.load(f)
    keys = list(sorted(fmap.items()))
    step = max(1, len(keys) // n)
    rows = []
    for key, rel in keys[::step][:n]:
        r = _load(os.path.join(raw_dir, mode, rel))
        x = _load(os.path.join(fixed_dir, mode, rel))
        rows.append(torch.cat([r, x], dim=2))     # concat along width (raw | fixed)
    grid = torch.cat(rows, dim=1)                  # stack rows along height
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torchvision.utils.save_image(grid, out_path)
    print(f"montage {mode} -> {out_path} ({len(rows)} rows raw|fixed)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--fixed-dir", required=True)
    ap.add_argument("--modes", nargs="+", default=["lateral_3m", "lateral_6m"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    for m in a.modes:
        montage(a.raw_dir, a.fixed_dir, m, os.path.join(a.out_dir, f"montage_{m}.png"), a.n)


if __name__ == "__main__":
    main()
