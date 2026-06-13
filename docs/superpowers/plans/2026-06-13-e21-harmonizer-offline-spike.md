# E2.1 Harmonizer 离线修复 spike — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 量化 DiffusionHarmonizer（三代）对 baseline 在 lateral_3m/6m 渲染帧的离线修复增益，用 E1 全套指标（lane warp / NTA-IoU / FID-KID）做修复前/后对比 + 目视，产出 E2.2 go/no-go 判据。

**Architecture:** 解耦三段式（spec §2）：① render.py novel-save 增强 → 把 baseline 全部 novel 帧 + timestamp `frames_map` 落盘（raw）；② 独立批修复脚本经 socket IPC 调 inceptio 现成 `harmonizer_server`（nontemporal）逐帧修复 + color_transfer → fixed；③ 复用 `eval_frames_dir.py` 对 raw（交叉验证）与 fixed 评同口径指标 → 对比表。零训练代码改动（trainer/difix.py/yaml 不碰）；render.py 仅 eval-loop 存帧增量。

**Tech Stack:** PyTorch / torchvision；socket length-prefixed IPC（`>Q` + `torch.save` dict）；kornia color_transfer；inceptio RTX 4090 + `harmonizer-cosmos-env` 容器；git worktree（CLAUDE.md inceptio 工作流）。

**Spec:** [docs/superpowers/specs/2026-06-13-e21-harmonizer-offline-spike-design.md](../specs/2026-06-13-e21-harmonizer-offline-spike-design.md)

**已确认接口事实（探查于 2026-06-13，写 step code 的依据）：**
- `eval_frames_dir.resolve_pred_path`（scripts/eval_frames_dir.py:60-80）：键优先级 `ts:<camera_id>:<timestamp_us>` → `<camera_id>:<frame_idx>` → `<frames_dir>/<camera_id>/<frame_idx:06d>.png`。本 dataset `frame_idx` 全 -1（E0.4 实证）→ **必须走 timestamp frames_map**。
- eval 取键：`cam = getattr(batch,"camera_id")`、`timestamp_us=int(getattr(batch,"timestamp_us",-1))`（eval_frames_dir.py:139,149）。
- render.py novel loop 的 `gpu_batch` **同样有** `camera_id`（render.py:656）与 `timestamp_us`（render.py:859）→ 同款取法保证键对齐。
- render.py 现存帧：`ours_<step>/novel_view/<mode>/{iteration:05d}.png`，仅前 `novel_save_first_n=5` 帧（render.py:531,887-895）；扁平、无 camera/ts 元数据 → 需改。
- socket server（inceptio `~/work/nurec_e0/e07/ipc/harmonizer_server.py`）：收 `{input:(h·w,3) flatten RGB, img_size:[h,w]}` length-prefixed → 返回 `(h·w,3)` 修复 tensor；nontemporal V=1；端口 env `HARMONIZER_PORT`（smoke 用 59488，正式可任选）。
- client + color_transfer 范本：inceptio `~/work/nurec_e0/e07/ipc/model_ipc.py`（`_ipc_fixer_call` + `color_transfer` Reinhard/kornia，已读全）。
- baseline metrics 锚（v4_plan §1.3，用于 Task 3 交叉验证 + Task 4 对比表「修复前」列）：lane grad_corr@3m **0.384** / @6m **0.303**；NTA@3m **0.096** / @6m **0.062**；FID@3m **168** / @6m **193**。

---

## File Structure

| 文件 | 动作 | 责任 |
|---|---|---|
| `threedgrut/render.py` | 改（eval-loop 增量） | novel-save：`--novel-save-n`（默认 5，-1=全部）+ 存帧按 `<mode>/<camera_id>/<save_idx:06d>.png` + 累积写 `<mode>/frames_map.json`（`ts:<cam>:<ts>` 键） |
| `threedgrut/utils/novel_view.py` | 改（加纯函数） | `novel_frame_key(camera_id, timestamp_us)` + `novel_frame_relpath(camera_id, save_idx)` — 对齐键/路径单一来源，render.py 与测试共用 |
| `scripts/e21_harmonizer_batch_fix.py` | 建 | 读 raw 帧目录 + frames_map → socket client 逐帧修复 → color_transfer(fixed, raw) → 写 fixed 帧 + copy frames_map |
| `scripts/e21_compare_metrics.py` | 建 | 读 metrics_raw/fixed json + 锚 → 修复前/后对比表（markdown + 判别摘要） |
| `scripts/e21_visual_montage.py` | 建 | raw \| fixed 并排抽帧拼图（重伪影区） |
| `tests/test_e21_frame_align.py` | 建 | `novel_frame_key/relpath` 纯函数 TDD（Mac） |
| `tests/test_e21_ipc_client.py` | 建 | socket client 协议往返 TDD（mock echo server，Mac，纯 CPU） |
| `tests/test_e21_compare.py` | 建 | 对比表生成纯函数 TDD（Mac） |

**测试分层**：纯函数测试（frame_align / ipc_client / compare）在 **Mac pytest** 跑（无 GPU）；render-save 存帧、harmonizer 批修复、eval_frames_dir 评测是 **inceptio GPU smoke**（命令 + 输出验证，非 pytest）。

---

## Task 0: E2.1 worktree + baseline ckpt 锁定 + harmonizer server smoke

**Files:** 无代码改动（环境就绪 + 事实锁定）。

- [ ] **Step 1: Mac push 分支到 inceptio**

```bash
cd /Users/etendue/repo/3dgrut2
git remote get-url inceptio >/dev/null 2>&1 || git remote add inceptio inceptio:/home/inceptio/repo/3dgrut2
git push inceptio e21-harmonizer-spike:e21-harmonizer-spike
```
Expected: `* [new branch] e21-harmonizer-spike -> e21-harmonizer-spike`（或 up-to-date）。

- [ ] **Step 2: inceptio 建 worktree + 补 submodule**

```bash
ssh inceptio 'cd ~/repo/3dgrut2 && git worktree prune && git worktree add ~/repo/3dgrut2-wt/e21 e21-harmonizer-spike 2>&1 | tail -3'
ssh inceptio 'cd ~/repo/3dgrut2; WT=~/repo/3dgrut2-wt/e21; for p in $(git config --file .gitmodules --get-regexp path | cut -d" " -f2); do rsync -a ~/repo/3dgrut2/$p/ $WT/$p/; done; echo submodules-synced'
ssh inceptio 'cd ~/repo/3dgrut2-wt/e21 && git log --oneline -1'
```
Expected: 末条 = `60ef21c docs(spec): E2.1 ...`（= Mac 分支 head，注：后续 task commit 后会前移）。

- [ ] **Step 3: 锁定 baseline ckpt（比对 gap 表数字）**

逐个候选读 metrics.json 的 lane grad_corr，匹配锚值 0.384@3m 的那个即 baseline。先列候选的 interp lane grad_corr（应 ≈ 0.6931 门锚的非-B3 配方）：
```bash
ssh inceptio 'for d in ~/work/output/v3_kpi_sym5cam_30k ~/work/output/p1_2_runB_fix_30k ~/work/output/v3_base_scratch30k_lam01 ~/work/output/baseline_30k_novel; do echo "=== $d ==="; python3 -c "import json,glob; f=glob.glob(\"$d/**/metrics.json\",recursive=True); print(f[:1]); d=json.load(open(f[0])) if f else {}; print({k:round(v,4) for k,v in d.items() if \"lane_grad_corr\" in k or \"cc_psnr_masked\"==k})" 2>/dev/null; done'
```
Expected: 找到 interp `mean_lane_grad_corr`≈0.69（非 B3 的 0.739）且 `cc_psnr_masked`≈25.79 的目录。**把命中目录的 `ckpt_last.pt` 绝对路径记为 `$BASELINE_CKPT`，写进本 plan 顶部 + spec §7**。若无 novel-档历史 metrics 可辨，回退判据：baseline = 与 B3(`p31b3`)/aniso20(`p31aniso20`) 同期、不开 lane loss 的 depth-off 30k run（查 run config）。

- [ ] **Step 4: harmonizer server 起 + e2e smoke（确认 socket 通）**

```bash
ssh inceptio 'docker run -d --rm --name e21_harmonizer_server --gpus all \
  -e HARMONIZER_PORT=59489 --network host \
  -v ~/repo/harmonizer/src:/work/src -v ~/repo/harmonizer/models:/work/models \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/work/nurec_e0/e07/ipc:/shared \
  --entrypoint python harmonizer-cosmos-env:latest /shared/harmonizer_server.py'
# 轮询 READY（模型加载 ~3 min）
ssh inceptio 'for i in $(seq 1 40); do docker logs e21_harmonizer_server 2>&1 | grep -q "READY listening" && echo READY && break; sleep 10; done; docker logs --tail 5 e21_harmonizer_server'
```
Expected: `READY listening 127.0.0.1:59489`。**保留此 server 容器供 Task 2**（GPU 占用 ~8GB）。若端口/挂载与 E0.7 launch 脚本不同，以 `~/work/nurec_e0/e07/launch_harmonizer_train.sh` 内的实际挂载为准对齐。

- [ ] **Step 5: Commit（锁定事实入档）**

```bash
# 在 Mac 编辑 plan 顶部 + spec §7 填入 $BASELINE_CKPT 实际路径后：
git add docs/superpowers/plans/2026-06-13-e21-harmonizer-offline-spike.md docs/superpowers/specs/2026-06-13-e21-harmonizer-offline-spike-design.md
git commit -m "chore(E2.1): lock baseline ckpt path + harmonizer server smoke OK"
```

---

## Task 1: render.py novel-save 增强（存全部 + timestamp frames_map）

**Files:**
- Modify: `threedgrut/utils/novel_view.py`（加两个纯函数）
- Modify: `threedgrut/render.py`（存帧逻辑 + CLI 开关 + 传参链）
- Test: `tests/test_e21_frame_align.py`

- [ ] **Step 1: 写失败测试（对齐键/路径纯函数）**

`tests/test_e21_frame_align.py`:
```python
from threedgrut.utils.novel_view import novel_frame_key, novel_frame_relpath


def test_frame_key_matches_eval_frames_dir_format():
    # eval_frames_dir.resolve_pred_path builds: ts:<camera_id>:<timestamp_us>
    assert novel_frame_key("camera_front_wide", 1717000000123456) == \
        "ts:camera_front_wide:1717000000123456"


def test_frame_key_casts_timestamp_to_int():
    assert novel_frame_key("cam_x", 100.0) == "ts:cam_x:100"


def test_frame_relpath_is_camera_subdir_zero_padded():
    assert novel_frame_relpath("camera_front_wide", 7) == "camera_front_wide/000007.png"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_e21_frame_align.py -v`
Expected: FAIL — `ImportError: cannot import name 'novel_frame_key'`.

- [ ] **Step 3: 实现纯函数**

在 `threedgrut/utils/novel_view.py` 末尾追加：
```python
def novel_frame_key(camera_id: str, timestamp_us) -> str:
    """E2.1 frame-alignment key, must match eval_frames_dir.resolve_pred_path's
    ``ts:<camera_id>:<timestamp_us>`` join key (NCore batches carry no frame_idx)."""
    return f"ts:{camera_id}:{int(timestamp_us)}"


def novel_frame_relpath(camera_id: str, save_idx: int) -> str:
    """Per-camera subdir, 6-digit zero-padded — matches eval_frames_dir fallback layout."""
    return f"{camera_id}/{int(save_idx):06d}.png"
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_e21_frame_align.py -v`
Expected: 3 passed.

- [ ] **Step 5: render.py — 加 `novel_save_n` 构造参数 + 传参链**

`threedgrut/render.py`，在 `RenderAll`（或同名 eval 类）构造签名（render.py:92 区）把 `novel_fid=False,` 行后加：
```python
        novel_save_n: int = 5,
```
构造体（render.py:116 区，`self.novel_fid = bool(novel_fid)` 后）加：
```python
        # E2.1: how many novel-view frames to persist (-1 = all). Default 5
        # preserves the historical visual-sample behaviour byte-for-byte.
        self.novel_save_n = int(novel_save_n)
```
然后在 **每一处** `from_checkpoint` / 工厂转发（render.py:173-178 / 384-386 / 400-402 / 426-428 四处，搜 `novel_fid=novel_fid,`）紧跟着加一行 `novel_save_n=novel_save_n,`，并给对应外层函数签名加 `novel_save_n: int = 5,`（与 `novel_fid` 完全平行）。

- [ ] **Step 6: render.py — 存帧逻辑改用全部 + 对齐命名 + frames_map**

render.py:531 `novel_save_first_n = 5 if self.novel_view else 0` 改为：
```python
        novel_save_first_n = (
            (self.novel_save_n if self.novel_save_n >= 0 else 10**9)
            if self.novel_view else 0
        )
        # E2.1: per-mode {ts:<cam>:<ts> -> relpath} maps, written after the loop.
        novel_frames_map: dict[str, dict] = {m: {} for m in NOVEL_VIEW_MODES} if self.novel_view else {}
        novel_save_counter: dict[str, int] = {m: 0 for m in NOVEL_VIEW_MODES} if self.novel_view else {}
```
render.py:887-895 存帧块改为（用 import 的纯函数 + per-camera 子目录 + 累积 map）：
```python
                    if iteration < novel_save_first_n:
                        from threedgrut.utils.novel_view import (
                            novel_frame_key, novel_frame_relpath,
                        )
                        _cam = str(getattr(gpu_batch, "camera_id", "cam0"))
                        _ts = int(getattr(gpu_batch, "timestamp_us", -1))
                        _sidx = novel_save_counter[mode]
                        novel_save_counter[mode] += 1
                        _rel = novel_frame_relpath(_cam, _sidx)
                        _dst = os.path.join(
                            self.out_dir, f"ours_{int(self.global_step)}",
                            "novel_view", mode, _rel,
                        )
                        os.makedirs(os.path.dirname(_dst), exist_ok=True)
                        torchvision.utils.save_image(
                            pred_novel.squeeze(0).permute(2, 0, 1), _dst,
                        )
                        if _ts >= 0:
                            novel_frames_map[mode][novel_frame_key(_cam, _ts)] = _rel
```
在 novel loop 结束后（与现有 metrics 汇总同级，`if self.novel_view:` 块内）写出每 mode 的 frames_map：
```python
        if self.novel_view:
            import json as _json
            for _m, _fm in novel_frames_map.items():
                if _fm:
                    _mp = os.path.join(self.out_dir, f"ours_{int(self.global_step)}",
                                       "novel_view", _m, "frames_map.json")
                    with open(_mp, "w") as _f:
                        _json.dump(_fm, _f)
```
注：render.py:575-581 现有的“预建 `novel_view/<mode>/` 空目录”块保留无害（存帧时按 camera 子目录 makedirs）。

- [ ] **Step 7: render.py — CLI 暴露 `--novel-save-n`**

找到 render.py 的 CLI 入口（与 `--dataset-cameras` 注册同处；该开关在 render.py 内通过 `dataset_cameras` 参数链贯通，照其样式）。在该 parser 加：
```python
    parser.add_argument("--novel-save-n", type=int, default=5,
                        help="E2.1: # novel frames to save per mode (-1=all). "
                             "Default 5 = historical visual-sample behaviour.")
```
并把 `args.novel_save_n` 接到顶层 `RenderAll`/`from_checkpoint` 调用（与 `dataset_cameras=args.dataset_cameras` 平行）。若 render.py 走 hydra 而非 argparse，则在其 entry 读 `conf.get("novel_save_n", 5)` 转发——按入口实际机制二选一。

- [ ] **Step 8: 提交代码（纯函数已绿；GPU smoke 在 Step 9）**

```bash
git add threedgrut/utils/novel_view.py threedgrut/render.py tests/test_e21_frame_align.py
git commit -m "feat(E2.1): render.py novel-save-n + timestamp frames_map for offline fix"
git push inceptio e21-harmonizer-spike
```

- [ ] **Step 9: inceptio GPU smoke — 渲 baseline 全部 novel 帧 + frames_map**

```bash
ssh inceptio 'cd ~/repo/3dgrut2-wt/e21 && git pull --ff-only 2>&1 | tail -2'
ssh inceptio 'export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH \
  && export CUDA_VISIBLE_DEVICES=0 && cd ~/repo/3dgrut2-wt/e21 \
  && python -m threedgrut.render --checkpoint <$BASELINE_CKPT> \
       --path ~/work/data/9ae151dc/pai_9ae151dc-e87b-41a7-8e85-71772f9603d7.json \
       --out-dir ~/work/e21/raw --novel-view --novel-fid --novel-save-n -1 \
       2>&1 | tee /tmp/e21_render_raw.log | tail -20'
```
注：`--novel-view`/`--out-dir`/`--checkpoint` 等开关名以 render.py 实际 CLI 为准（Step 7 已对齐）。
Expected: 退出 0；`ours_<step>/novel_view/lateral_3m/` 与 `lateral_6m/` 下出现按 `<camera_id>/NNNNNN.png` 组织的帧 + 各一份 `frames_map.json`。
验证落盘：
```bash
ssh inceptio 'D=$(ls -d ~/work/e21/raw/*/ours_*/novel_view 2>/dev/null | head -1); echo "$D"; for m in lateral_3m lateral_6m; do echo "-- $m --"; find $D/$m -name "*.png" | wc -l; python3 -c "import json;print(\"map keys:\",len(json.load(open(\"$D/$m/frames_map.json\"))))"; done'
```
Expected: 每档 png 数 = frames_map 键数 = 全 5 相机 val 帧总数（≈375，以实际 val split 为准；**不是 5**）。若仍是 5 → `--novel-save-n -1` 没接通，回 Step 5/7。

---

## Task 2: harmonizer 批修复脚本（socket client + color_transfer）

**Files:**
- Create: `scripts/e21_harmonizer_batch_fix.py`
- Test: `tests/test_e21_ipc_client.py`

- [ ] **Step 1: 写失败测试（socket 协议往返，mock echo server）**

`tests/test_e21_ipc_client.py`:
```python
import io, socket, struct, threading
import torch
from scripts.e21_harmonizer_batch_fix import harmonizer_fix_frame


def _echo_server(port, ready):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port)); srv.listen(1); ready.set()
    c, _ = srv.accept()
    n = struct.unpack(">Q", _recvall(c, 8))[0]
    d = torch.load(io.BytesIO(_recvall(c, n)), weights_only=False)
    out = d["input"]                       # echo back unchanged (h*w, 3)
    bio = io.BytesIO(); torch.save(out, bio); p = bio.getvalue()
    c.sendall(struct.pack(">Q", len(p)) + p); c.close(); srv.close()


def _recvall(s, n):
    b = b""
    while len(b) < n:
        d = s.recv(n - len(b)); b += d
    return b


def test_fix_frame_roundtrip_shape_and_values():
    port = 59600
    ready = threading.Event()
    threading.Thread(target=_echo_server, args=(port, ready), daemon=True).start()
    ready.wait(timeout=5)
    img = torch.rand(8, 12, 3)             # (H, W, 3) float [0,1]
    out = harmonizer_fix_frame(img, host="127.0.0.1", port=port)
    assert out.shape == (8, 12, 3)
    assert torch.allclose(out, img, atol=1e-5)   # echo server returns input
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_e21_ipc_client.py -v`
Expected: FAIL — `ModuleNotFoundError: scripts.e21_harmonizer_batch_fix`.

- [ ] **Step 3: 实现批修复脚本（client + color_transfer + 批处理）**

`scripts/e21_harmonizer_batch_fix.py`:
```python
"""E2.1: batch-fix baseline novel frames through the Harmonizer IPC server.

Reads <raw_dir>/<mode>/frames_map.json + frames, sends each (H,W,3) RGB to the
harmonizer_server (nontemporal, socket length-prefixed protocol), applies
Reinhard color_transfer(fixed -> raw) to match nre DifixModel behaviour, writes
fixed frames to <fixed_dir>/<mode>/ with an identical frames_map.json.
"""
import argparse, io, json, os, shutil, socket, struct
import torch
import torchvision

try:
    import kornia
except ImportError:
    kornia = None


def _recvall(s, n):
    b = b""
    while len(b) < n:
        d = s.recv(n - len(b))
        if not d:
            raise EOFError("harmonizer server closed")
        b += d
    return b


def harmonizer_fix_frame(img_hw3: torch.Tensor, host: str, port: int) -> torch.Tensor:
    """img_hw3: (H,W,3) float[0,1] CPU -> repaired (H,W,3). Protocol mirrors
    harmonizer_server.py: {input:(h*w,3), img_size:[h,w]} length-prefixed."""
    H, W, _ = img_hw3.shape
    inp = img_hw3.reshape(H * W, 3).contiguous()
    bio = io.BytesIO()
    torch.save({"input": inp.cpu(), "img_size": (int(H), int(W))}, bio)
    p = bio.getvalue()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    s.sendall(struct.pack(">Q", len(p)) + p)
    n = struct.unpack(">Q", _recvall(s, 8))[0]
    out = torch.load(io.BytesIO(_recvall(s, n)), weights_only=False)
    s.close()
    return out.reshape(H, W, 3).float()


def color_transfer(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Reinhard et al. — move source(fixed) color stats onto target(raw). (H,W,3)."""
    if kornia is None:
        return source
    src = kornia.color.rgb_to_lab(source.permute(2, 0, 1)).permute(1, 2, 0)
    tgt = kornia.color.rgb_to_lab(target.permute(2, 0, 1)).permute(1, 2, 0)
    sm = src.reshape(-1, 1, 3).mean(0, keepdim=True); ss = src.reshape(-1, 1, 3).std(0, keepdim=True)
    tm = tgt.reshape(-1, 1, 3).mean(0, keepdim=True); ts = tgt.reshape(-1, 1, 3).std(0, keepdim=True)
    lab = (src - sm) * (ts / (ss + 1e-8)) + tm
    lab = lab.clamp(-128, 127)
    return kornia.color.lab_to_rgb(lab.permute(2, 0, 1)).permute(1, 2, 0).clamp(0, 1)


def _load_img(path):
    return torchvision.io.read_image(path).float().div(255.0)[:3].permute(1, 2, 0)


def fix_mode(raw_dir, fixed_dir, mode, host, port, do_ct=True):
    src_root = os.path.join(raw_dir, mode)
    dst_root = os.path.join(fixed_dir, mode)
    with open(os.path.join(src_root, "frames_map.json")) as f:
        fmap = json.load(f)
    os.makedirs(dst_root, exist_ok=True)
    for i, (key, rel) in enumerate(sorted(fmap.items())):
        raw = _load_img(os.path.join(src_root, rel))
        fixed = harmonizer_fix_frame(raw, host, port)
        if do_ct:
            fixed = color_transfer(fixed, raw)
        dst = os.path.join(dst_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        torchvision.utils.save_image(fixed.permute(2, 0, 1).clamp(0, 1), dst)
        if (i + 1) % 50 == 0:
            print(f"[{mode}] {i + 1}/{len(fmap)}", flush=True)
    shutil.copy(os.path.join(src_root, "frames_map.json"),
                os.path.join(dst_root, "frames_map.json"))
    print(f"[{mode}] done {len(fmap)} frames -> {dst_root}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="<.../ours_N/novel_view>")
    ap.add_argument("--fixed-dir", required=True)
    ap.add_argument("--modes", nargs="+", default=["lateral_3m", "lateral_6m"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=59489)
    ap.add_argument("--no-color-transfer", action="store_true")
    a = ap.parse_args()
    for m in a.modes:
        fix_mode(a.raw_dir, a.fixed_dir, m, a.host, a.port, do_ct=not a.no_color_transfer)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_e21_ipc_client.py -v`
Expected: 1 passed（echo server 往返，reshape 正确）。

- [ ] **Step 5: Commit**

```bash
git add scripts/e21_harmonizer_batch_fix.py tests/test_e21_ipc_client.py
git commit -m "feat(E2.1): harmonizer batch-fix script (socket client + color_transfer)"
git push inceptio e21-harmonizer-spike
```

- [ ] **Step 6: inceptio GPU smoke — 单帧真修复（确认 server+client+CT 全链）**

```bash
ssh inceptio 'cd ~/repo/3dgrut2-wt/e21 && git pull --ff-only 2>&1 | tail -1'
ssh inceptio 'export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH && cd ~/repo/3dgrut2-wt/e21 \
  && python -c "
import torch, scripts.e21_harmonizer_batch_fix as m
img = torch.rand(512, 896, 3)
out = m.harmonizer_fix_frame(img, \"127.0.0.1\", 59489)
print(\"fixed\", tuple(out.shape), float(out.min()), float(out.max()))
ct = m.color_transfer(out, img); print(\"ct\", tuple(ct.shape))
"'
```
Expected: `fixed (512, 896, 3) ...` 范围 [0,1] 附近；`ct (512, 896, 3)`。失败（connection refused）→ Task 0 Step 4 的 server 容器没活，重起。

- [ ] **Step 7: inceptio GPU — 批修复全部 raw 帧**

```bash
ssh inceptio 'export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH && cd ~/repo/3dgrut2-wt/e21 \
  && RAW=$(ls -d ~/work/e21/raw/*/ours_*/novel_view | head -1) \
  && python scripts/e21_harmonizer_batch_fix.py --raw-dir $RAW \
       --fixed-dir ~/work/e21/fixed --port 59489 2>&1 | tee /tmp/e21_fix.log | tail -10'
```
Expected: 每档 `done <N> frames -> .../fixed/lateral_3m`；fixed 帧数 = raw 帧数，含 copy 的 frames_map.json。耗时 ~20-30 min（750 帧 × ~0.5-1s）。

---

## Task 3: eval_frames_dir 评 raw（交叉验证）+ fixed

**Files:** 无新代码（编排 + 验证）。

- [ ] **Step 1: 评 raw（交叉验证 ≈ E1 锚）— 两档**

```bash
ssh inceptio 'export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH && export CUDA_VISIBLE_DEVICES=0 && cd ~/repo/3dgrut2-wt/e21 \
  && RAW=$(ls -d ~/work/e21/raw/*/ours_*/novel_view | head -1) && mkdir -p ~/work/e21/evals \
  && for M in lateral_3m lateral_6m; do \
      python scripts/eval_frames_dir.py --checkpoint <$BASELINE_CKPT> \
        --path ~/work/data/9ae151dc/pai_9ae151dc-e87b-41a7-8e85-71772f9603d7.json \
        --frames-dir $RAW/$M --frames-map $RAW/$M/frames_map.json \
        --mode $M --lane --nta-iou --novel-fid \
        > ~/work/e21/evals/metrics_raw_$M.json 2>/tmp/e21_eval_raw_$M.log; \
      echo "raw $M done"; done'
```
注：`--novel-fid` 开关名以 eval_frames_dir.py 实际 argparse 为准（grep `add_argument` 确认；spec §2 已注 FID/KID 在该工具内）。
Expected: 两份 `metrics_raw_<M>.json`，含 `mean_novel_lane_grad_corr_<M>` / `mean_novel_nta_iou_<M>` / `mean_novel_fid_<M>`。

- [ ] **Step 2: 交叉验证 raw ≈ E1 锚（口径一致性 gate）**

```bash
ssh inceptio 'python3 -c "
import json
for M,gc,fid in [(\"lateral_3m\",0.384,168),(\"lateral_6m\",0.303,193)]:
    d=json.load(open(f\"/home/inceptio/work/e21/evals/metrics_raw_{M}.json\"))
    g=d.get(f\"mean_novel_lane_grad_corr_{M}\"); f=d.get(f\"mean_novel_fid_{M}\")
    print(M,\"grad_corr raw\",round(g,3),\"anchor\",gc,\"| fid raw\",round(f,1),\"anchor\",fid)
"'
```
Expected: raw grad_corr 与锚差 < ~0.02、FID 同量级。**若偏差大 → 暴露 render_all↔eval_frames_dir 路径口径暗差**：此时 Task 4 对比表「修复前」列改用 `metrics_raw_*`（自洽优先），并在 Done Log 记差异来源。否则「修复前」列引 E1 锚。

- [ ] **Step 3: 评 fixed — 两档**

```bash
ssh inceptio 'export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH && export CUDA_VISIBLE_DEVICES=0 && cd ~/repo/3dgrut2-wt/e21 \
  && for M in lateral_3m lateral_6m; do \
      python scripts/eval_frames_dir.py --checkpoint <$BASELINE_CKPT> \
        --path ~/work/data/9ae151dc/pai_9ae151dc-e87b-41a7-8e85-71772f9603d7.json \
        --frames-dir ~/work/e21/fixed/$M --frames-map ~/work/e21/fixed/$M/frames_map.json \
        --mode $M --lane --nta-iou --novel-fid \
        > ~/work/e21/evals/metrics_fixed_$M.json 2>/tmp/e21_eval_fixed_$M.log; \
      echo "fixed $M done"; done'
```
Expected: 两份 `metrics_fixed_<M>.json`，新 key 齐全（CLAUDE.md §B6：没见到新 key 不许标 ✅）。

---

## Task 4: 对比表生成（纯函数 TDD）+ 目视拼图

**Files:**
- Create: `scripts/e21_compare_metrics.py`
- Create: `scripts/e21_visual_montage.py`
- Test: `tests/test_e21_compare.py`

- [ ] **Step 1: 写失败测试（对比表纯函数）**

`tests/test_e21_compare.py`:
```python
from scripts.e21_compare_metrics import compare_metric, build_table_rows


def test_compare_metric_delta_and_direction():
    row = compare_metric("FID", "lateral_3m", before=168.0, after=120.0, higher_is_better=False)
    assert row["delta"] == -48.0
    assert row["improved"] is True            # lower FID is better


def test_compare_metric_grad_corr_higher_is_better():
    row = compare_metric("lane_grad_corr", "lateral_6m", before=0.303, after=0.300,
                         higher_is_better=True)
    assert row["delta"] == -0.003
    assert row["improved"] is False


def test_build_table_rows_handles_missing_after():
    rows = build_table_rows(
        before={"mean_novel_fid_lateral_3m": 168.0},
        after={},                              # fixed eval failed for this key
        modes=["lateral_3m"],
    )
    fid = [r for r in rows if r["metric"] == "FID" and r["mode"] == "lateral_3m"][0]
    assert fid["after"] is None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_e21_compare.py -v`
Expected: FAIL — `ModuleNotFoundError: scripts.e21_compare_metrics`.

- [ ] **Step 3: 实现对比表脚本**

`scripts/e21_compare_metrics.py`:
```python
"""E2.1: build a before/after comparison table from raw/fixed metrics jsons."""
import argparse, json

# (display, key_suffix, higher_is_better)
METRICS = [
    ("lane_grad_corr", "mean_novel_lane_grad_corr", True),
    ("lane_band_psnr", "mean_novel_lane_band_psnr", True),
    ("NTA_IoU",        "mean_novel_nta_iou",        True),
    ("FID",            "mean_novel_fid",            False),
    ("KID",            "mean_novel_kid",            False),
]


def compare_metric(metric, mode, before, after, higher_is_better):
    delta = None if (before is None or after is None) else round(after - before, 4)
    improved = None
    if delta is not None:
        improved = (delta > 0) if higher_is_better else (delta < 0)
    return {"metric": metric, "mode": mode, "before": before, "after": after,
            "delta": delta, "improved": improved, "higher_is_better": higher_is_better}


def build_table_rows(before, after, modes):
    rows = []
    for disp, suf, hib in METRICS:
        for mode in modes:
            k = f"{suf}_{mode}"
            rows.append(compare_metric(disp, mode, before.get(k), after.get(k), hib))
    return rows


def _markdown(rows):
    out = ["| 指标 | 档 | 修复前 | 修复后 | Δ | 方向 |", "|---|---|---|---|---|---|"]
    for r in rows:
        arrow = "—" if r["improved"] is None else ("✅↑" if r["improved"] else "⚠️↓")
        b = "—" if r["before"] is None else f'{r["before"]:.3f}'
        a = "—" if r["after"] is None else f'{r["after"]:.3f}'
        d = "—" if r["delta"] is None else f'{r["delta"]:+.3f}'
        out.append(f'| {r["metric"]} | {r["mode"]} | {b} | {a} | {d} | {arrow} |')
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", nargs="+", required=True, help="raw/anchor metrics json(s)")
    ap.add_argument("--after", nargs="+", required=True, help="fixed metrics json(s)")
    ap.add_argument("--modes", nargs="+", default=["lateral_3m", "lateral_6m"])
    a = ap.parse_args()
    before = {}; after = {}
    for f in a.before:
        before.update(json.load(open(f)))
    for f in a.after:
        after.update(json.load(open(f)))
    rows = build_table_rows(before, after, a.modes)
    print(_markdown(rows))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_e21_compare.py -v`
Expected: 3 passed.

- [ ] **Step 5: 实现目视拼图脚本（smoke 验证，无 TDD）**

`scripts/e21_visual_montage.py`:
```python
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
        rows.append(torch.cat([r, x], dim=2))     # concat along width
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
```

- [ ] **Step 6: Commit + 生成对比表与拼图**

```bash
git add scripts/e21_compare_metrics.py scripts/e21_visual_montage.py tests/test_e21_compare.py
git commit -m "feat(E2.1): before/after comparison table + visual montage"
git push inceptio e21-harmonizer-spike
ssh inceptio 'cd ~/repo/3dgrut2-wt/e21 && git pull --ff-only 2>&1 | tail -1'
# 对比表（修复前=E1锚或raw，按 Task3 Step2 gate 决定）
ssh inceptio 'export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH && cd ~/repo/3dgrut2-wt/e21 \
  && python scripts/e21_compare_metrics.py \
       --before ~/work/e21/evals/metrics_raw_lateral_3m.json ~/work/e21/evals/metrics_raw_lateral_6m.json \
       --after  ~/work/e21/evals/metrics_fixed_lateral_3m.json ~/work/e21/evals/metrics_fixed_lateral_6m.json \
       | tee ~/work/e21/comparison_table.md'
# 拼图
ssh inceptio 'export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH && cd ~/repo/3dgrut2-wt/e21 \
  && RAW=$(ls -d ~/work/e21/raw/*/ours_*/novel_view | head -1) \
  && python scripts/e21_visual_montage.py --raw-dir $RAW --fixed-dir ~/work/e21/fixed \
       --out-dir ~/work/e21/montage'
```
Expected: `comparison_table.md` 打印 + `montage/montage_lateral_{3m,6m}.png` 生成。把表 + 拼图取回 Mac（`scp` 或贴入 Done Log）。

---

## Task 5: 判别 + 回填文档 + 收尾

**Files:**
- Modify: `v4_plan.md`（§1.3 gap 表新列 + §5 Done Log + §1.2 状态 + §1.1 看板）
- Modify: `docs/superpowers/specs/2026-06-13-e21-harmonizer-offline-spike-design.md`（§1 成功判据回填实测结论）

- [ ] **Step 1: 写 E2.2 go/no-go 判别**

依 spec §1 判据，对照 `comparison_table.md` + 拼图：
- **go**（投 E2.2）：FID/KID 显著降（参 E0.2 论文量级）**且** lane_grad_corr/NTA 不大幅退（≤ ~0.02 / 噪声内）+ 目视去伪影显著、无异物。
- **no-go / 转 E2.4**：Harmonizer 引入异物（目视新增结构）/ FID 不降 / 几何大幅退（R-v4.4 域差坐实）。
写成 3-5 句结论（含域差判别），备 Done Log。

- [ ] **Step 2: 回填 v4_plan.md §1.3 gap 表「E2.1 后」列**

把 lane_grad_corr / NTA / FID/KID @3m/6m 的修复后数字填入 gap 表对应行的「E2/E3 后」列（标注「E2.1 离线 Harmonizer」），interpolated 守护线一行不动（spike 不碰 interp）。

- [ ] **Step 3: 追加 §5 Done Log 条目**

格式：日期 + commit hash + 实测数（修复前/后对比表关键行）+ raw 交叉验证结论 + E2.2 判别 + 产物路径（`~/work/e21/{raw,fixed,evals,montage}`）。

- [ ] **Step 4: 更新 §1.2 任务表 + §1.1 看板 + §1.3 Phase 汇总**

§1.2：E2.1 状态 ⬜→✅，「改动/新增」填本 plan commit 短 hash。§1.1：E2.1 卡从 Backlog 移 Done。§1.3：E2 Phase「任务数 (Done/Total)」0/6→1/6。

- [ ] **Step 5: Mermaid 全角括号自查（CLAUDE.md 铁律）**

Run: `awk '/```mermaid/{i=1;next} /```/&&i{i=0;next} i&&/\(/{print FILENAME":"NR": "$0}' v4_plan.md`
Expected: 零输出（看板/依赖图内无半角 `(`）。有则改全角 `（）`。

- [ ] **Step 6: Commit 文档同步**

```bash
git add v4_plan.md docs/superpowers/specs/2026-06-13-e21-harmonizer-offline-spike-design.md
git commit -m "$(cat <<'EOF'
docs(plan): mark E2.1 done — Harmonizer 离线修复 spike 实测回填

<对比表关键行 + E2.2 go/no-go 一句话>

docs(plan): E2.1 gap 表「E2.1 后」列 + Done Log + 看板状态同步
EOF
)"
```

- [ ] **Step 7: 收尾 — 释放 server 显存**

```bash
ssh inceptio 'docker rm -f e21_harmonizer_server 2>/dev/null; echo cleaned'
```
worktree `~/repo/3dgrut2-wt/e21` 暂留（E2.2 可能复用渲帧/修复链）；E2.2 启动前若不需要再 `git worktree remove`。

- [ ] **Step 8: 全量 Mac pytest 回归（守护线）**

Run: `python -m pytest tests/test_e21_*.py -v`
Expected: 全绿（frame_align 3 + ipc_client 1 + compare 3 = 7 passed）。确认 E2.1 纯函数无回归。

---

## Self-Review（写完即查）

**Spec coverage（spec §→task）：**
- §1 目标/成功判据/预期管理 → Task 5 Step 1 判别。✅
- §2 数据流（渲→修→评→对比）→ Task 1（渲）/ Task 2（修）/ Task 3（评）/ Task 4（对比+拼图）。✅
- §3 nontemporal → Task 0 Step 4 起 harmonizer_server（其 nontemporal V=1）+ Task 2 client 不传时序。✅
- §4 覆盖（baseline / 3m+6m / 全5相机375帧）→ Task 1 Step 9（--novel-save-n -1 存全部）+ Task 3 两档。✅
- §5 修复前引锚 + raw 交叉验证 → Task 3 Step 2 gate。✅
- §6 产出（gap表/Done Log/拼图/判别）→ Task 5。✅
- §7 执行环境（worktree/baseline锁定/GPU）→ Task 0。✅
- §8 出界（不碰 difix/temporal/B3/Fixer/训练）→ 全 plan 未触及。✅

**Placeholder scan：** `<$BASELINE_CKPT>` 是 Task 0 Step 3 显式锁定的变量（非占位符，有解析步骤）；render.py CLI 开关名/`--novel-fid` 标注「以实际 argparse 为准」并给确认方式（非 placeholder，是带验证的具体指令）。无 TBD/TODO。

**Type consistency：** `novel_frame_key/novel_frame_relpath`（Task 1）↔ `harmonizer_fix_frame`/`color_transfer`/`fix_mode`（Task 2）↔ `compare_metric`/`build_table_rows`（Task 4）签名跨 task 一致；socket 协议字段 `{input, img_size}` 与 server 端、测试 echo server 三处一致。✅
