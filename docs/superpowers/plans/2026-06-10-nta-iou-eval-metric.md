# NTA-IoU 评测指标接入 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给现有 per-class 评测新增 NTA-IoU（Novel Trajectory Agent IoU）指标——在渲染图上用 2D 车辆检测器检车、与投影后的 GT cuboid 2D box 算 IoU，作为对车辆位姿/形状比 PSNR 更灵敏的标尺，写进 `metrics.json`。

**Architecture:** 新增一个纯函数模块 `threedgrut/model/nta_iou.py`（逐 track 投影取 2D AABB + box-IoU + 每个 GT 取最佳匹配），一个懒加载检测器封装 `threedgrut/model/vehicle_detector.py`，再把它们挂到 `threedgrut/render.py` 的 eval loop（与现有 `compute_class_psnr` 同一段，复用同样的 `active` tracks / `ftheta_params` / `T_w2c`）。检测器以协议（duck-typed）注入，使核心匹配逻辑能在 Mac 上用合成数据做 TDD，YOLO 只在 GPU smoke 时真正加载。

**Tech Stack:** PyTorch, torchvision.ops.box_iou, ultralytics(YOLO), 复用 `threedgrut/layers/dynamic_mask.py::project_cuboids_to_mask`。

---

## File Structure

- Create `threedgrut/model/nta_iou.py` — 纯函数：单 track 投影成 2D box、box IoU、对一帧算 NTA-IoU。无 GPU、无 YOLO 依赖，检测框作为入参传入。
- Create `threedgrut/model/vehicle_detector.py` — `VehicleDetector` 懒加载单例，包 ultralytics YOLO，输出车辆类 2D 框 `[M,4]`。唯一与 ultralytics 耦合的文件。
- Create `tests/test_nta_iou.py` — Mac pytest，合成 box 验证投影/IoU/匹配，注入 fake detector。
- Modify `threedgrut/render.py` — eval loop（约 :711–:755）调 NTA-IoU；metrics 聚合（约 :825–:949）写盘。
- Modify `requirements.txt` — 新增 `ultralytics`。

**关键复用事实（勘探所得）：**
- `render.py:711-726`：`ftheta_params = getattr(gpu_batch, "intrinsics_FThetaCameraModelParameters", None)`；`T_w2c = torch.linalg.inv(gpu_batch.T_to_world[0])`；`active = collect_active_tracks_for_frame(...)`，每个元素 `{"id","class","pose":[4,4],"size":[3]}`。
- `dynamic_mask.py:111-203`：`project_cuboids_to_mask(poses[T,4,4], sizes[T,3], K, T_w2c, H, W, ftheta_params=, device=) -> [H,W] bool`，内部 8 角点 → world → cam → FTheta/pinhole 投影 → AABB 填 mask。**这是避开 BUG-1 的正确投影**（BUG-1 在 viser 侧 `threedgrut_playground/utils/ftheta_projector.py`，本 plan 不碰它）。
- `pred_rgb` = `outputs["pred_rgb"]` `[1,H,W,3]`，值域 `[0,1]`（`render.py:519`）。
- metrics 写盘段 `render.py:825-949`，模式：`metrics_json["mean_xxx"]=...; metrics_json["xxx_n_records"]=...`。

---

## Task 0: 确认 GT track 的车辆类字符串（勘探 spike，不写代码）

**Files:** 无（只读）

- [ ] **Step 1: grep 一个真实 manifest 的 track class 取值**

Run（Mac，路径按本地数据集；A800 上对应 `/root/work/yusun/ncore-nurec/data/...`）：
```bash
python3 -c "import json,sys,collections; d=json.load(open(sys.argv[1])); print(collections.Counter(t.get('class') or t.get('category') for t in d.get('tracks',[])))" <path-to>/pai_<clip>.json
```
Expected: 打印各 class 计数，确认车辆类的确切字符串（预期是 `automobile` / `heavy_truck` / `bus` 一类，而非 `vehicle`）。

- [ ] **Step 2: 把确认到的车辆类写进常量**

把结果记到 `nta_iou.py` 的 `VEHICLE_TRACK_CLASSES`（Task 1 会建该文件）。若 manifest 用的是 `automobile/heavy_truck/bus`，常量即 `{"automobile","heavy_truck","bus","car","truck","vehicle"}`（多写几个别名不影响，只用于过滤）。

---

## Task 1: 纯函数模块 `nta_iou.py` —— 单 track 投影成 2D box

**Files:**
- Create: `threedgrut/model/nta_iou.py`
- Test: `tests/test_nta_iou.py`

- [ ] **Step 1: 写失败测试（投影一个正前方 cuboid 得到合理 2D box）**

```python
# tests/test_nta_iou.py
import torch
from threedgrut.model.nta_iou import project_track_to_2d_box

def _pinhole_K(fx=500.0, fy=500.0, cx=320.0, cy=240.0):
    return torch.tensor([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=torch.float32)

def test_project_track_front_center_box():
    # cuboid 在相机正前方 10m，尺寸 2x2x2，T_w2c=I（world==cam）
    pose = torch.eye(4); pose[2,3] = 10.0          # 中心 z=10
    size = torch.tensor([2.0, 2.0, 2.0])
    T_w2c = torch.eye(4)
    box = project_track_to_2d_box(pose, size, K=_pinhole_K(), ftheta_params=None,
                                  T_w2c=T_w2c, H=480, W=640)
    assert box is not None
    x1, y1, x2, y2 = box
    assert 0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480
    # 中心应接近主点 (320,240)
    assert abs((x1+x2)/2 - 320) < 30 and abs((y1+y2)/2 - 240) < 30

def test_project_track_behind_returns_none():
    pose = torch.eye(4); pose[2,3] = -10.0         # 相机背后
    size = torch.tensor([2.0,2.0,2.0])
    box = project_track_to_2d_box(pose, size, K=_pinhole_K(), ftheta_params=None,
                                  T_w2c=torch.eye(4), H=480, W=640)
    assert box is None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_nta_iou.py -k project -v`
Expected: FAIL（`ModuleNotFoundError` 或 `project_track_to_2d_box` 未定义）。

- [ ] **Step 3: 实现 `project_track_to_2d_box`（复用 project_cuboids_to_mask 逐 track 取 AABB）**

```python
# threedgrut/model/nta_iou.py
"""NTA-IoU (Novel Trajectory Agent IoU): 渲染图车辆检测框 vs 投影 GT cuboid 2D 框的 IoU。"""
from __future__ import annotations
import torch
from torchvision.ops import box_iou as _tv_box_iou
from threedgrut.layers.dynamic_mask import project_cuboids_to_mask

VEHICLE_TRACK_CLASSES = {"automobile", "heavy_truck", "bus", "car", "truck", "vehicle"}
# COCO: car=2, motorcycle=3, bus=5, truck=7
VEHICLE_COCO_IDS = (2, 3, 5, 7)


def project_track_to_2d_box(pose, size, K, ftheta_params, T_w2c, H, W):
    """单个 GT cuboid -> 轴对齐 2D box [x1,y1,x2,y2]（像素）。看不到则 None。"""
    device = pose.device if isinstance(pose, torch.Tensor) else "cpu"
    mask = project_cuboids_to_mask(
        pose.unsqueeze(0), size.unsqueeze(0), K, T_w2c, H, W,
        ftheta_params=ftheta_params, device=device,
    )  # [H,W] bool
    ys, xs = torch.where(mask)
    if ys.numel() == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_nta_iou.py -k project -v`
Expected: PASS（两个 test 都过）。如 `project_cuboids_to_mask` 的入参顺序与勘探记录不符，以源码 `dynamic_mask.py:111` 的真实签名为准微调本调用，并在测试里固定住。

- [ ] **Step 5: Commit**

```bash
git add threedgrut/model/nta_iou.py tests/test_nta_iou.py
git commit -m "feat(nta-iou): project single GT cuboid to 2D box (reuse project_cuboids_to_mask)"
```

---

## Task 2: 一帧 NTA-IoU 计算（每个 GT 取最佳匹配 IoU）

**Files:**
- Modify: `threedgrut/model/nta_iou.py`
- Test: `tests/test_nta_iou.py`

- [ ] **Step 1: 写失败测试（注入 fake detector，验证匹配与边界情形）**

```python
# 追加到 tests/test_nta_iou.py
from threedgrut.model.nta_iou import compute_frame_nta_iou

class _FakeDetector:
    def __init__(self, boxes): self._b = torch.tensor(boxes, dtype=torch.float32)
    def detect_vehicles(self, rgb_hw3_01): return self._b  # [M,4] xyxy

def _front_vehicle_track():
    pose = torch.eye(4); pose[2,3] = 10.0
    return {"id": 1, "class": "automobile", "pose": pose, "size": torch.tensor([2.0,2.0,2.0])}

def test_nta_iou_perfect_match():
    track = _front_vehicle_track()
    gt = project_track_to_2d_box(track["pose"], track["size"], _pinhole_K(), None, torch.eye(4), 480, 640)
    det = _FakeDetector([list(gt)])               # 检测框 == GT 框
    out = compute_frame_nta_iou(torch.zeros(480,640,3), [track], _FakeDetectorWrap := det,
                                K=_pinhole_K(), ftheta_params=None, T_w2c=torch.eye(4), H=480, W=640)
    assert out is not None and out["n_gt"] == 1
    assert out["mean_nta_iou"] > 0.99

def test_nta_iou_no_detection_scores_zero():
    track = _front_vehicle_track()
    out = compute_frame_nta_iou(torch.zeros(480,640,3), [track], _FakeDetector([]),
                                K=_pinhole_K(), ftheta_params=None, T_w2c=torch.eye(4), H=480, W=640)
    assert out["n_gt"] == 1 and out["n_det"] == 0 and out["mean_nta_iou"] == 0.0

def test_nta_iou_no_gt_vehicle_returns_none():
    ped = {"id": 9, "class": "pedestrian", "pose": torch.eye(4), "size": torch.tensor([1.,1.,2.])}
    out = compute_frame_nta_iou(torch.zeros(480,640,3), [ped], _FakeDetector([[0,0,10,10]]),
                                K=_pinhole_K(), ftheta_params=None, T_w2c=torch.eye(4), H=480, W=640)
    assert out is None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_nta_iou.py -k nta_iou -v`
Expected: FAIL（`compute_frame_nta_iou` 未定义）。

- [ ] **Step 3: 实现 `compute_frame_nta_iou`**

```python
# 追加到 threedgrut/model/nta_iou.py
def compute_frame_nta_iou(pred_rgb_hw3, active_tracks, detector, K, ftheta_params, T_w2c, H, W):
    """一帧 NTA-IoU。无 GT 车辆 -> None（该帧不计入）。返回 dict。"""
    gt_boxes = []
    for t in active_tracks:
        if str(t.get("class", "")).lower() not in VEHICLE_TRACK_CLASSES:
            continue
        b = project_track_to_2d_box(t["pose"].to(torch.float32), t["size"].to(torch.float32),
                                    K, ftheta_params, T_w2c, H, W)
        if b is not None:
            gt_boxes.append(b)
    if not gt_boxes:
        return None
    gt = torch.tensor(gt_boxes, dtype=torch.float32)
    det = detector.detect_vehicles(pred_rgb_hw3)
    if det is None or det.numel() == 0:
        return {"mean_nta_iou": 0.0, "n_gt": len(gt_boxes), "n_det": 0}
    iou = _tv_box_iou(gt, det.to(torch.float32))         # [G,M]
    best = iou.max(dim=1).values                          # 每个 GT 的最佳匹配 IoU
    return {"mean_nta_iou": float(best.mean()), "n_gt": len(gt_boxes), "n_det": int(det.shape[0])}
```

修正 Step 1 测试里 `_FakeDetectorWrap := det` 的笔误，直接传 `det`：
```python
    out = compute_frame_nta_iou(torch.zeros(480,640,3), [track], det,
                                K=_pinhole_K(), ftheta_params=None, T_w2c=torch.eye(4), H=480, W=640)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_nta_iou.py -v`
Expected: PASS（全部，含 Task 1 的）。

- [ ] **Step 5: Commit**

```bash
git add threedgrut/model/nta_iou.py tests/test_nta_iou.py
git commit -m "feat(nta-iou): per-frame NTA-IoU with best-match IoU per GT vehicle"
```

---

## Task 3: 车辆检测器封装（唯一耦合 ultralytics 的文件）

**Files:**
- Create: `threedgrut/model/vehicle_detector.py`
- Modify: `requirements.txt`

- [ ] **Step 1: 在 requirements.txt 增加依赖**

在 `requirements.txt` 末尾追加一行：
```
ultralytics>=8.2.0
```

- [ ] **Step 2: 实现懒加载检测器**

```python
# threedgrut/model/vehicle_detector.py
"""YOLO 车辆检测封装。唯一 import ultralytics 的文件，懒加载，eval 时单例复用。"""
from __future__ import annotations
import torch
from threedgrut.model.nta_iou import VEHICLE_COCO_IDS

_SINGLETON = None


class VehicleDetector:
    def __init__(self, weights: str = "yolov8m.pt", conf: float = 0.3, device: str = "cuda"):
        from ultralytics import YOLO            # 局部 import：未装时不拖累其余评测
        self.model = YOLO(weights)
        self.conf = conf
        self.device = device

    @torch.no_grad()
    def detect_vehicles(self, rgb_hw3_01):
        """rgb [H,W,3] in [0,1] -> 车辆类 2D 框 [M,4] xyxy(像素)。"""
        arr = (rgb_hw3_01.detach().clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        res = self.model.predict(arr, conf=self.conf, device=self.device, verbose=False)[0]
        b = res.boxes
        if b is None or b.shape[0] == 0:
            return torch.zeros((0, 4), dtype=torch.float32)
        cls = b.cls.to(torch.int64)
        keep = torch.zeros_like(cls, dtype=torch.bool)
        for cid in VEHICLE_COCO_IDS:
            keep |= cls == cid
        return b.xyxy[keep].float().cpu()


def get_vehicle_detector(weights="yolov8m.pt", conf=0.3, device="cuda"):
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = VehicleDetector(weights, conf, device)
    return _SINGLETON
```

- [ ] **Step 3: 装依赖并冒烟（GPU 机，A800 / inceptio）**

Run（inceptio）:
```bash
ssh inceptio 'export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH && cd ~/repo/3dgrut2 && pip install "ultralytics>=8.2.0" -q && python -c "from threedgrut.model.vehicle_detector import get_vehicle_detector; d=get_vehicle_detector(device=\"cpu\"); import torch; print(d.detect_vehicles(torch.rand(480,640,3)).shape)"'
```
Expected: 打印一个 `torch.Size([M, 4])`（M≥0，随机图多半为 0），无异常即代表权重可加载、车辆类过滤生效。首次会自动下载 `yolov8m.pt`。

- [ ] **Step 4: Commit**

```bash
git add threedgrut/model/vehicle_detector.py requirements.txt
git commit -m "feat(nta-iou): add lazy YOLO vehicle detector wrapper + ultralytics dep"
```

---

## Task 4: 接进 render.py eval loop + 写 metrics.json

**Files:**
- Modify: `threedgrut/render.py`（eval loop ~:711-755；metrics 聚合 ~:825-949）

- [ ] **Step 1: 在 eval loop 顶部初始化累加器 + 检测器开关**

在 render.py 累加器初始化处（与 `per_class_psnr = {}` 等同段，约 :390-468）加：
```python
nta_iou_records = []   # 每帧 dict {"mean_nta_iou","n_gt","n_det"}
_nta_detector = None
if getattr(self, "enable_nta_iou", True):
    try:
        from threedgrut.model.vehicle_detector import get_vehicle_detector
        _nta_detector = get_vehicle_detector(device=str(self.device))
    except Exception as e:
        print(f"[NTA-IoU] detector unavailable, skipping: {e}")
        _nta_detector = None
```

- [ ] **Step 2: 在 compute_class_psnr 同段调用 NTA-IoU**

在 `render.py:714-735`（已拿到 `active` / `T_w2c` / `ftheta_params` / `H_,W_` / `pred_rgb_full` 的那段）后插入：
```python
if _nta_detector is not None and active:
    from threedgrut.model.nta_iou import compute_frame_nta_iou
    K_eval = getattr(gpu_batch, "intrinsics_pinhole_K", None)   # FTheta 时为 None，走 ftheta_params
    nta = compute_frame_nta_iou(
        pred_rgb_full[0], active, _nta_detector,
        K=K_eval, ftheta_params=ftheta_params, T_w2c=T_w2c, H=H_, W=W_,
    )
    if nta is not None:
        nta_iou_records.append(nta)
```

- [ ] **Step 3: 在 metrics 聚合段写盘**

在 per-class sseg 写盘之后（约 `render.py:941-949`）追加：
```python
if nta_iou_records:
    _vals = [r["mean_nta_iou"] for r in nta_iou_records]
    metrics_json["mean_nta_iou"] = float(sum(_vals) / len(_vals))
    metrics_json["nta_iou_n_frames"] = int(len(_vals))
    table["mean_nta_iou"] = metrics_json["mean_nta_iou"]
```
（`table` 即现有 console 汇总表对象；若该 eval 段无 `table` 变量，仅写 `metrics_json`。）

- [ ] **Step 4: GPU smoke —— 跑一次短 eval，确认 metrics.json 出现 mean_nta_iou**

Run（inceptio，复用已有 ckpt；无则先按 CLAUDE.md 标准配方训 5k）:
```bash
ssh inceptio 'export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH && export CUDA_VISIBLE_DEVICES=0 && cd ~/repo/3dgrut2 && python render.py --checkpoint <ckpt.pt> --out_dir ~/work/output/nta_smoke 2>&1 | tail -40'
```
然后：
```bash
ssh inceptio 'python3 -c "import json; m=json.load(open(\"/home/inceptio/work/output/nta_smoke/ours_<step>/metrics.json\")); print(\"mean_nta_iou\",m.get(\"mean_nta_iou\"),\"n_frames\",m.get(\"nta_iou_n_frames\"))"'
```
Expected: 打印 `mean_nta_iou` 为 (0,1] 区间的浮点、`n_frames>0`。CLAUDE.md § B 清单第 6 条：metrics.json 未见新 key 则状态保持 🟡 不可标 ✅。

- [ ] **Step 5: Commit**

```bash
git add threedgrut/render.py
git commit -m "feat(nta-iou): wire NTA-IoU into render.py eval loop + metrics.json"
```

---

## Task 5: 文档同步（CLAUDE.md 强制）

**Files:**
- Modify: `v3_plan_revised.md`、`v2_architecture.md`

- [ ] **Step 1: v3_plan_revised.md 登记 + Done Log**

在 § 1.2 任务表新增一行（编号建议 P0.4 evaluator 增强，或按 Phase 0 评测线归类），状态随进度更新；§ 6 Done Log 追加：日期 2026-06-10 + commit hash + 「NTA-IoU 接入（render.py），smoke `mean_nta_iou=<实测>`」。注意 mermaid 看板卡片内括号一律全角（CLAUDE.md Mermaid 铁律）。

- [ ] **Step 2: v2_architecture.md § 6 文件清单 + § 7 不变量**

§ 6 文件清单加 `threedgrut/model/nta_iou.py`、`vehicle_detector.py`（标 ✅）；§ 7 加一行不变量：「NTA-IoU 复用 `project_cuboids_to_mask`（eval 侧投影），不走 viser `FthetaForwardProjector`（BUG-1 隔离）」。

- [ ] **Step 3: Commit**

```bash
git add v3_plan_revised.md v2_architecture.md
git commit -m "docs(plan): mark NTA-IoU done in kanban + Done Log; docs(arch): register nta_iou module"
```

---

## Self-Review notes
- **Spec 覆盖**：投影(Task1)/匹配(Task2)/检测器(Task3)/接入+写盘(Task4)/文档(Task5) 全覆盖。
- **BUG-1**：全程只用 `dynamic_mask.project_cuboids_to_mask`，未引用 `ftheta_projector.py`，风险隔离（已写入 Task4 与架构不变量）。
- **类型一致**：`detect_vehicles` 返回 `[M,4]` float CPU tensor；`compute_frame_nta_iou` 用 `torchvision.ops.box_iou` 消费；GT 框由 `project_track_to_2d_box` 产 `(x1,y1,x2,y2)` float。一致。
- **已知待确认点**：①`project_cuboids_to_mask` 真实签名/入参顺序以源码为准（Task1 Step4 提示）；②车辆 track class 字符串由 Task0 实测确定；③`render.py` 中 pinhole K 是否在 gpu_batch（FTheta clip 下为 None，不影响）。
- **向后兼容**：无 GT 车辆帧→None 不计入；检测器不可用→静默跳过，旧 metrics.json 不新增 key（比较脚本不炸）。
