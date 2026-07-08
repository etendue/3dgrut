# 视觉多边形静态 ego mask 实现计划（b6a9 10 相机，P0.3）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 Claude 视觉标注的多边形/圆环为 b6a9 的 10 台相机各生成一张静态 ego mask（含后视镜、鱼眼黑圈），写回 egomask itar 替换旧的，供已合入的 `EgomaskAuxReader` / `resolve_ego_valid_mask` 直读。

**Architecture:** 纯逻辑（栅格化 + 并集补强合成）在 `threedgrut/datasets/egomask_static.py`（Mac TDD）；可视化（网格参考图 + resolve-后叠图）与 itar 写驱动在 `scripts/`（inceptio 实跑，原图/现有 itar 在那）。标注是 Claude 人在环闭环：网格参考图 → 目测顶点 → 叠图自检 → 大g 目检 gate → 写 itar。

**Tech Stack:** Python + numpy + PIL（ImageDraw.polygon）+ scipy.ndimage + ncore SDK IndexedTarStore/zarr（itar 读写）；pytest（Mac CPU venv + conftest stubs）；inceptio 出图/写 itar。

## Global Constraints

- **plan 格式（大g 约定）**：本 plan 不贴 code snippet；每步给签名 / 断言要点 / 命令意图。
- **Mac 测试命令**：`/Users/etendue/repo/3dgrut2/.venv/bin/python -m pytest threedgrut/tests/<file> -v`（venv 在主 repo，worktree 用绝对路径；scipy 已装）。
- **itar write-once 纪律**：`.zarr.itar` 不可 in-place —— 新 itar 先写临时名 → 旧 itar `mv` 到 `aux_backup/` → 临时名改回正名；`discover_aux_path` 同目录多个同类 itar 会 ValueError，替换后目录内只留 1 个正名 itar；itar 中途绝不 `docker stop`/kill（write-once header 末尾才写）。
- **字节等价不变量**：本任务只改 b6a9 clip 目录的**数据文件**（egomask itar），不改任何代码判定路径 → PAI 线（无 egomask itar）行为逐字节不变；下游 P0.2 Task 1 reader / Task 2 接线零改动。
- **范围**：做 10 台；**跳过 front_tele_30fov / front_standard_55fov**（实测自车不入镜，不写 itar 条目 → reader `has_camera` False → resolve branch 3 全 True）。
- **保守覆盖**：宁大勿漏；**目检以 resolve dilation(默认30)后 valid 区域为准**，非裸多边形。
- **inceptio 铁律**：`source ~/miniforge3/etc/profile.d/conda.sh && conda activate 3dgrut2`；ssh 间隔留白防限流；查 GPU/进程用 `ps`/`pgrep [p]ython` 不用 nvidia-smi/pkill（易断连）。
- **数据不进 git**：itar 是数据；顶点 JSON（标注产物）入 git；关键叠图路径写 commit message / Done Log 引用。
- **commit 署名**：`Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`。

**Spec:** [`docs/superpowers/specs/2026-07-08-visual-polygon-egomask-design.md`](../specs/2026-07-08-visual-polygon-egomask-design.md)

---

### Task 1: 栅格化纯函数（多边形 + 鱼眼圆环 + 并集）

**Files:**
- Create: `threedgrut/datasets/egomask_static.py`
- Test: `threedgrut/tests/test_egomask_static.py`

**Interfaces:**
- Consumes: numpy、`PIL.Image`/`PIL.ImageDraw`（`ImageDraw.polygon` 含边填充）。
- Produces:
  - `rasterize_polygons(polygons: list[list[tuple[int, int]]], hw: tuple[int, int]) -> np.ndarray` —— `(H, W)` bool，各多边形内部 True 的**并集**；空 list → 全 False。
  - `rasterize_fisheye_outer(center_xy: tuple[float, float], radius: float, hw: tuple[int, int]) -> np.ndarray` —— 成像圆**外**为 True（黑边=无效），圆内 False。
  - `build_camera_mask(hw, polygons=None, fisheye_circle=None, base_mask=None) -> np.ndarray` —— `base_mask ∪ rasterize_polygons(polygons) ∪ rasterize_fisheye_outer(*fisheye_circle)`；各参数 None 时跳过该项；`base_mask` 供并集补强传入现有 itar mask；返回 `(H, W)` bool。

- [ ] **Step 1: 写失败测试。** 断言要点（`hw=(20,30)` 小图，精确二值 `np.array_equal`）：① 单个矩形多边形 `[(x0,y0),(x1,y0),(x1,y1),(x0,y1)]` 内部像素 True、外部 False（按 PIL polygon 含边语义核对角点/边像素）；② 两个不相交多边形 → 并集两块都 True；③ 空 `polygons=[]` → 全 False；④ `rasterize_fisheye_outer(center=图心, r)`：圆心处 False、四角（距心>r）True、半径边界按 `dist>r` 判定；⑤ `build_camera_mask` 并集正确——给 `base_mask`(一块) + `polygons`(另一块) + `fisheye_circle` → 三者并集精确相等；`base_mask=None`/`polygons=None`/`fisheye_circle=None` 组合不报错。
- [ ] **Step 2: 跑测试确认失败**（ImportError/AttributeError 级，非 collection error）。命令意图：pytest 该文件，见 `cannot import name 'rasterize_polygons'`。
- [ ] **Step 3: 实现三函数。** `rasterize_polygons` 用 `Image.new("1")` + `ImageDraw.Draw().polygon(pts, fill=1)` 逐多边形画后 `np.asarray(bool)` 并集；`rasterize_fisheye_outer` 用 `np.ogrid` 距离场 `(xx-cx)**2+(yy-cy)**2 > r**2`；`build_camera_mask` 起始全 False 依次 `|=`。
- [ ] **Step 4: 跑新测试全绿 + 既有回归**（`test_egomask_aux_reader.py` 等同目录全套 `-x -q` 零失败）。
- [ ] **Step 5: Commit** `feat(P0.3): egomask 多边形/鱼眼圆环栅格化纯函数 + Mac 单测`。

---

### Task 2: 合成纯函数（并集补强 + per-camera 组装 + 跳过清单）

**Files:**
- Modify: `threedgrut/datasets/egomask_static.py`（追加）
- Test: `threedgrut/tests/test_egomask_static.py`（追加）

**Interfaces:**
- Consumes: `build_camera_mask`（Task 1）；`EgomaskAuxReader`（P0.2，已合 `4a9f2e6`，`has_camera`/`read_static_mask`）。
- Produces:
  - `compose_egomask_set(visual_specs, existing_reader, hw, reinforce_cams, skip_cams) -> dict[str, np.ndarray]`
    - `visual_specs: dict[camera_id -> {"polygons": list, "fisheye_circle": tuple|None}]`
    - `reinforce_cams: set[str]` —— 这些相机 `base_mask = existing_reader.read_static_mask(cam)`，与视觉并集；若 `not existing_reader.has_camera(cam)` → `raise KeyError`（防误配）。
    - `skip_cams: set[str]` —— 不出现在返回 dict（也不应出现在 visual_specs；若出现则忽略并 warn）。
    - 返回 `{camera_id: (H, W) bool}`，仅含要写 itar 的相机（visual_specs 键减去 skip_cams）。

- [ ] **Step 1: 写失败测试。** 用轻量 fake reader（`has_camera`/`read_static_mask` 返回预置 `(H,W)` bool）。断言要点：① `reinforce_cam` 结果 = fake base ∪ 该 cam 视觉多边形（精确 `np.array_equal`，证明保留了现有精细 mask 又补了漏项）；② 纯视觉 cam（不在 reinforce_cams）结果 = 仅视觉多边形（**不含** base，即便 reader 有该 cam 也不取）；③ `skip_cam` 不在返回 dict；④ reinforce_cam 但 `fake.has_camera=False` → `raise KeyError`；⑤ 返回 dict 键集 = `set(visual_specs) - skip_cams`。
- [ ] **Step 2: 跑测试确认失败。**
- [ ] **Step 3: 实现 `compose_egomask_set`**：遍历 visual_specs，skip 的跳过；reinforce 的取 base（缺则 raise）；调 `build_camera_mask(hw, polygons, fisheye_circle, base_mask)`。
- [ ] **Step 4: 全绿 + 既有回归零失败。**
- [ ] **Step 5: Commit** `feat(P0.3): egomask 并集补强合成纯函数（现有 itar mask ∪ 视觉多边形）+ 单测`。

---

### Task 3: 可视化工具（网格参考图 + resolve-后叠图）

**Files:**
- Create: `scripts/egomask_viz.py`

**Interfaces:**
- Consumes: `threedgrut.datasets.aux_readers._open_itar_zarr`（读原图 `cameras/<cam>/frames/<ts>/image` = 0-D PNG bytes）；`resolve_ego_valid_mask`（P0.2）用于叠图口径一致。
- Produces:
  - `read_first_frame_rgb(clip_dir, camera_id) -> PIL.Image` —— clip 目录内 `*.ncore4-<camera_id>.zarr.itar` 首帧解码 RGB。
  - `render_grid_reference(clip_dir, camera_id, out_png, grid=100)` —— 原图 + 每 `grid` px 网格线 + 边缘像素刻度数字 → 存 PNG（Claude 读顶点用；全分辨率不缩放以保坐标真实）。
  - `render_resolved_overlay(clip_dir, camera_id, mask, out_png, dilation_iters=30)` —— `valid = resolve dilation 后取反` 语义：把 `logical_not(binary_dilation(mask, iters))` 的**被 mask 区**红色半透明叠原图 → 存 PNG（目检最终效果）。

- [ ] **Step 1: 写脚本**（三函数 + `__main__` 支持 `--clip-dir --camera-id --mode grid|overlay --out`）。无单测（视觉工具，见 spec §8）；Mac 只做 `python -c "import ast; ast.parse(open('scripts/egomask_viz.py').read())"` 语法检查。
- [ ] **Step 2: inceptio 出 10 台网格参考图**（rsync 脚本 + aux_readers.py 到 `/tmp/egoviz/`，conda 跑 grid 模式），scp 回 Mac。验收：Claude 读参考图能看清网格坐标与自车结构边缘（对齐、刻度可读）。
- [ ] **Step 3: Commit 脚本** `feat(P0.3): egomask 标注可视化工具（网格参考图 + resolve后叠图）`。

---

### Task 4: itar 写函数 + inceptio round-trip 验证

**Files:**
- Create: `scripts/gen_static_egomask_b6a9.py`（含 `write_egomask_itar` + round-trip 自检；驱动主流程在 Task 5 补全）

**Interfaces:**
- Consumes: ncore SDK `stores.IndexedTarStore(path, mode="w")` + `zarr`；`EgomaskAuxReader`（回读校验）；`merge_lidar_aux.py` 的 itar 写模式（`create_dataset(name, shape=(), dtype)` + 0-D bytes 写）为参照。
- Produces:
  - `write_egomask_itar(masks: dict[str, np.ndarray], out_path)` —— 每 `camera_id` 写 `aux/egomask/<camera_id>/"0"` = `(H,W)` bool → `uint8*255` → PNG bytes（0-D `|S` array）；写完 finalize（with/close 触发 tar index header）。内部 group 结构与 nre-tools 原样一致（Task 1 probe 确认 `aux/egomask/<cam>/<ts>`）。

- [ ] **Step 1: 写 `write_egomask_itar` + `__main__` round-trip 自检**：自检逻辑 = 合成 2 台已知假 mask（不同区域）→ `write_egomask_itar` 到临时 itar → `EgomaskAuxReader` 读回 → 两台 `read_static_mask` 与写入 `np.array_equal`，打印 `ROUNDTRIP OK`。
- [ ] **Step 2: inceptio 跑 round-trip 自检**（Mac 无 IndexedTarStore，只能 inceptio；rsync 脚本 + aux_readers.py，conda 跑）。验收：stdout 出 `ROUNDTRIP OK`，无 `invalid index header`。
- [ ] **Step 3: Commit** `feat(P0.3): egomask itar 写函数 + inceptio round-trip 验证`。

---

### Task 5: 标注闭环 + 生成 + 写 itar 替换 + 回读 + 大g 目检 gate

**Files:**
- Create: `scripts/egomask_polygons_b6a9.json`（顶点/圆参数标注数据，入 git）
- Modify: `scripts/gen_static_egomask_b6a9.py`（补全驱动主流程：读 JSON → compose → 叠图 → 写 itar 替换 → 回读）

**Interfaces:**
- Consumes: Task 1–4 全部产物 + P0.2 `EgomaskAuxReader`。
- Produces: b6a9 clip 目录内**替换后的**完整 egomask itar（10 台静态 mask、2 台无条目）+ `egomask_polygons_b6a9.json` + 10 台 resolve-后叠图。
- 驱动 CLI 意图：`gen_static_egomask_b6a9.py --clip-dir <D> --polygons scripts/egomask_polygons_b6a9.json --mode overlay|write`。

- [ ] **Step 1: inceptio 生成 10 台网格参考图**（Task 3 工具 grid 模式），scp 回 Mac。
- [ ] **Step 2: Claude 逐台标注顶点。** Claude 读每张网格参考图 → 目测自车结构多边形顶点（像素坐标）+ 鱼眼圆参数，写入 `egomask_polygons_b6a9.json`。规则：4 台 reinforce（cross_left/cross_right/left_wide/right_wide）**只标现有 itar 漏掉的结构**（主要后视镜/防护杆）；6 台纯视觉（front_wide/back_rear_wide/rear_left/rear_right/front_fisheye/back_rear_fisheye）标全部结构；两台鱼眼加 `fisheye_circle`。JSON schema：`{camera_id: {"polygons": [[[x,y],...], ...], "fisheye_circle": [cx,cy,r] 或 null, "reinforce": bool}}`。
- [ ] **Step 3: inceptio 生成叠图。** 驱动 overlay 模式：读 JSON → `compose_egomask_set`（reinforce 台读现有 itar 并集）→ 每台 `render_resolved_overlay`（resolve-后口径）→ scp 10 张回 Mac。
- [ ] **Step 4: Claude 自检迭代。** Claude 读 10 台叠图 → 自车结构（后视镜/防护杆/车体/鱼眼黑圈）是否被 valid=False 覆盖、有无大块误盖有效路面 → 不达标回 Step 2 调顶点重跑 Step 3（记录迭代轮次）。
- [ ] **Step 5: 大g 目检 gate。** 10 台最终叠图交大g，**显式请大g 视觉验证并集补强的后视镜确被补上**；大g 确认放行才继续。未确认不写 itar。
- [ ] **Step 6: 写新 itar + write-once 替换。** 驱动 write 模式：`compose_egomask_set` → `write_egomask_itar` 到临时名 `*.aux.egomask.zarr.itar.new` → 旧 itar `mv` 到 `<clip>/aux_backup/` → 临时名改回正名。绝不中途中断（write-once）。
- [ ] **Step 7: 回读验证。** `diag_egomask_itar.py`（或等价）复扫新 itar：10 台 `nonzero>0`、跳过 2 台无条目；`EgomaskAuxReader` 读新 itar `camera_ids()` = 10 台且各 `has_camera` True。双证据（diag 数字 + reader）入 Done Log。
- [ ] **Step 8: Commit** `feat(P0.3): b6a9 视觉多边形静态 ego mask 标注数据 + 生成驱动 + itar 替换实跑`（JSON + 驱动最终态入 git；itar 数据不进 git，叠图关键截图路径写 commit message 描述）。

---

### Task 6: 文档回填

**Files:**
- Modify: `docs/superpowers/specs/2026-07-08-expand-cameras-campaign-design.md`（§3 P0.3 加 supersede 指针）
- Modify: `v5_plan.md`（Phase C 看板 P0.3 状态 + Done Log）
- Modify: `v2_architecture.md`（§6 文件清单 + §7 不变量）

**Interfaces:** 纯文档，无代码接口。

- [ ] **Step 1: expand-cameras spec §3 P0.3** 段首加一行 supersede 指针 → 本 spec（`2026-07-08-visual-polygon-egomask-design.md`），标注「生成源改视觉多边形，存储/下游不变」。
- [ ] **Step 2: v2_architecture.md** §6 文件清单加 `egomask_static.py` / `egomask_viz.py` / `gen_static_egomask_b6a9.py`（标 ✅ + commit 短 hash）；§7 不变量加一行「b6a9 egomask 视觉多边形静态 mask 替换 itar；PAI 线无 egomask itar 字节等价」。
- [ ] **Step 3: v5_plan.md** Phase C 看板 P0.3 移到 Done（✅）+ Done Log 追加一条（日期 + commit + 10 台/跳过 2 台 + 回读 nonzero 数字 + 迭代轮次 + 大g gate 确认）。若含 mermaid 改动，**全角括号自查**（`awk` 扫零输出）。
- [ ] **Step 4: Commit** `docs(plan)+docs(arch): P0.3 视觉多边形静态 egomask 回填看板/架构/spec supersede`。

---

## Self-Review 记录

- **Spec 覆盖**：spec §2 范围表→Task 5 Step 2 标注规则 + Global Constraints 跳过清单；§3 生成方法→Task 1（栅格化）；§4 标注闭环→Task 5 Step 1–5；§5 存储替换 write-once→Task 4（写函数）+ Task 5 Step 6；§6 dilation 注记→Task 3 `render_resolved_overlay` + Global Constraints；§7 验收→Task 5 Step 5/7；§8 测试→Task 1/2（Mac 单测）+ Task 4（round-trip）；§9 plan 关系→Task 6；并集补强→Task 2。无缺口。
- **签名一致性**：`build_camera_mask`（Task 1 定义）→ Task 2 `compose_egomask_set` 调用同名同参；`compose_egomask_set` / `write_egomask_itar` / `render_grid_reference` / `render_resolved_overlay` / `read_first_frame_rgb` Task 间引用一致；`EgomaskAuxReader.has_camera`/`read_static_mask`/`camera_ids` 与 P0.2 已合实现一致。
- **占位符扫描**：无 TBD/TODO；视觉/执行类任务（3/5）为 runbook 型，验收判据与命令意图明确，不虚构结果数字（叠图质量/回读 nonzero 执行期实测填 Done Log）。
- **公差**：栅格化/合成单测二值精确 `np.array_equal`；round-trip `np.array_equal`。
