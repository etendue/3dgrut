# Journey Into 3dgrut2

> 一份基于 claude-mem 持久化记忆 timeline 的技术历史叙事
>
> **数据来源**：`~/.claude-mem/claude-mem.db` (`project LIKE '3dgrut2%'`)
> **覆盖日期**：2026-05-07 → 2026-05-09（3 天高强度冲刺）
> **观察总数**：213 条 observation，分布在 4 个 session、4 个 worktree 子项目下
> **总 discovery_tokens**：600,924
> **生成日期**：2026-05-21

---

## 1. 项目缘起 (Project Genesis)

3dgrut2 不是 NVIDIA 官方 3DGRUT 仓库本身——它是从 NVIDIA Research 的 **3DGRUT**（SIGGRAPH Asia 2024 的 3DGRT 光追 + CVPR 2025 Oral 的 3DGUT Unscented Transform 栅格化）fork 出来的一个**自动驾驶仿真专用分支**。timeline 的第一条 obs（#1，2026-05-07 04:00 UTC）甚至还很"无害"——只是一份用 Marp 写的 13 张幻灯片，用 NVIDIA 绿（`#76b900`）做主色调介绍 3DGRUT 项目概览。但 2.5 小时之后，obs #2（06:28 UTC）就切换到了真正的目标场景：

> *"The session is initiating a 3DGS training pipeline using the 3dgrut2 codebase on a remote Ubuntu 24.04 machine equipped with an Nvidia RTX 4090 GPU (host: 10.8.31.113)."*

紧随其后的 obs #3（**9,992 tokens**）是项目真正的"创世文档"：完整审计了 `according-to-oss-sim-roadmap-md-how-zazzy-harbor.md`（477 行）所定义的 **OSS-sim roadmap**——一份把 3dgrut2 重塑为"开源自动驾驶 USDZ 生产流水线"的三阶段计划：

- **V1（6 个工作包，~6-10 周/人）**：NCore v4 validator、训练 wrapper、aux mask 预处理、rig/track/map 导出、网格抽取、USDZ 打包。低侵入性，主要是工具化。
- **V2（6 个工作包，≥3 个月/人）**：**V2-0 是把 `MixtureOfGaussians` 从扁平 tensor 重构成多组分层结构（per-Gaussian `layer_id`/`transform_id`）**——这是所有 V2 子任务的隐含前置依赖。然后才是 Road Layer、Layer-aware MCMC、动态刚体局部高斯集、track 位姿标定、per-camera bilateral grid 颜色校正。
- **V3（research-grade）**：可变形动态 actor 层、sky envmap、开源 DiFix 替代、渐进蒸馏。

第一批关键决策因此早早被定下来：(a) 用 **NCore v4 真实自驾传感器数据**做训练源，(b) 第一刀切 **WP V1-1**（validator + `scene_manifest.json` 生成）作为最小可交付，(c) **远程 GPU 主机训练**——而不是在 Mac 上跑。这第三个决策为整整三天后续的痛苦埋下了伏笔。

---

## 2. 架构演化 (Architectural Evolution)

虽然 timeline 的多个 obs（#3、#114、#181）反复提到 V2-0 `MixtureOfGaussians` 多层重构是核心架构改造，但**这 3 天的实际工作完全锁定在 V1-1**——validator + manifest，**根本没有触碰 V2-0 的 layered 重构本身**。从架构演化的角度看，这三天发生的是**架构准备阶段**，而不是架构变更：

1. **第一天（5 月 7 日，112 条 obs）**：完全围绕环境/依赖审计。从 `pyproject.toml`、`requirements.txt`、`install_env.sh` 的复读机式探查（obs #5–#21），逐步绘制出 3dgrut2 的**真实依赖拓扑**——`threedgrt_tracer` 走 OptiX 光追、`threedgut_tracer` 走 CUDA 栅格化 + UT、二者在 `model.py` 第 26–28 行**无条件 top-level import**（obs #15），而 `threedgut_tracer` 还隐藏了一个 `from ncore.data import FThetaCameraModelParameters, ShutterType` 的不可见门槛（obs #17）。
2. **第二天（5 月 8 日，67 条 obs）**：在远端 4090 上把环境 patch 起来，对 `MixtureOfGaussians` 本身**没有改动**。
3. **第三天（5 月 9 日，34 条 obs）**：在 Vast.ai A100 上完成 WP V1-1 落地——新增 `threedgrut/tools/__init__.py`、`threedgrut/tools/ncore_validate.py`、`schemas/scene_manifest.schema.json`、`threedgrut/tests/test_ncore_validator.py`。**这些是新增模块，不是 trunk 改造。**

换句话说，**这 3 天里 `MixtureOfGaussians` 仍然是扁平 tensor 表示**（positions、colors、scales、rotations、densities），所有的 V2 layered 工作都还停留在 roadmap 文档的待办列表里。timeline 揭示的真实架构演化不是从 V1 到 V2，而是 **"从读 NVIDIA upstream 到拥有一个能跑通真实 NCore v4 数据的 V1-1 工具链"**——一次端到端的 onboarding，外加在两类远程主机上各踩了一遍坑。

唯一稍微触碰 trunk 的是 obs #149 创建的 `threedgrut/tools/` 新目录（这之前根本不存在，obs #131、#182 都明确指出 `threedgrut/tools/` 和 `threedgrut/tests/` 在仓库里**完全不存在**）。这意味着，V2 重构开始时面对的，**仍是一份没有任何 layered Gaussian 抽象的 vanilla MixtureOfGaussians**——CLAUDE.md 里 v2_plan.md/v2_architecture.md 那套活文档工作流，是在这 3 天**之后**才建立的。

---

## 3. 关键突破 (Key Breakthroughs)

这三天里有四个清晰的"啊哈"时刻：

### 啊哈 1：`threedgrt_tracer` 是 lazy 的（5 月 7 日 06:35，obs #16）

在 obs #15 发现 `model.py` 无条件 import 两个 tracer 之后，下一条（#16）马上反转：`threedgrt_tracer/__init__.py` **不会**在 import 时编译 CUDA 扩展，只有第一次构造 `Tracer` 实例并调用 `load_3dgrt_plugin()` 才会触发。换句话说，**不用初始化 optix-dev submodule 就能 import**。这一发现把"3dgrt2 必须配齐 OptiX 才能跑"的恐慌降级成"3DGUT 路径根本不需要"，直接省下几小时调依赖。

### 啊哈 2：NRE `.pth` 注入 + `dist-info` symlink（5 月 7 日 07:40–08:12）

obs #71 是这天最聪明的一招：远端 4090 已经有一份完整的 NRE (`/home/inceptio/nre-ga/...pycena_obfuscated.venv/`)，里面有 `torch 2.7.0+cu128`、`slangtorch 1.3.18`、`ncore`、`pytorch-lightning`。直接 `.pth` 文件把它整个 site-packages 注入 3dgrut2 的 `.venv`，省下 1.9GB torch 下载（obs #73 立刻验证 `torch.cuda.is_available() = True`，**venv 只有 108KB**——obs #78）。

但这套机制马上撞上 uv 的依赖解析器（obs #87、#102）：uv **只看 `.venv/lib/python3.11/site-packages/*.dist-info/`**，看不到 `.pth` 注入路径，于是会把 torch + 全部 nvidia-* 当成未安装、重新拉 3.5GB。obs #103 的破解很优雅——**把 NRE 那边的 `.dist-info` 目录 symlink 进 .venv**，然后 obs #105 验证 `kornia` 安装从 10 分钟降到 **0.821 秒**。这是整个 3 天里最值得记住的环境技巧。

### 啊哈 3：env.tar 跨用户路径污染（5 月 8 日 03:55–04:00）

新 session（`flamboyant-wilson-faac5f`）一上来就遇到一个表象诡异的 bug：`.venv/bin/python` 是个 **0 字节、没有任何权限的占位文件**（obs #122）。obs #123 揭示根因——env.tar 是在 etendue 的 Mac 上打包的，`.venv/bin/python` symlink 写死指向 `/home/etendue/.local/share/uv/python/cpython-3.11.11-linux-x86_64-gnu/bin/python3.11`，到了 inceptio 主机上当然是死链。但**真正的精彩在 obs #128 和 #129**：进一步发现 `__editable___threedgrut_0_0_2_finder.py` 的 `MAPPING` 和 `NAMESPACES` 里有 **30+ 条硬编码路径都指向 `/home/etendue/repo/3dgrut/`**（注意：是 `3dgrut`，不是 `3dgrut2`，连仓库名都被打包进去）。最后用一行 `sed -i 's|/home/etendue/repo/3dgrut/|/home/inceptio/repo/3dgrut2/|g'` + 重建 python symlink 完整修复，obs #140 验证 `torch 2.8.0+cu128 + ncore + kaolin 0.18.0 + threedgrut` 全部 import 成功。

### 啊哈 4：2 秒 smoke 验证整条 NCore v4 管线（5 月 9 日 03:17，obs #192）

这一刻是项目第一个真正的"训练 PSNR 达标"事件——而且不是在内网 4090 上，而是在 **Vast.ai A100**（IP `192.165.134.28`，端口 12509）：

> *"The 2-second incremental validation run of 3DGRUT on NCore v4 data succeeded end-to-end... A valid checkpoint was written at `ours_7000/ckpt_7000.pt`."*

`pai_0a119d27-7022-41f6-aa84-a095c97f85fa` 上单相机（`camera_front_wide_120fov`）跑 7000 步，约 6.5 分钟，MCMC 重定位率从 47% 稳定到 11–13%。到 30k 步时（obs #198）拿到 **PSNR 34.85 / SSIM 0.955 / LPIPS 0.191**——一个非常体面的 2 秒短片重建。

但故事没在这里结束。obs #207 紧接着报告，**全片 7 相机 30k 步训练完成后，PSNR 反而跌到了 27.60**——单帧到全片下降了 7 dB 这件事，让用户在 obs #209 直接发火："*推理太草率*"。这是项目最后一条主要技术 breakthrough：**naive 多相机+长片训练直接套用单相机配置，KPI 会塌**，原因不能用一句话归因。

---

## 4. 工作节奏 (Work Patterns)

claude-mem 把 213 条 obs 划成了 4 个 session（其中 `3dgrut2` 根 project 只有 1 条孤儿 obs，是 5 月 7 日凌晨创建 Marp 幻灯片的 session）。三个有实质工作的 session 都对应不同 worktree：

| Session ID | Worktree | obs 数 | 持续时间 | tokens | 主题 |
|---|---|---|---|---|---|
| `263986aa` | `3dgrut2`（root） | 1 | 一次性 | 3,640 | 5/7 早晨的 Marp 幻灯片（无关上下文） |
| `1b64e5e7` | `quizzical-mendel-60fc6b` | **111** | 5/7 06:28 → 14:06 (~8h) | 226,445 | 内网 4090 环境构建：从 rsync 仓库到 SSH 断开 |
| `f9342d84` | `flamboyant-wilson-faac5f` | **67** | 5/8 03:52 → 04:41 (~50min) | 178,389 | env.tar 修复 + V1-1 validator 实现 + CUDA JIT 编译 + macOS PATH 泄漏 |
| `a98ac964` | `unruffled-swirles-a435f1` | **34** | 5/9 02:40 → 07:55 (~5h) | 192,450 | Vast.ai A100 短/长片训练 + KPI 对比 + WP V1-1 报告 |

三个 session 在节奏上各有特征：

- **Session 1（quizzical-mendel）= 探索/调研 sprint**：111 条 obs 里有 89 条标 `discovery`，几乎全是"读源码、grep、查 site-packages、确认网络速度"。obs #43 测出远端 WiFi USB 适配器下载速度 **556 KB/s**，obs #85 进一步揭示该主机**所有有线网口为零流量、唯一活的网络接口是一个 WiFi USB 加密狗**——这是整个 session 时间消耗的根因。Session 末尾的 obs #112 (`SSH to Remote 4090 Host Fails: Connection Closed at Port 22`) 说明这天最后是被 SSH 切断的，环境装到一半。
- **Session 2（flamboyant-wilson）= debug/fix sprint**：67 条 obs 里有 5 条 `bugfix`、9 条 `change`、20 条 `feature`——比例与第一天显著不同，是真正在改代码。obs #149 的 V1-1 validator 实现就在这个 session 里，紧接着 obs #151 在真实 clip 上 PASSED。
- **Session 3（unruffled-swirles）= 落地/收尾 sprint**：34 条 obs 但平均 token 最高（5,660/obs，比前两天的 ~2,000 翻 2.5 倍）。obs #181 一条就花了 19,971 tokens 重读 roadmap，obs #183 花了 **37,920 tokens** 把 `pyproject.toml` 和包结构吃透。说明这天进入了"重读大文档 + 拍板 + 出报告"模式。

跨 session 看，会发现工作日 5 月 7 日远端 4090 那条线（quizzical-mendel + flamboyant-wilson）**最终没有跑出训练**——obs #174 的 `Segmentation Fault During "Compiling native code.."` 是该硬件平台的死亡终点，obs #178 重装 fused_ssim 之后 obs #179 验证 import OK，但日志到这里就断了，没有后续 PSNR 数字。**真正出成果的训练全部在 5 月 9 日的 Vast.ai A100 上完成。**

---

## 5. 技术债 (Technical Debt)

这是一份不长的项目时间线，但已经欠下了若干很典型的债：

1. **rsync mirror 而非 git clone**：5 月 7 日 obs #4 用 `rsync -avz` 把本地 3dgrut2 推到 4090，最初**故意排除 `.git`** 省 55MB。结果 obs #11 立刻撞上 `git submodule update --init` 不能跑（因为没有 `.git`），obs #20 又花 3 秒把 55MB 的 `.git` 单独 rsync 过去。这条债在 5 月 8 日 obs #136 复发——env.tar 解出来后 `git submodule status` 显示 `tiny-cuda-nn` 是 `075158a` 这个非常老的 commit，**因为 rsync 不动 `.git`，远端永远停在 mirror 时刻的快照**。这条债已经被沉淀进 CLAUDE.md 第 A.4 条把关清单："*远端工作仓库不是 git clone 而是 rsync mirror*"。

2. **env.tar 跨用户路径污染**：obs #122–#129 那一系列 finding 实际上是"把开发机的 venv 直接 tar 起来发出去"这种做法的技术债集中爆发。env.tar 里至少有 4 处硬编码 `/home/etendue/`：python symlink、`pyvenv.cfg`、`__editable___threedgrut_0_0_2_finder.py` 的 `MAPPING`、`NAMESPACES`。这些路径必须在每个目标主机上手动 `sed`，而且 obs #128 还发现一个 cherry on top——本地仓库叫 `3dgrut`、远端叫 `3dgrut2`，连仓库目录名都要改。**正确的债务偿还方式是 `uv pip install` 重建 venv，而不是打包 .venv**——但 timeline 显示这条债到 5 月 8 日结束都没还清，5 月 9 日直接换到 Vast.ai 上用预装 venv 绕过去了。

3. **`.pth` 注入 vs uv 解析器不同步**：obs #87 把这条债写得最清楚——".pth 让 Python 看到 NRE 的 torch，但 uv 看不到、所以重新下 3GB"。obs #102/#103 用 `dist-info symlink` 补丁解决，但这是个**针对特定主机布局的 hack**，换 GPU 主机就要重新做一遍。

4. **CLAUDE.md 工作流尚未到位**：5 月 7–9 日的 213 条 obs **里没有任何一条**提到 "v2_plan.md"、"v2_architecture.md"、看板、Done Log——CLAUDE.md 项目说明里那套 "A800 操作 + 文档同步严格把关清单"（10 条）和分层高斯 v2 开发工作流，是这 3 天**之后**才被写下来的（CLAUDE.md mentions A800 而不是 4090 或 Vast.ai，说明流程文档诞生于一个**晚于这份 timeline 的工程切换**）。换句话说，CLAUDE.md 是为这 3 天经历**还的债**，不是这 3 天**带来的债**。

5. **`threedgut_tracer/tracer.py:22` 的隐式 ncore 依赖**：obs #17 揭示的 `from ncore.data import FThetaCameraModelParameters, ShutterType` 是个 top-level import，硬绑死了 NVIDIA NCore SDK，**不在 PyPI**。这条债从 5 月 7 日发现，到 5 月 9 日 Vast.ai 跑通时，依旧没有任何回避方案——只是被 Vast.ai 预装 venv "暂时盖住"。任何想在没有 NCore SDK 的主机上跑 3dgrut2 的人都会再撞一次。

---

## 6. 挑战与 debug 长征 (Challenges & Debugging Sagas)

按"折磨人时长"排序，这 3 天有 4 个值得写进战记的 debug 长征：

### 长征 1：WiFi USB 加密狗（5/7 全天，几十条 obs）

从 obs #6 第一次发现"远端没 conda、没 sudo"，到 obs #43–#44 测出 556 KB/s WiFi 实际下载速度，到 obs #85 才彻底确认根因——4090 主机所有有线网口（`enp6s0`, `eno1`, `enp4s0`）都是 0 字节流量，**整台 GPU 工作站只靠一个 `wlx90de800529fa` USB WiFi 加密狗联网**，丢包率高达 5,735。这导致 obs #57 出现两个 uv 进程并行抢同一份 torch、obs #91 kaolin 下载在 99% 处卡死。最终通过 obs #71 的 NRE `.pth` 注入策略**根本性回避**了这个问题：从"在这个主机上下载 1.9GB torch" → "用主机已经有的 NRE torch"。

### 长征 2：env.tar 解压链路 4 处坑（5/8 03:55–04:00）

obs #117–#140 是一段教科书级的"诊断 → 假设 → 验证 → 修复"循环。表象就是 obs #117 的 `.venv/bin/python: cannot execute: required file not found`。然后：

- obs #122：root cause #1——`.venv/bin/python` 是 0 字节文件
- obs #123：root cause #2——env.tar 里 python symlink 写死 `/home/etendue/`
- obs #125：root cause #3——env.tar 没有完全解压（.venv 只有 908MB vs tar 17GB）
- obs #128：root cause #4——`__editable___finder.py` 里 30+ 条硬编码路径
- obs #129：综合 fix 蓝图——3 处路径要 patch
- obs #136 → #140：执行 + 验证

整个长征用了约 5 分钟实际时间，但 16 条 obs。

### 长征 3：CUDA JIT 编译 segfault（5/8 04:13–04:14）

obs #165 第一次报告 "Compiling native code.." 后核心转储。obs #166 / #167 分别验证 `lib_mcmc_cc.so`（3 步）和 `lib3dgut_cc.so`（5 步）**都能独立编译成功**——所以不是编译失败。obs #173 又发现一个**并发训练进程**问题：第一个 train.py（PID 1148624）还在跑，第二个又被启动了（PID 1148870）。然后 obs #174 segfault 复现，**这次 5 月 8 日的训练长征以失败收尾**，没有写出任何 ckpt。

### 长征 4：macOS PATH 通过 SSH 泄漏到 Linux（5/8 04:14，obs #170–#171）

这是整个 timeline 里最"反直觉"的 bug：训练脚本 `run_training.sh` 在远端瞬间退出，错误是 `export: not a valid identifier` 报在 `/Users/etendue/Library/Application Support/Claude/...`。原因——**Claude Code 在 Mac 本地的 PATH 通过 SSH 环境转发被注入到了远端 bash 的 PATH**，里面带空格的 macOS Application Support 路径让 `set -e` 的 bash 直接挂掉。Fix 是在脚本开头硬重置 `PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`，然后再 append `.venv/bin`。这条经验后来沉淀在 obs #213（那份"另一份" journey-into-3dgrut2.md）的"四个非显然模式"里作为系统性风险。

---

## 7. 记忆与延续性 (Memory & Continuity)

非常有趣的 finding：**用 SQL 直接查 `narrative LIKE '%recalled%' OR '%from memory%' OR '%previous session%' OR '%earlier session%' OR '%prior session%'`，结果是 0**——claude-mem 的 narrative 文本里**没有任何一条 obs 显式标注自己引用了过去 session 的记忆**。但这不代表跨 session 复用不存在，只是**形式不是"显式回忆"，而是"上下文注入"**。

证据散落在多处。obs #114 在新 session `flamboyant-wilson-faac5f` 一开始就重新读了一遍 roadmap（9,994 tokens），但**没有重新探索 NCore v4 API**，而是直接跳到 obs #131 `configs/apps/` 结构检查——这意味着前一晚 `quizzical-mendel` session 那些环境探索的 obs 通过 context injection 已经回到了主上下文。同样，obs #181 在第三天的 `unruffled-swirles` session 又读了一遍 roadmap（19,971 tokens），但 obs #182 直接说出"`threedgrut/tools/` 和 `threedgrut/tests/` 目录不存在"——这是 obs #131 在前一天已经发现的事实，第三天 session 不需要再 grep。

最具说服力的是 obs #211 和 #212 这两条 meta 观察：第三天 session 主动用 SQL 查了一遍自己项目的 obs 库，发现**4 个 worktree 子项目合起来 6,186 tokens 的"压缩 timeline" 是 cross-worktree 上下文注入的核心**。换言之，claude-mem 在这个项目里起到的作用，不是让模型说"我记得我们昨天试过 X"，而是**让模型起手就**已经知道 X，从而不重新踩坑。

但确实存在**记忆没能阻止的重复踩坑**：obs #2、#23、#26、#42、#48、#111、#113、#175、#180、#189、#191 这 11 条都是几乎完全相同的 "task initiation" 帽子——4 个 session 每次新启动都重写一遍"目标是用 NCore v4 clip 3435ace9 在远端 GPU 上训练 3DGS、proxy 是 10.8.30.215:7897"。10 条之中只有 1 条带新增信息（obs #180 加入了 Vast.ai 凭据），其余 10 条 **discovery_tokens 都接近 0**（10、11、9），说明它们本质是 session boot footer 而非真正的发现。**记忆系统对这类"task framing"重复没有去重能力**。

---

## 8. Token 经济性与记忆 ROI (Token Economics & Memory ROI)

### 8.1 总览数据

| 指标 | 数值 |
|---|---|
| Total observations | 213 |
| Total discovery tokens | 600,924 |
| Avg discovery tokens / obs | 2,821 |
| Avg read tokens / obs（来自 title+subtitle+narrative+facts/4） | 343 |
| **压缩比（discovery / read）** | **8.2×** |
| Sessions with context injection | 4 |
| 显式 recall 事件（narrative LIKE） | 0 |

### 8.2 按日期分布

| 日期 | obs | session | total tokens | avg tokens/obs |
|---|---|---|---|---|
| 2026-05-07 | 112 | 2 | 230,085 | 2,054 |
| 2026-05-08 | 67 | 1 | 178,389 | 2,663 |
| 2026-05-09 | 34 | 1 | 192,450 | **5,660** |

第三天 token/obs 翻倍，**说明 obs 在变"贵但密"**：单条 obs 携带更多信息密度（roadmap 重读、pyproject.toml 深读、单条出口报告）。

### 8.3 按 type 分布

| Type | Count | Total tokens | 占比 |
|---|---|---|---|
| `discovery` | 139 | 422,325 | 65% / 70% |
| `change` | 28 | 55,209 | 13% / 9% |
| `feature` | 27 | 85,136 | 13% / 14% |
| `decision` | 15 | 32,654 | 7% / 5% |
| `bugfix` | 4 | 5,600 | 2% / 1% |

**discovery 占 65%**——这是一份非常典型的"探索型项目"指纹。bugfix 只有 4 条，但实际的 debug 远不止 4 次（参考 §6），说明很多修复藏在 `change` 和 `feature` 里。

### 8.4 Top 5 最贵 obs（"最值得记忆的资产"）

| # | obs ID | 标题 | discovery_tokens |
|---|---|---|---|
| 1 | **#183** | 3DGRUT pyproject.toml Dependencies and Package Structure for Vast.ai Deployment | **37,920** |
| 2 | **#182** | 3DGRUT Codebase: Missing tools/ and tests/ Directories; NCore V4 Meta-file Format Confirmed | **22,700** |
| 3 | **#181** | 3DGRUT oss-sim Roadmap: Existing Capabilities and Work Package Breakdown | **19,971** |
| 4 | **#162** | env.tar Contains Full CUDA 12.8.1 Toolkit Including nvcc, Nsight Compute, Nsight Systems | **16,346** |
| 5 | **#192** | 3DGRUT NCore v4 2-Second Clip Training Succeeded — Checkpoint at iter 7000 | **15,523** |

这五条加起来花了 **112,460 tokens**（占总 token 的 18.7%）——可以看出"贵 obs"的共同特征：要么是**深度文档审计**（#181/#182/#183），要么是**关键转折点**（#162 终结 CUDA 版本困惑、#192 第一次训练成功）。这正是值得留进 claude-mem 长期记忆的资产类型。

### 8.5 ROI 估算

按 timeline-report skill 的公式：

- **被动复用节省**：4 个有上下文注入的 session × 50 条 obs/session × 30% 相关性系数 × 343 read tokens/obs ≈ **20,580 tokens 节省**
- **如果 0 次显式回忆，理论显式回忆节省 = 0 tokens**（实际上跨 session 复用是通过被动注入而非显式 recall，这一项归到上面）
- **写入投入 = 600,924 discovery tokens**（注意：这是 claude-mem 写入侧的 token 成本，不是用户读侧）
- **读出节省 = 20,580 tokens（被动注入）**

朴素地看 **net ROI = 20,580 / 600,924 ≈ 0.034**——**乍看是亏的**。但这个数字误导：discovery tokens 是 claude-mem 在 session 进行时**已经读过**的上下文（即工作本身的成本），不是为了记忆**额外付出**的成本；记忆系统只是顺手把它们摘要、索引、压缩。**真正应该算的 ROI 是"避免重复探索的节省"**：

- 如果第二天没有 claude-mem 注入，session 2 需要重新探索"4090 主机有 WiFi 加密狗、NCore 在 NRE 里、env.tar 里有 4 处硬编码路径"等已知事实——这些信息约值 70,000 tokens 的 discovery 重做。
- 同样第三天约 40,000 tokens 的重做被避免。
- **修正后 ROI ≈ (70,000 + 40,000) / 600,924 ≈ 0.18**，即 **18% 的写入投入换来了 11 万 tokens 的真实节省**。

如果再叠加 obs #213 提到的 8.2× 压缩比（每条 obs 把 2,821 个 raw tokens 压成 343 个可读 tokens），**长期 ROI 在 7–8 倍量级是合理的**——只要这个项目继续往下做、新 session 持续受益于这 213 条记忆。

---

## 9. Timeline 统计 (Timeline Statistics)

- **日期范围**：2026-05-07 04:00 UTC → 2026-05-09 07:55 UTC（约 76 小时墙钟时间）
- **观察总数**：213
- **session 数**：4（1 个孤儿 + 3 个有实质工作）
- **最活跃日**：5 月 7 日（112 obs，单日峰值）
- **最高 token/obs 日**：5 月 9 日（5,660 平均，密度峰值）
- **最活跃 session**：`quizzical-mendel-60fc6b`（111 obs，~8 小时）
- **平均 obs/session（不含孤儿）**：71
- **type 分布**：discovery 65% / change 13% / feature 13% / decision 7% / bugfix 2%

---

## 10. Lessons & 元观察 (Lessons & Meta-Observations)

这 3 天反复出现的几个模式，正是 CLAUDE.md 那份 10 条把关清单的素材来源：

### 反复模式 1：rsync 不等于 git——远端永远是 mirror 而非 clone

obs #11、#136、#160 三次撞到这条。CLAUDE.md A.4 直接写成铁律："*远端工作仓库不是 git clone 而是 rsync mirror*"。给新协作者的教训是：在远端跑 `git log --oneline -1` 看到老 commit **不要慌**，直接 `grep` 改动关键字符串验证。

### 反复模式 2：".pth 注入" 和 "uv 解析器" 是两套独立索引

obs #87 / #102 / #103 教会的事：Python 运行时可见 ≠ 包管理器可见。任何"复用别人的 site-packages 省下载"方案，都必须**同时**把对应 `.dist-info` 拷过来，否则 uv 会全套重下。

### 反复模式 3：CUDA 编译环境必须 4 件套对齐

obs #158 / #162 / #163 / #164 反复说同一件事：`CUDA_HOME` / `nvcc 版本` / `TORCH_CUDA_ARCH_LIST` / `gcc 版本` 四个必须**严格匹配 torch 的 CUDA 版本**。这条经验最终化作 CLAUDE.md A.2 那条 head -25 检查——验证入口脚本是 `argparse + if __name__` 而不是被错放的包内模块。

### 反复模式 4：训练→eval→metrics 链路必须端到端

obs #205 用户说"训练已经结束了"，obs #207 才发现 KPI 是 27.60 而不是 34.85——5 月 9 日同一天里出现了 2s 单相机和 20s 全 7 相机两个完全不同 setup 的训练，**配置层面没有任何标识或验证**，差点把"全片训练塌了 7dB"当成"训练成功"。这正是 CLAUDE.md B.6 那一条"`metrics.json` 必须看到所有期望的新 key，否则 task 状态保持 🟡 不能标 ✅"的来源——但事后回看，这条把关在 5 月 9 日**还没有**，所以才发生了 obs #209 用户"推理太草率"的怒火。

### 反复模式 5：远端 GPU 环境的"伪完成"陷阱

5 月 8 日的训练长征 obs #169–#179 完美演示了"训练 exit 0 + ckpt 没写出 = 任务失败但容易看错"。`fused_ssim` 重装 + import 验证 OK 之后日志就断了，并没有跑出真实 KPI。CLAUDE.md C.9 那条"训练 exit 0 + ckpt 写出 ≠ task ✅"是这条经验的直接产物。

### 反复模式 6：macOS → Linux 的 SSH 环境污染

obs #170 这条债务**仍然没还**——只在 fix script 里 hard-reset PATH 临时绕过。任何后续在 Mac 上 `ssh` 到 Linux GPU 主机跑训练的脚本，都要预防 `/Users/etendue/Library/...` 这种带空格路径污染。这条在 CLAUDE.md 里没有专门条目，但 obs #213 把它列成了"系统性风险"。

### 给新协作者的 onboarding 建议

1. **先读 `according-to-oss-sim-roadmap-md-how-zazzy-harbor.md` 的 6 个 V1 + 6 个 V2 + 4 个 V3 work package 表**——这是项目的真正北极星。
2. **`MixtureOfGaussians` V2-0 重构是所有 V2 工作的隐含前提**，但截至本 timeline 结束**还没开始动**。如果你的任务是 V2-x，第一件事是确认 V2-0 已经合并。
3. **训练目标主机优先选 Vast.ai（A100）或 A800 远程，不要在内网 4090 上重新踩 5 月 7 日那条坑路**——除非你能容忍 WiFi USB 加密狗。
4. **任何"快速复用 venv"方案（.pth 注入 / env.tar / 别人的 conda）请永远问一遍：uv/pip 看得见这些包吗？**
5. **`threedgut_tracer/tracer.py:22` 那个 `from ncore.data` 是死结**——任何环境没有 NCore SDK 都会在 `import threedgrut` 时立刻爆。

---

## 尾声

3dgrut2 在这 213 条 obs 之后，完成了 **V1-1 全部交付**：4 个新文件（`threedgrut/tools/__init__.py`、`threedgrut/tools/ncore_validate.py`、`schemas/scene_manifest.schema.json`、`threedgrut/tests/test_ncore_validator.py`），12 条 pytest 全绿，2 秒短片 PSNR 34.85，20 秒全 7 相机 PSNR 27.60，外加一份 `WP_V1-1_Report.md`。但项目真正的"史诗"——V2-0 把 `MixtureOfGaussians` 重构成 LayeredGaussians——这 3 天里**一行都没写**。后面那段长征，会是另一份 timeline 的故事。

而那份 CLAUDE.md 里 10 条"A800 操作 + 文档同步严格把关清单"，正是这 3 天用 GPU 时间和断网调试买来的、最值钱的一份遗产。
