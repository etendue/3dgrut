# SPDX-License-Identifier: Apache-2.0
"""Post-process novel-view rendered images through NVIDIA Fixer (Difix3D+).

V3-T15.2 (HF main path). Wraps the vendored ``third_party/Fixer/`` model class
so ``render.py`` can compute ``psnr_difix`` / ``ssim_difix`` / ``lpips_difix``
alongside the raw metrics, without forcing any DiFix dependency onto the
3dgrut2 main env.

Lazy-import contract (critical): importing this module pulls only ``torch``.
The ``Pix2Pix_Turbo`` model and its ``cosmos_predict2`` / ``imaginaire`` /
``transformer_engine`` dependencies are imported on the first ``forward()``
call that actually needs them — Mac dev machines without those deps can still
import ``threedgrut.correction.difix`` (e.g. via ``render.py`` startup) as long
as ``enabled=False``. See ``third_party/Fixer/INSTALL.md`` for the GPU env.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# DiFix's training input shape (height, width). README maps resolution=1024 to
# (1024, 576); the model file uses h=1024, w=576 as fixed inference dims.
_DIFIX_H: int = 1024
_DIFIX_W: int = 576


class DifixPostProcessor(nn.Module):
    """Single-step image refiner wrapping NVIDIA Fixer.

    Args:
        enabled: master switch. When ``False``, ``forward`` is the identity.
            Default ``False`` so the module costs nothing for runs that do not
            opt in.
        ckpt_path: absolute path to ``pretrained_fixer.pkl``. If ``None``,
            falls back to ``$HF_HOME/nvidia-Fixer/pretrained_fixer.pkl``.
        timestep: diffusion timestep used at inference. README's
            ``inference_pretrained_model.py`` defaults to 400 but the user-
            requested CLI in the README example uses 250.
        target_h, target_w: DiFix native resolution. Inputs are bilinear-
            resized to ``(target_h, target_w)`` and resized back to the
            caller's ``(H, W)``.
        dtype: torch dtype used inside the model. README inference uses
            ``bfloat16`` on CUDA.
    """

    def __init__(
        self,
        enabled: bool = False,
        ckpt_path: Optional[str] = None,
        timestep: int = 250,
        target_h: int = _DIFIX_H,
        target_w: int = _DIFIX_W,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self._ckpt_path = ckpt_path
        self._timestep = int(timestep)
        self._target_h = int(target_h)
        self._target_w = int(target_w)
        self._dtype = dtype
        # The heavy ``Pix2Pix_Turbo`` instance, populated by ``_lazy_init`` on
        # first use. Not an ``nn.Module`` attribute so we don't accidentally
        # save its weights into 3dgrut2 checkpoints (DiFix weights stay on disk).
        self._model = None

    def _resolve_ckpt_path(self) -> Path:
        """Path to the Pix2Pix_Turbo state_dict pickle.

        Real layout from ``hf download nvidia/Fixer``:
            $HF_HOME/nvidia-Fixer/pretrained/pretrained_fixer.pkl  (3.8 GB)
            $HF_HOME/nvidia-Fixer/base/model_fast_tokenizer.pt     (1.2 GB)
            $HF_HOME/nvidia-Fixer/base/tokenizer_fast.pth          (221 MB)
        """
        if self._ckpt_path:
            return Path(self._ckpt_path)
        import os
        hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        return Path(hf_home) / "nvidia-Fixer" / "pretrained" / "pretrained_fixer.pkl"

    def _ensure_tokenizer_symlinks(self, ckpt: Path) -> None:
        """Bridge HF cache layout to vendored pix2pix_turbo_*.py hardcoded paths.

        ``third_party/Fixer/pix2pix_turbo_nocond_cosmos_base_faster_tokenizer.py``
        hardcodes:
            config.dit_path           = '/work/models/base/model_fast_tokenizer.pt'
            config.tokenizer["vae_pth"] = '/work/models/base/tokenizer_fast.pth'

        These are the Docker layout. Off-Docker (e.g. Vast.ai), we symlink
        ``/work/models/base/{dit,vae}`` to the HF cache's ``base/`` siblings
        so the import-time ``Pix2Pix_Turbo.initialize_cosmos_model()`` finds
        them. Idempotent.
        """
        import os
        base_dir = ckpt.parent.parent / "base"   # nvidia-Fixer/base/
        targets = {
            Path("/work/models/base/model_fast_tokenizer.pt"): base_dir / "model_fast_tokenizer.pt",
            Path("/work/models/base/tokenizer_fast.pth"):     base_dir / "tokenizer_fast.pth",
        }
        for link, real in targets.items():
            if not real.exists():
                raise RuntimeError(
                    f"DiFix tokenizer file missing in HF cache: {real}. "
                    "Re-run scripts/download_difix.sh — incomplete download?"
                )
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink() or link.exists():
                try:
                    if link.resolve() == real.resolve():
                        continue  # already correct
                    link.unlink()
                except (OSError, FileNotFoundError):
                    pass
            os.symlink(real, link)

    def _lazy_init(self) -> None:
        if self._model is not None:
            return

        ckpt = self._resolve_ckpt_path()
        if not ckpt.exists():
            raise RuntimeError(
                f"DiFix enabled but checkpoint missing: {ckpt}. "
                "Run scripts/download_difix.sh on the GPU host first "
                "(see third_party/Fixer/INSTALL.md)."
            )

        # Bridge HF cache layout to vendored pix2pix's hardcoded /work/models/base/ paths.
        self._ensure_tokenizer_symlinks(ckpt)

        # Add third_party/Fixer/ to sys.path so the vendored
        # pix2pix_turbo_*.py's `from model import ...` (line 49) resolves to
        # third_party/Fixer/model.py rather than failing as a top-level
        # `model` module miss. Verified on ThinkPad cosmos container 2026-06-01.
        import sys
        fixer_dir = Path(__file__).resolve().parent.parent.parent / "third_party" / "Fixer"
        if str(fixer_dir) not in sys.path:
            sys.path.insert(0, str(fixer_dir))

        # Lazy import so this module loads cleanly on hosts (e.g. Mac dev) that
        # do not have cosmos_predict2 / transformer_engine / imaginaire.
        try:
            from third_party.Fixer.pix2pix_turbo_nocond_cosmos_base_faster_tokenizer import (
                Pix2Pix_Turbo,
            )
        except ImportError as exc:
            raise RuntimeError(
                "DiFix enabled but its dependencies are not installed. "
                "See third_party/Fixer/INSTALL.md to set up cosmos_predict2 / "
                "imaginaire / transformer_engine on the GPU host. "
                f"Original error: {exc}"
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError(
                "DiFix requires CUDA. Pix2Pix_Turbo hardcodes device='cuda' "
                "at construction time; Mac/CPU dev must keep enabled=False."
            )

        model = Pix2Pix_Turbo(
            pretrained_path=str(ckpt),
            timestep=self._timestep,
            vae_skip_connection=False,
            batch_size=1,
        ).to(device="cuda", dtype=self._dtype)
        model.eval()
        model.requires_grad_(False)
        self._model = model

    @torch.no_grad()
    def forward(self, image: Tensor) -> Tensor:
        """Run DiFix on ``image`` and return a same-shape tensor.

        Args:
            image: ``(B, H, W, 3)`` or ``(H, W, 3)``, float in [0, 1], on cuda
                when ``enabled=True``.

        Returns:
            Same shape / dtype / device / value range as ``image``.
        """
        if not self.enabled:
            return image

        single = image.dim() == 3
        if single:
            image = image.unsqueeze(0)
        if image.dim() != 4 or image.shape[-1] != 3:
            raise ValueError(
                f"DiFix forward expects (B,H,W,3) or (H,W,3), got {tuple(image.shape)}"
            )

        self._lazy_init()
        orig_dtype = image.dtype
        orig_h, orig_w = image.shape[1], image.shape[2]

        # NHWC -> NCHW, [0,1] -> [-1,1], cast + resize to DiFix native size.
        x = image.permute(0, 3, 1, 2).contiguous()         # (B,3,H,W)
        x = x * 2.0 - 1.0
        x = F.interpolate(
            x,
            size=(self._target_h, self._target_w),
            mode="bilinear",
            align_corners=False,
        )
        x = x.to(device="cuda", dtype=self._dtype)

        # nv-tlabs/Fixer inference_pretrained_model.py::model_inference wraps
        # forward in autocast so the DiT (which holds some fp32 buffers /
        # parameters internally even after .to(bfloat16)) sees a matching
        # dtype at the x_embedder Linear. Without this we get
        # ``RuntimeError: expected mat1 and mat2 to have the same dtype,
        # but got: float != c10::BFloat16`` mid-forward.
        with torch.autocast(device_type="cuda", dtype=self._dtype, enabled=True):
            y = self._model(x)                              # (B,3,target_h,target_w)

        # Resize back, denormalize, clamp, NCHW -> NHWC, restore dtype.
        y = F.interpolate(
            y.float(),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )
        y = (y * 0.5 + 0.5).clamp(0.0, 1.0)
        y = y.permute(0, 2, 3, 1).contiguous().to(dtype=orig_dtype)

        if single:
            y = y.squeeze(0)
        return y
