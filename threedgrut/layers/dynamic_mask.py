# SPDX-License-Identifier: Apache-2.0
"""Dynamic mask projection: cuboid → 2D AABB pixel mask (T4.4, U2 D5).

Why not sseg-based dyn mask: sseg covers vehicles outside our tracked set
(traffic cones, untracked pedestrians), so the dynamic_rigids layer would be
asked to explain pixels it cannot own. Cuboid projection ties dynamic loss
exactly to tracked actors.

Why AABB (not exact convex hull): cuboid projects to a 2D convex octagon;
AABB overestimates ≈10–15% in area but lands in 5 lines of vectorised
PyTorch (no scanline edge traversal needed). D5: upgrade to convex hull
only if Stage 7 KPI delta < +0.3 dB attributable to mask slack.

Why pure PyTorch (no OpenCV / nvdiffrast): keeps the dependency surface
minimal, runs on GPU without host copy, easy to differentiate later.

**T8/B3 addition** — FTheta polynomial branch. NCore wide-FOV cameras
(``camera_front_wide_120fov``, ``max_angle = 1.221 rad ≈ 70°``) wrap pinhole
projection past ±90° → ``u = fx * x / z`` saturates to ±∞ and clamps to the
image edges, painting whole columns as dyn mask. The FTheta path applies
``r_pix = angle_to_pixeldist_poly(atan2(r_xy, z))`` so off-axis cuboids land
in well-bounded AABBs. Behind-camera corners (``z ≤ 0``) are excluded from
the AABB; tracks with all 8 corners behind are skipped.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

# 8 cuboid corner sign template: ±1 along each of (x, y, z). Generated once
# at module load.
_CORNER_SIGNS = torch.tensor(
    [[(i & 1), ((i >> 1) & 1), ((i >> 2) & 1)] for i in range(8)],
    dtype=torch.float32,
) * 2.0 - 1.0  # [8, 3]; values in {-1, +1}


def _horner_ascending_torch(poly: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Evaluate p(x) = poly[0] + poly[1]*x + poly[2]*x^2 + ... (ascending).

    Mirrors ``threedgrut_playground/utils/ftheta_projector.py:_horner_ascending``
    for the FTheta polynomial; NCore's ``angle_to_pixeldist_poly`` and
    ``pixeldist_to_angle_poly`` are both stored ascending.
    """
    out = torch.zeros_like(x)
    for k in range(len(poly) - 1, -1, -1):
        out = out * x + poly[k]
    return out


def _normalize_ftheta_params(
    ftheta_params: Dict[str, object], device: torch.device | str,
) -> Dict[str, torch.Tensor]:
    """Normalize ``ftheta_params`` (numpy or torch) → torch tensors on device.

    Only the two keys actually consumed by the forward FTheta projection
    are returned (``angle_to_pixeldist_poly``, ``principal_point``); other
    8-key entries (``resolution``, ``max_angle``, ``linear_cde``, ...) are
    ignored — the caller's H/W already pins image bounds.
    """
    poly = ftheta_params["angle_to_pixeldist_poly"]
    pp = ftheta_params["principal_point"]
    if not torch.is_tensor(poly):
        poly = torch.as_tensor(poly)
    if not torch.is_tensor(pp):
        pp = torch.as_tensor(pp)
    return {
        "angle_to_pixeldist_poly": poly.to(device=device, dtype=torch.float32),
        "principal_point": pp.to(device=device, dtype=torch.float32),
    }


def _corners_to_pixels_pinhole(
    corners_cam: torch.Tensor, K: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pinhole projection. ``corners_cam`` shape ``[T, 8, 4]`` (homogeneous)."""
    z = corners_cam[..., 2].clamp(min=0.1)
    u = K[0, 0] * corners_cam[..., 0] / z + K[0, 2]
    v = K[1, 1] * corners_cam[..., 1] / z + K[1, 2]
    return u, v


def _aabb_from_visible_corners(
    u: torch.Tensor, v: torch.Tensor, z_pos: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-track AABB reduction over corners, ignoring behind-camera ones.

    Behind-camera corners (``z ≤ 0``) project to unbounded (FTheta) or
    z-clamped huge-magnitude (pinhole) pixels; letting them into the min/max
    smears the AABB across the image. A track with all 8 corners behind is
    flagged invisible so the caller skips it entirely.
    """
    per_track_visible = z_pos.any(dim=-1)                         # [T]
    big = torch.full_like(u, float("inf"))
    u_for_min = torch.where(z_pos, u, big)
    u_for_max = torch.where(z_pos, u, -big)
    v_for_min = torch.where(z_pos, v, big)
    v_for_max = torch.where(z_pos, v, -big)
    return (
        per_track_visible,
        u_for_min.min(dim=-1).values,
        u_for_max.max(dim=-1).values,
        v_for_min.min(dim=-1).values,
        v_for_max.max(dim=-1).values,
    )


def resolve_batch_cuboid_intrinsics(gpu_batch) -> Tuple[
    Optional[torch.Tensor], Optional[Dict[str, object]],
]:
    """Batch → ``(K, ftheta_params)`` for :func:`project_cuboids_to_mask`.

    Shared by ``trainer._maybe_fill_cuboid_mask`` and render.py's class_psnr
    eval so training and eval dispatch camera models identically (A5).
    FTheta wins when both are present, keeping FTheta clips byte-identical.
    The OpenCVPinhole path builds a distortion-free K from ``focal_length`` /
    ``principal_point``; radial/tangential/thin-prism coeffs are ignored —
    acceptable at AABB-mask granularity for 90/120° pinhole rigs.
    """
    ftheta = getattr(gpu_batch, "intrinsics_FThetaCameraModelParameters", None)
    if ftheta is not None:
        return None, ftheta
    pinhole = getattr(
        gpu_batch, "intrinsics_OpenCVPinholeCameraModelParameters", None,
    )
    if pinhole is None:
        return None, None

    def _pair(val) -> Tuple[float, float]:
        if torch.is_tensor(val):
            flat = val.detach().cpu().reshape(-1).tolist()
        else:
            flat = torch.as_tensor(val, dtype=torch.float64).reshape(-1).tolist()
        return float(flat[0]), float(flat[1] if len(flat) > 1 else flat[0])

    fx, fy = _pair(pinhole["focal_length"])
    cx, cy = _pair(pinhole["principal_point"])
    K = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32,
    )
    return K, None


def _corners_to_pixels_ftheta(
    corners_cam: torch.Tensor, ftheta_params: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FTheta polynomial projection. ``corners_cam`` shape ``[T, 8, 4]``.

    Same math as ``ftheta_projector.FthetaForwardProjector.project_points``
    but stays in torch (GPU-friendly, differentiable). The viser/ckpt
    ``Y-down Z-backward`` axis flip done in the overlay path does NOT apply
    here — the training dataset hands us ``T_world2cam`` already in OpenCV
    convention (Y down, Z forward).
    """
    cam = corners_cam[..., :3]
    x, y, z = cam[..., 0], cam[..., 1], cam[..., 2]
    r_xy = torch.sqrt(x * x + y * y)
    # atan2 handles z ≤ 0 correctly (returns ∈ [π/2, π]); the angle is
    # then clipped against max_angle implicitly via the polynomial extrapolating,
    # and we drop those corners from the AABB via the z>0 visibility mask below.
    angle = torch.atan2(r_xy, z)
    r_pix = _horner_ascending_torch(ftheta_params["angle_to_pixeldist_poly"], angle)
    safe_r = torch.where(r_xy < 1e-9, torch.ones_like(r_xy), r_xy)
    cx = ftheta_params["principal_point"][0]
    cy = ftheta_params["principal_point"][1]
    u = cx + r_pix * x / safe_r
    v = cy + r_pix * y / safe_r
    return u, v


def project_cuboids_to_mask(
    tracks_poses: torch.Tensor,    # [T, 4, 4] active tracks at this frame
    tracks_size: torch.Tensor,     # [T, 3] full extent (NOT half)
    K: Optional[torch.Tensor],     # [3, 3] pinhole intrinsics; pass None when ftheta_params is set
    T_world2cam: torch.Tensor,     # [4, 4]
    H: int,
    W: int,
    device: torch.device | str = "cpu",
    *,
    ftheta_params: Optional[Dict[str, object]] = None,
) -> torch.Tensor:
    """Project each cuboid → 8 image-space corners → 2D AABB → fill mask.

    Args:
        tracks_poses: ``[T, 4, 4]`` object→world for T active tracks at
            this frame. T may be 0 (no active tracks → empty mask).
        tracks_size:  ``[T, 3]`` cuboid full extent in meters.
        K:            ``[3, 3]`` pinhole intrinsics (fx, fy, cx, cy). Pass
                      ``None`` when ``ftheta_params`` is provided.
        T_world2cam:  ``[4, 4]`` world → camera SE(3) in OpenCV convention.
        H, W:         output mask size in pixels.
        device:       output device.
        ftheta_params: optional dict with keys ``angle_to_pixeldist_poly``
                      (ascending coefficients) and ``principal_point`` (cx, cy).
                      When provided, routes through the FTheta polynomial
                      instead of pinhole — required for wide-FOV cameras
                      (B3 fix in T8_buglists.md).

    Returns:
        bool ``[H, W]`` mask; True wherever any cuboid AABB covers.
    """
    T = int(tracks_poses.shape[0])
    mask = torch.zeros(H, W, dtype=torch.bool, device=device)
    if T == 0:
        return mask

    poses = tracks_poses.to(device=device, dtype=torch.float32)
    sizes = tracks_size.to(device=device, dtype=torch.float32)
    T_w2c = T_world2cam.to(device=device, dtype=torch.float32)
    signs = _CORNER_SIGNS.to(device=device)                        # [8, 3]

    # 1. 8 cuboid corners in object-local frame: half-extent × ±sign
    corners_local = signs.unsqueeze(0) * (sizes.unsqueeze(1) * 0.5)  # [T, 8, 3]
    ones = torch.ones(T, 8, 1, dtype=torch.float32, device=device)
    corners_h = torch.cat([corners_local, ones], dim=-1)             # [T, 8, 4]

    # 2. local → world (per-track pose) → camera (single T_w2c)
    world = torch.einsum("tij,tkj->tki", poses, corners_h)           # [T, 8, 4]
    cam = torch.einsum("ij,tkj->tki", T_w2c, world)                  # [T, 8, 4]

    # 3. project corners to pixels via the requested intrinsics model
    if ftheta_params is not None:
        ftheta = _normalize_ftheta_params(ftheta_params, device)
        u, v = _corners_to_pixels_ftheta(cam, ftheta)
    else:
        if K is None:
            raise ValueError("project_cuboids_to_mask: pass K or ftheta_params")
        K_dev = K.to(device=device, dtype=torch.float32)
        u, v = _corners_to_pixels_pinhole(cam, K_dev)
    # Ignore behind-camera corners when computing the AABB; a track with all
    # 8 corners behind is skipped (per_track_visible == False). A5: applied
    # to BOTH branches — the pinhole z.clamp(min=0.1) otherwise projects
    # behind corners at huge |u|,|v| and smears the AABB across the image
    # (the T8/B3 column smear that originally made the trainer skip pinhole).
    z_pos = cam[..., 2] > 0                                           # [T, 8]
    per_track_visible, u_min_f, u_max_f, v_min_f, v_max_f = (
        _aabb_from_visible_corners(u, v, z_pos)
    )

    # 4. clip 2D AABB to image bounds
    u_min = u_min_f.clamp(0, W - 1).long()                            # [T]
    u_max = u_max_f.clamp(0, W - 1).long()
    v_min = v_min_f.clamp(0, H - 1).long()
    v_max = v_max_f.clamp(0, H - 1).long()

    # 5. Fill mask. Loop over T (typically < 30 active actors per frame; the
    # python overhead is negligible vs the einsum cost above). Slice
    # assignment is in-place; Boolean OR semantics emerge naturally because
    # writes set True over existing False.
    for i in range(T):
        if not bool(per_track_visible[i]):
            continue
        mask[v_min[i]:v_max[i] + 1, u_min[i]:u_max[i] + 1] = True
    return mask
