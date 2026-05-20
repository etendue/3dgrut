# SPDX-License-Identifier: Apache-2.0
"""4D viz helpers for v2 LayeredGaussians ckpts.

Pure CPU; no viser / kaolin / OptiX dependency. Imported by both the trainer
(for ckpt write) and the viser_gui_4d viewer (for ckpt read + dataset
fallback).
"""

from threedgrut.viz.metadata import extract_4d_metadata

__all__ = ["extract_4d_metadata"]
