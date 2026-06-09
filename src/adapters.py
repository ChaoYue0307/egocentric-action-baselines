from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class FrameSource(Protocol):
    """Provides video frames for RGB feature extraction."""

    video_path: Path


class HandPoseSource(Protocol):
    """Provides left and right 3D hand joints aligned to frame indices."""

    annotation_path: Path


class LabelSource(Protocol):
    """Provides action or subtask labels aligned to frame indices."""

    annotation_path: Path


@dataclass(frozen=True)
class XperienceActionAdapter:
    """Small boundary object for the Xperience-10M sample layout."""

    data_root: Path
    video_name: str = "fisheye_cam0.mp4"

    @property
    def video_path(self) -> Path:
        return self.data_root / self.video_name

    @property
    def annotation_path(self) -> Path:
        return self.data_root / "annotation.hdf5"

    def describe(self) -> dict:
        return {
            "adapter": "XperienceActionAdapter",
            "video_path": str(self.video_path),
            "annotation_path": str(self.annotation_path),
            "signals": ["rgb_frames", "hand_joints_3d", "caption_action_labels"],
        }
