from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class RSNAConfig:
    data_dir: Path
    output_path: Path
    checkpoint_paths: List[Path]
    batch_size: int = 1
    num_workers: int | None = None
    use_amp: bool = True
    device: str | None = None
    img_size: tuple[int, int] = (512, 512)
    in_chans: int = 42
    n_labels: int = 25
    n_classes: int = 3 * 25
    model_name: str = "edgenext_base.in21k_ft_in1k"
    conditions: list[str] = None
    levels: list[str] = None

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_path = Path(self.output_path)
        self.checkpoint_paths = [Path(p) for p in self.checkpoint_paths]
        self.num_workers = self.num_workers if self.num_workers is not None else (os.cpu_count() or 1)
        self.device = self.device or ("cuda:0" if self.has_cuda else "cpu")
        self.conditions = self.conditions or [
            "spinal_canal_stenosis",
            "left_neural_foraminal_narrowing",
            "right_neural_foraminal_narrowing",
            "left_subarticular_stenosis",
            "right_subarticular_stenosis",
        ]
        self.levels = self.levels or [
            "l1_l2",
            "l2_l3",
            "l3_l4",
            "l4_l5",
            "l5_s1",
        ]

    @property
    def has_cuda(self) -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    @classmethod
    def from_data_dir(
        cls,
        data_dir: Path | str,
        output_path: Path | str,
        checkpoint_paths: Iterable[Path | str],
        batch_size: int = 1,
        num_workers: int | None = None,
        use_amp: bool = True,
        device: str | None = None,
    ) -> "RSNAConfig":
        return cls(
            data_dir=Path(data_dir),
            output_path=Path(output_path),
            checkpoint_paths=[Path(p) for p in checkpoint_paths],
            batch_size=batch_size,
            num_workers=num_workers,
            use_amp=use_amp,
            device=device,
        )

    @property
    def test_series_descriptions_path(self) -> Path:
        return self.data_dir / "test_series_descriptions.csv"

    @property
    def sample_submission_path(self) -> Path:
        return self.data_dir / "sample_submission.csv"

    @property
    def test_images_dir(self) -> Path:
        return self.data_dir / "test_images"
