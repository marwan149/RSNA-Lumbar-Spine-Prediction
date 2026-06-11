from __future__ import annotations

import numpy as np
import pandas as pd
import cv2
import pydicom
from glob import glob
from pathlib import Path
from torch.utils.data import Dataset

from ..core import RSNAConfig
from ..core import natural_keys


class RSNA24TestDataset(Dataset):
    def __init__(self, config: RSNAConfig, study_ids: list[str], transform=None):
        self.config = config
        self.df = pd.read_csv(config.test_series_descriptions_path)
        self.study_ids = study_ids
        self.transform = transform

    def __len__(self) -> int:
        return len(self.study_ids)

    def get_img_paths(self, study_id: str, series_desc: str) -> list[str]:
        series_df = self.df[
            (self.df["study_id"] == study_id) &
            (self.df["series_description"] == series_desc)
        ]
        img_paths = []
        for _, row in series_df.iterrows():
            paths = sorted(
                glob(f"{self.config.test_images_dir / study_id / row['series_id']}/*.dcm"),
                key=natural_keys,
            )
            img_paths.extend(paths)
        return img_paths

    def read_dcm_image(self, src_path: str) -> np.ndarray:
        dicom_data = pydicom.dcmread(src_path)
        image = dicom_data.pixel_array
        norm_img = (image - image.min()) / (image.max() - image.min() + 1e-6) * 255
        resized_img = cv2.resize(norm_img, self.config.img_size, interpolation=cv2.INTER_CUBIC)
        return resized_img.astype(np.uint8)

    def load_series_images(self, study_id: str, series_desc: str) -> np.ndarray:
        images = np.zeros((*self.config.img_size, 14), dtype=np.uint8)
        img_paths = self.get_img_paths(study_id, series_desc)

        if not img_paths:
            raise FileNotFoundError(f"{study_id}: {series_desc} has no images")

        step = len(img_paths) / 14.0
        mid_point = len(img_paths) / 2.0 - 6.0 * step

        for j, i in enumerate(np.arange(mid_point, len(img_paths), step)):
            idx = max(0, int(round(i - 0.5)))
            images[..., j] = self.read_dcm_image(img_paths[idx])

        return images

    def __getitem__(self, idx: int) -> tuple[np.ndarray, str]:
        study_id = self.study_ids[idx]
        x = np.zeros((*self.config.img_size, self.config.in_chans), dtype=np.uint8)

        x[..., :14] = self.load_series_images(study_id, "Sagittal T1")
        x[..., 14:28] = self.load_series_images(study_id, "Sagittal T2/STIR")
        x[..., 28:] = self.load_series_images(study_id, "Axial T2")

        if self.transform:
            x = self.transform(image=x)["image"]

        x = x.transpose(2, 0, 1)
        return x, study_id
