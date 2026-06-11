from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A

from ..core import RSNAConfig
from ..datasets import RSNA24TestDataset
from ..models import RSNA24Model


def get_test_transform(config: RSNAConfig):
    return A.Compose([
        A.Resize(config.img_size[0], config.img_size[1]),
        A.Normalize(mean=0.5, std=0.5),
    ])


def load_models(config: RSNAConfig) -> list[RSNA24Model]:
    models = []
    for checkpoint_path in config.checkpoint_paths:
        model = RSNA24Model(
            model_name=config.model_name,
            in_c=config.in_chans,
            n_classes=config.n_classes,
            pretrained=False,
        )
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        model.half()
        model.to(config.device)
        models.append(model)
    return models


def run_submission_prediction(config: RSNAConfig) -> None:
    sample_submission = pd.read_csv(config.sample_submission_path)
    labels = sample_submission.columns[1:].tolist()

    df = pd.read_csv(config.test_series_descriptions_path)
    study_ids = df["study_id"].unique().tolist()

    transforms_test = get_test_transform(config)
    test_ds = RSNA24TestDataset(config, study_ids, transform=transforms_test)
    test_dl = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    models = load_models(config)
    autocast = torch.cuda.amp.autocast(enabled=config.use_amp, dtype=torch.half)

    all_preds = []
    row_names = []

    with torch.no_grad():
        for x, study_id in tqdm(test_dl, leave=True):
            x = x.to(config.device)
            pred_per_study = np.zeros((config.n_labels, 3), dtype=np.float32)

            for cond in config.conditions:
                for level in config.levels:
                    row_names.append(f"{study_id[0]}_{cond}_{level}")

            with autocast:
                for model in models:
                    y = model(x)[0]
                    for col in range(config.n_labels):
                        pred = y[col * 3 : (col + 1) * 3]
                        y_pred = pred.float().softmax(dim=0).cpu().numpy()
                        pred_per_study[col] += y_pred / len(models)

            all_preds.append(pred_per_study)

    y_preds = np.concatenate(all_preds, axis=0)
    submission_df = pd.DataFrame({
        "row_id": row_names,
        **{label: y_preds[:, i] for i, label in enumerate(labels)},
    })
    submission_df.to_csv(config.output_path, index=False)
    print(f"Saved submission to {config.output_path}")
