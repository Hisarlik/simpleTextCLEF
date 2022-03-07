# -- fix path -- Source: https://github.com/KimChengSHEANG/TS_T5
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --

import torch
from utils import logging_module
from source.optimization import Experiment
from source.model import LoggingCallback
from conf import WIKILARGE_CHUNK_DATASET, PREPROCESSED_DIR, OUTPUT_DIR, DEVICE
import pytorch_lightning as pl

logger = logging_module.get_logger(__name__)


def main(model_hyperparameters, trainer_params, features):
    experiment = Experiment(
        model_hyperparameters['model_name'],
        model_hyperparameters['dataset_path'],
        features,
        trainer_params,
        model_hyperparameters
    )

    experiment.start()


if __name__ == "__main__":

    features = dict(
        WordLengthRatio=dict(target_ratio=0.8),
        CharLengthRatio=dict(target_ratio=0.8),
        LevenshteinRatio=dict(target_ratio=0.8),
        DependencyTreeDepthRatio=dict(target_ratio=0.8),
        WordRankRatio=dict(target_ratio=0.8)
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=OUTPUT_DIR,
        filename="checkpoint-{epoch}",
        monitor="val_loss",
        verbose=True,
        mode="min",
        save_top_k=5)

    trainer_params = dict(
        accelerator="gpu",
        accumulate_grad_batches=1,
        gpus=torch.cuda.device_count(),
        max_epochs=1,
        precision=32,
        callbacks=[LoggingCallback(), checkpoint_callback],
        num_sanity_val_steps=0,
        progress_bar_refresh_rate=1,
    )

    model_hyperparameters = dict(
        model_name='t5-small',
        max_seq_length=32,
        learning_rate=3e-4,
        weight_decay=0.1,
        adam_epsilon=1e-8,
        warmup_steps=5,
        train_batch_size=6,
        valid_batch_size=6,
        num_train_epochs=5,
        custom_loss=False,
        gradient_accumulation_steps=1,  # 16
        n_gpu=torch.cuda.device_count(),
        fp_16=False,
        opt_level='O1',
        max_grad_norm=1.0,
        seed=42,
        nb_sanity_val_steps=0,
        train_sample_size=1,
        valid_sample_size=1,
        dataset_path=WIKILARGE_CHUNK_DATASET,
        preprocess_dir=PREPROCESSED_DIR,
        output_dir=OUTPUT_DIR,
        device=DEVICE
    )

    main(model_hyperparameters, trainer_params, features)
