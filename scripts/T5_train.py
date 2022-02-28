# -- fix path -- Source: https://github.com/KimChengSHEANG/TS_T5
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --

import torch
from utils import logging_module
from source.data import SimplificationDataModule
from source.model import T5SimplificationModel, LoggingCallback
from conf import WIKILARGE_CHUNK_DATASET, PREPROCESSED_DIR, OUTPUT_DIR, DEVICE
import pytorch_lightning as pl
logger = logging_module.get_logger(__name__)

if __name__ == "__main__":


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
        seed=12,
        nb_sanity_val_steps=0,
        train_sample_size=1,
        valid_sample_size=1,
        dataset_path=WIKILARGE_CHUNK_DATASET,
        preprocess_dir=PREPROCESSED_DIR,
        output_dir=OUTPUT_DIR,
        device=DEVICE
    )



    features = {
        'WordLengthRatio': {'target_ratio': 0.8},
        'CharLengthRatio': {'target_ratio': 0.8},
        'LevenshteinRatio': {'target_ratio': 0.8},
        'DependencyTreeDepthRatio': {'target_ratio': 0.8},
        'WordRankRatio': {'target_ratio': 0.8}
    }

    conf = {
                "model_hyperparameters": model_hyperparameters,
                "features": features

    }

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=OUTPUT_DIR,
        filename="checkpoint-{epoch}",
        monitor="val_loss",
        verbose=True,
        mode="min",
        save_top_k=5)

    train_params = dict(
        accelerator="gpu",
        accumulate_grad_batches=1,
        gpus=torch.cuda.device_count(),
        max_epochs=1,
        precision=32,
        callbacks=[LoggingCallback(), checkpoint_callback],
        num_sanity_val_steps=0,
        progress_bar_refresh_rate=1,
    )

    dm = SimplificationDataModule('t5-small', WIKILARGE_CHUNK_DATASET, features)
    dm.load_data()
    dm.setup("fit")


    model = T5SimplificationModel(**model_hyperparameters)

    trainer = pl.Trainer(**train_params)
    trainer.fit(model, datamodule=dm)