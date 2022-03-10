from typing import Dict
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

import torch
from pytorch_lightning import seed_everything
import pytorch_lightning as pl

from conf import OUTPUT_DIR
from utils import logging_module
from source.data import SimplificationDataModule
from source.model import T5SimplificationModel, LoggingCallback

logger = logging_module.get_logger(__name__)

@dataclass
class Experiment:
    """Class for controlling experiments (Different features or hyperparameters) """

    hparams: Dict
    features: Dict

    def start(self):
        logger.info(f"Init experiment. hparams: {self.hparams}. features:{self.features.keys()}")
        seed_everything(self.hparams['seed'], workers=True)

        # Creating path to store the experiment
        path = self._create_experiment_id()

        # Get the hyperparameters of the trainer and instantiate it
        trainer_configuration = self._create_trainer_configuration(path)
        trainer = pl.Trainer(**trainer_configuration)

        # Creating Pytorch Lightning DataModule
        dm = SimplificationDataModule(self.hparams.get("model_name"),
                                      self.hparams.get("dataset_path"),
                                      self.features,
                                      self.hparams.get("max_seq_length"),
                                      self.hparams.get("train_batch_size"),
                                      self.hparams.get("valid_batch_size")
                                      )
        dm.load_data()
        dm.setup()

        # Creating T5 simplification model.
        # TODO: Modify to instantiate the model class using its name. e.g T5 -> T5SimplificationModel-
        # This allow us to use different models without changing code.
        model = T5SimplificationModel(**self.hparams)
        trainer.fit(model, datamodule=dm)

    def _create_experiment_id(self) -> Path:
        experiment_id = datetime.now().strftime('%Y%m%d%H%M%S')
        return Path(OUTPUT_DIR / experiment_id)

    def _create_trainer_configuration(self, path: Path) -> Dict:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=path,
            filename="checkpoint-{epoch}",
            monitor="val_loss",
            verbose=True,
            mode="min",
            save_top_k=5)

        trainer_conf = dict(
            accelerator=self.hparams.get("accelerator", "gpu"),
            accumulate_grad_batches=1,
            gpus=torch.cuda.device_count(),
            max_epochs=self.hparams.get("max_epochs", 1),
            precision=16 if self.hparams.get("fp_16") else 32,
            callbacks=[LoggingCallback(), checkpoint_callback],
            num_sanity_val_steps=0,
            progress_bar_refresh_rate=1,
        )

        return trainer_conf
