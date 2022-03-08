from pytorch_lightning import seed_everything
from datetime import datetime
from pathlib import Path
import torch
import pytorch_lightning as pl
from source.data import SimplificationDataModule
from source.model import T5SimplificationModel, LoggingCallback
from conf import OUTPUT_DIR


class Experiment():

    def __init__(self, hyperparameters, features):
        self.hparams = hyperparameters
        self.features = features

    def start(self):
        seed_everything(self.hparams['seed'])
        path = self._create_storage()
        trainer_configuration = self._create_trainer_configuration(path)
        trainer = pl.Trainer(**trainer_configuration)

        dm = SimplificationDataModule('t5-small',
                                      self.hparams.get("dataset_path"),
                                      self.features
                                      )
        dm.load_data()
        dm.setup("fit")

        model = T5SimplificationModel(**self.hparams)
        trainer.fit(model, datamodule=dm)

    def _create_storage(self):
        experiment_id = datetime.now().strftime('%Y%m%d%H%M%S')

        return Path(OUTPUT_DIR / experiment_id)

    def _create_trainer_configuration(self, path: Path):

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
