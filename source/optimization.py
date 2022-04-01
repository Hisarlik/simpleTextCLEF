import os
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from easse.sari import corpus_sari

from conf import OUTPUT_DIR
from source.utils import logging_module, storage
from source.data import SimplificationDataModule
from source.model import T5SimplificationModel, LoggingCallback, LitProgressBar

logger = logging_module.get_logger(__name__)


@dataclass
class Experiment:
    """Class for controlling experiments (Different features or hyperparameters) """

    experiment_path: Path = field(init=False)
    hparams: Dict
    features: Dict

    def __post_init__(self):
        self.experiment_path = self._create_experiment_id()

    def start(self):
        logger.info(f"Init experiment. hparams: {self.hparams}. features:{self.features.keys()}")
        seed_everything(self.hparams['seed'], workers=True)

        # Get the hyperparameters of the trainer and instantiate it
        trainer = self.create_trainer()
        dataset_path = self.hparams.get('dataset_path')

        # Creating Pytorch Lightning DataModule
        dm = self.create_and_setup_data_module(dataset_path, self.features, "fit")

        # Creating T5 simplification model.
        # TODO: Modify to instantiate the model class using its name. e.g T5 -> T5SimplificationModel-
        # This allow us to use different models without changing code.
        self.hparams['experiment_path'] = self.experiment_path
        model = T5SimplificationModel(**self.hparams)
        trainer.fit(model, datamodule=dm)

    def create_trainer(self) -> pl.Trainer:
        trainer_configuration = self._create_trainer_configuration()
        trainer = pl.Trainer(**trainer_configuration)
        return trainer

    def create_and_setup_data_module(self,
                                     dataset: Path,
                                     features: Dict,
                                     stage: Optional[str]) -> SimplificationDataModule:
        """Method to create a datamodule for training or testing the model"""

        dm = SimplificationDataModule(self.hparams.get("model_name"),
                                      dataset,
                                      features,
                                      self.hparams.get("max_seq_length"),
                                      self.hparams.get("train_batch_size"),
                                      self.hparams.get("valid_batch_size")
                                      )
        dm.setup(stage)
        return dm

    def load_best_model(self):
        checkpoints = self.experiment_path.glob('checkpoint*')
        best_checkpoint = str(sorted(checkpoints, reverse=True)[0])
        model = T5SimplificationModel.load_from_checkpoint(best_checkpoint)
        return model

    def _create_experiment_id(self) -> Path:
        experiment_id = datetime.now().strftime('%Y%m%d%H%M%S')
        return Path(OUTPUT_DIR / experiment_id)

    def _create_trainer_configuration(self) -> Dict:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=self.experiment_path,
            filename="checkpoint-{epoch}",
            monitor="val_loss",
            verbose=True,
            mode="min",
            save_top_k=5)

        trainer_conf = dict(
            accelerator=self.hparams.get("accelerator", "gpu"),
            accumulate_grad_batches=1,
            gpus=torch.cuda.device_count(),
            max_epochs=self.hparams.get("number_epochs", 1),
            precision=16 if self.hparams.get("fp_16") else 32,
            callbacks=[LoggingCallback(), checkpoint_callback, LitProgressBar()],
            num_sanity_val_steps=0,
        )

        return trainer_conf

    def get_metrics(self, model: pl.LightningModule, dm: SimplificationDataModule):

        model_features = dm.get_features_and_values_string()
        dataset_path = dm.data_path
        predictions = model.predictions

        original_sents = []
        simple_sents = []

        for test_file in Path(dataset_path).glob("*.test.complex*"):
            original_sents = storage.load_file(test_file)

        for test_file in Path(dataset_path).glob("*.test.simple*"):
            test_text = storage.load_file(test_file)
            simple_sents.append(test_text)

        score = corpus_sari(original_sents, predictions, simple_sents)

        logger.info(f"test SARI: {score}")

        result_path = Path(self.experiment_path / \
                           dataset_path.name / \
                           model_features / \
                           "test_results.txt")


        if predictions:
            storage.save_text_file(result_path, predictions)


class ExperimentManager:

    @staticmethod
    def create_experiment(model_hyperparameters, features):
        experiment = Experiment(
            model_hyperparameters,
            features
        )
        return experiment

    @staticmethod
    def load_experiment(experiment_id: str = None):
        if experiment_id:
            experiment_path = OUTPUT_DIR / experiment_id / "experiment.pickle"

        else:
            experiments = [path for path in OUTPUT_DIR.iterdir() if os.path.isdir(path)]
            experiment_path = sorted(experiments, reverse=True)[0] / "experiment.pickle"

        experiment = storage.load_object(experiment_path)

        return experiment

    @staticmethod
    def save_experiment(experiment: Experiment):
        path = experiment.experiment_path / "experiment.pickle"
        storage.save_object(path, experiment)
