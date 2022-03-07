from pytorch_lightning import seed_everything

from source.data import SimplificationDataModule
from source.model import T5SimplificationModel
import pytorch_lightning as pl

class Experiment():

    def __init__(self, model_name, dataset_name, features, trainer_hyperparameters, model_hyperparameters):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.features = features
        self.trainer_hyperparameters = trainer_hyperparameters
        self.model_hyperparameters = model_hyperparameters

    def start(self):
        seed_everything(self.model_hyperparameters['seed'])

        dm = SimplificationDataModule('t5-small',
                                      self.dataset_name,
                                      self.features
                                      )
        dm.load_data()
        dm.setup("fit")

        model = T5SimplificationModel(**self.model_hyperparameters)
        trainer = pl.Trainer(**self.trainer_hyperparameters)
        trainer.fit(model, datamodule=dm)