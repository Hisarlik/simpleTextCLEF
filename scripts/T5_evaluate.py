# -- fix path --
from pathlib import Path
import sys

from pytorch_lightning import Trainer

sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from source.data import SimplificationDataModule
from source.model import T5SimplificationModel
from conf import WIKILARGE_CHUNK_DATASET


def evaluate(dataset, features):

    dm = SimplificationDataModule('t5-small', dataset, features)
    dm.load_data()

    PATH = "../resources/experiments/checkpoint-epoch=0.ckpt"
    model = T5SimplificationModel.load_from_checkpoint(PATH)
    print(model.hparams)

    trainer = Trainer(gpus=1)
    trainer.test(model, datamodule=dm)
    print(len(model.predictions))




if __name__ == "__main__":


    features_kwargs = {
        # 'WordRatioFeature': {'target_ratio': 1.05},
        'CharLengthRatio': {'target_ratio': 0.95},
        'LevenshteinRatio': {'target_ratio': 0.75},
        'WordRankRatio': {'target_ratio': 0.75},
        'DependencyTreeDepthRatio': {'target_ratio': 0.75}
    }


    evaluate(WIKILARGE_CHUNK_DATASET, features_kwargs)


