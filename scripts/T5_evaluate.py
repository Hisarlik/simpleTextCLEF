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
from easse.sari import corpus_sari
from pathlib import Path


def load_file(path):

    texts=[]
    with open(path, "r", encoding="utf8") as f:
        lines = f.readlines()
        for line in lines:
            texts.append(line.replace("\n",""))
    return texts


def evaluate(dataset, features):

    dm = SimplificationDataModule('t5-small', dataset, features)
    dm.load_data()

    PATH = "../resources/experiments/checkpoint-epoch=0.ckpt"
    model = T5SimplificationModel.load_from_checkpoint(PATH)
    print(model.hparams)

    trainer = Trainer(gpus=1)
    trainer.test(model, datamodule=dm)
    print(len(model.predictions))

    original_sents_paths = []
    simple_sents_paths = []

    for test_file in Path(WIKILARGE_CHUNK_DATASET).glob("*.test.complex"):
        original_sents_paths.append(test_file)

    for test_file in Path(WIKILARGE_CHUNK_DATASET).glob("*.test.simple"):
        simple_sents_paths.append(test_file)

    originals_sents = load_file(original_sents_paths[0])
    simple_sents = load_file(simple_sents_paths[0])

    print("Original", original_sents_paths)
    print("Simple", simple_sents_paths)

    print("SARI", corpus_sari(originals_sents,
                              model.predictions,
                              [simple_sents]))



if __name__ == "__main__":


    features_kwargs = {
        # 'WordRatioFeature': {'target_ratio': 1.05},
        'CharLengthRatio': {'target_ratio': 0.95},
        'LevenshteinRatio': {'target_ratio': 0.75},
        'WordRankRatio': {'target_ratio': 0.75},
        'DependencyTreeDepthRatio': {'target_ratio': 0.75}
    }


    evaluate(WIKILARGE_CHUNK_DATASET, features_kwargs)


