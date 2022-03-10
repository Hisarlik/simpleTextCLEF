# -- fix path --
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --

from typing import Dict
from pathlib import Path

from easse.sari import corpus_sari
from pytorch_lightning import Trainer

from source.utils import storage
from source.optimization import Experiment
from source.data import SimplificationDataModule
from source.model import T5SimplificationModel
from conf import WIKILARGE_CHUNK_DATASET


def load_file(path):
    texts = []
    with open(path, "r", encoding="utf8") as f:
        lines = f.readlines()
        for line in lines:
            texts.append(line.replace("\n", ""))
    return texts


def evaluate(experiment: Experiment,
             dataset: Path,
             features: Dict):

    #dm = SimplificationDataModule('t5-small', dataset, features)
    dm = experiment.create_and_setup_data_module()
    trainer = experiment.create_trainer()
    model = experiment.load_best_model()
    #trainer = Trainer(gpus=1)
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
    features = dict(
        WordLengthRatio=dict(target_ratio=0.8),
        CharLengthRatio=dict(target_ratio=0.8),
        LevenshteinRatio=dict(target_ratio=0.8),
        DependencyTreeDepthRatio=dict(target_ratio=0.8),
        WordRankRatio=dict(target_ratio=0.8)
    )

    experiment_id = "20220310185939"
    experiment = storage.load_experiment(experiment_id)
    print(experiment)
    evaluate(experiment, WIKILARGE_CHUNK_DATASET, features)
