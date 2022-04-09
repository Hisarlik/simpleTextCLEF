# -- fix path --
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --

from typing import Dict
from pathlib import Path

from source.experiments import Experiment
from conf import WIKILARGE_CHUNK_DATASET, TURKCORPUS_DATASET, WIKILARGE_DATASET, SIMPLETEXT_DATASET
from source.experiments import ExperimentManager


def evaluate(experiment: Experiment,
             dataset: Path,
             features: Dict):

    dm = experiment.create_and_setup_data_module(dataset, features, "test")
    trainer = experiment.create_trainer()
    model = experiment.load_best_model()
    trainer.test(model, datamodule=dm)
    return experiment.get_metrics(model, dm)