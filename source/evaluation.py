# -- fix path --
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --

from typing import Dict
from pathlib import Path

from source.experiments import Experiment


def evaluate(experiment: Experiment,
             dataset: Path,
             features: Dict,
             metrics : bool = True):
    dm = experiment.create_and_setup_data_module(dataset, features, "test")
    trainer = experiment.create_trainer()
    model = experiment.load_best_model()
    trainer.test(model, datamodule=dm)
    if metrics:
        return experiment.get_metrics(model, dm)
