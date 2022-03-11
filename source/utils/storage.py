import os
import pickle
from pathlib import Path
from source.optimization import Experiment
from conf import OUTPUT_DIR


def save_experiment(experiment: Experiment):
    path = experiment.experiment_path / "experiment.pickle"
    with open(path, "wb") as f:
        pickle.dump(experiment, f)


def load_experiment(experiment_id: str = None):
    if experiment_id:
        experiment_path = OUTPUT_DIR / experiment_id / "experiment.pickle"

    else:
        experiments = [path for path in OUTPUT_DIR.iterdir() if os.path.isdir(path)]
        experiment_path = sorted(experiments, reverse=True)[0] / "experiment.pickle"

    with open(experiment_path, "rb") as f:
        experiment = pickle.load(f)

    return experiment


