import pickle
from pathlib import Path
from source.optimization import Experiment
from conf import OUTPUT_DIR

def save_experiment(experiment: Experiment):
    path = experiment.experiment_path / "experiment.pickle"
    with open(path, "wb") as f:
        pickle.dump(experiment, f)


def load_experiment(experiment_id: str):
    path = OUTPUT_DIR / experiment_id / "experiment.pickle"
    with open(path, "rb") as f:
        experiment = pickle.load(f)

    return experiment


