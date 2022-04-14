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
from source.evaluation import evaluate

if __name__ == "__main__":
    features = dict(
        WordLengthRatio=dict(target_ratio=0.75),
        CharLengthRatio=dict(target_ratio=0.65),
        LevenshteinRatio=dict(target_ratio=0.5),
        DependencyTreeDepthRatio=dict(target_ratio=0.8),
        WordRankRatio=dict(target_ratio=0.9)
    )

    experiment_id = None

    experiment = ExperimentManager.load_experiment(experiment_id)
    evaluate(experiment, SIMPLETEXT_DATASET, features)
