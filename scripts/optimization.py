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

import optuna

from conf import WIKILARGE_CHUNK_DATASET
from source.experiments import ExperimentManager
from source.evaluation import evaluate


dict(
        WordLengthRatio=dict(target_ratio=1),
        CharLengthRatio=dict(target_ratio=1),
        LevenshteinRatio=dict(target_ratio=1),
        DependencyTreeDepthRatio=dict(target_ratio=1),
        WordRankRatio=dict(target_ratio=1))


def objective(trial: optuna.trial.Trial) -> float:



    features = dict(
            WordLengthRatio=dict(target_ratio=trial.suggest_float('WordRatio', 0.20, 1.5, step=0.05)),
            CharLengthRatio=dict(target_ratio=trial.suggest_float('CharRatio', 0.20, 1.5, step=0.05)),
            LevenshteinRatio=dict(target_ratio=trial.suggest_float('LevenshteinRatio', 0.20, 1.5, step=0.05)),
            DependencyTreeDepthRatio=dict(target_ratio=trial.suggest_float('DepthTreeRatio', 0.20, 1.5, step=0.05)),
            WordRankRatio=dict(target_ratio=trial.suggest_float('WordRankRatio', 0.20, 1.5, step=0.05)))

    experiment_id = None

    experiment = ExperimentManager.load_experiment(experiment_id)
    result = evaluate(experiment, WIKILARGE_CHUNK_DATASET, features)
    return result


if __name__ == '__main__':

    study_path = "test"
    study = optuna.create_study(study_name='Tokens_study', direction="maximize")
    study.optimize(objective, n_trials=10)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
