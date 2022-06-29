# -- fix path --
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --

from conf import WIKILARGE_CHUNK_DATASET, TURKCORPUS_DATASET, WIKILARGE_DATASET, SIMPLETEXT_DATASET
from source.experiments import ExperimentManager
from source.evaluation import evaluate

if __name__ == "__main__":
    features = dict(
        WordLengthRatio=dict(target_ratio=0.75),
        CharLengthRatio=dict(target_ratio=0.6),
        LevenshteinRatio=dict(target_ratio=0.6),
        DependencyTreeDepthRatio=dict(target_ratio=0.95),
        LMFillMaskRatio=dict(target_ratio=0.75)
    )
    #Select experiment_id value or put None to evaluate last trained model.
    experiment_id = None
    dataset = SIMPLETEXT_DATASET

    experiment = ExperimentManager.load_experiment(experiment_id)
    evaluate(experiment, dataset, features)
