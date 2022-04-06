# -- fix path --
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --

from typing import Dict
from pathlib import Path

from source.optimization import Experiment
from conf import WIKILARGE_CHUNK_DATASET, TURKCORPUS_DATASET, WIKILARGE_DATASET, SIMPLETEXT_DATASET
from source.optimization import ExperimentManager




def evaluate(experiment: Experiment,
             dataset: Path,
             features: Dict):

    dm = experiment.create_and_setup_data_module(dataset, features, "test")
    trainer = experiment.create_trainer()
    model = experiment.load_best_model()
    trainer.test(model, datamodule=dm)
    experiment.get_metrics(model, dm)




if __name__ == "__main__":
    features = dict(
        WordLengthRatio=dict(target_ratio=0.7),
        CharLengthRatio=dict(target_ratio=0.6),
        LevenshteinRatio=dict(target_ratio=0.8),
        DependencyTreeDepthRatio=dict(target_ratio=0.7),
        WordRankRatio=dict(target_ratio=0.6)
    )

    experiment_id = None

    experiment = ExperimentManager.load_experiment(experiment_id)
    evaluate(experiment, WIKILARGE_CHUNK_DATASET, features)
