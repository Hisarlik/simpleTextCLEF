# -- fix path -- Source: https://github.com/KimChengSHEANG/TS_T5
from pathlib import Path
import sys
from typing import Dict

sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --

from source.optimization import Experiment
from source.utils import storage
from conf import WIKILARGE_CHUNK_DATASET, DEVICE


def main(model_hyperparameters: Dict,
         features: Dict
         ):
    """
    :param model_hyperparameters: Language model parameters. e.g. : selected model, learning rate, ...
    :param features: A Dict with simplification features e.g. : WordLengthRatio, WordRankRatio, ...
    :return:
    """

    experiment = Experiment(
        model_hyperparameters,
        features
    )

    experiment.start()
    storage.save_experiment(experiment)


if __name__ == "__main__":
    features = dict(
        WordLengthRatio=dict(target_ratio=0.8),
        CharLengthRatio=dict(target_ratio=0.8),
        LevenshteinRatio=dict(target_ratio=0.8),
        DependencyTreeDepthRatio=dict(target_ratio=0.8),
        WordRankRatio=dict(target_ratio=0.8)
    )

    config = dict(
        model_name='t5-small',
        dataset_path=WIKILARGE_CHUNK_DATASET,
        number_epochs=5,
        max_seq_length=32,
        learning_rate=3e-4,
        weight_decay=0.1,
        adam_epsilon=1e-8,
        warmup_steps=5,
        train_batch_size=6,
        valid_batch_size=6,
        custom_loss=False,
        gradient_accumulation_steps=1,  # 16
        accelerator="gpu",
        fp_16=False,
        opt_level='O1',
        max_grad_norm=1.0,
        seed=42,
        nb_sanity_val_steps=0,
        train_sample_size=1,
        valid_sample_size=1,
        device=DEVICE
    )

    main(config, features)
