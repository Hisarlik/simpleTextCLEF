import os
import pickle
from pathlib import Path
from conf import OUTPUT_DIR


def save_object(path, experiment):
    with open(path, "wb") as f:
        pickle.dump(experiment, f)


def load_object(experiment_path):
    with open(experiment_path, "rb") as f:
        experiment = pickle.load(f)

    return experiment



def load_file(path):
    texts = []
    with open(path, "r", encoding="utf8") as f:
        lines = f.readlines()
        for line in lines:
            texts.append(line.replace("\n", ""))
    return texts

def save_file(path, texts):

    with open(path, 'w', encoding="utf8") as f:
        for line in texts:
            f.write(line)
            f.write('\n')

def yield_lines(filepath):
    #filepath = Path(filepath)
    with filepath.open('r', encoding="latin-1") as f:
        for line in f:
            yield line.rstrip()