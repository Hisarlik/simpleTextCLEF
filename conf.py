import torch
from pathlib import Path

RESOURCES_DIR = Path(__file__).resolve().parent / "resources"
DATASETS_DIR = RESOURCES_DIR/ "datasets"
DUMPS_DIR = RESOURCES_DIR / "DUMPS"
PREPROCESSED_DIR = RESOURCES_DIR / "preprocessed"
OUTPUT_DIR = RESOURCES_DIR / "experiments"
WORD_EMBEDDINGS_NAME = "glove.42B.300d"



##### WIKILARGE_CHUNK #####
WIKILARGE_CHUNK_DATASET = DATASETS_DIR / "wikilarge_chunk"



##### T5 #####




##### DEVICE #####
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
