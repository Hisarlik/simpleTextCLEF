from typing import Optional, Dict
from pathlib import Path
import datasets
import pandas as pd
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from source import features
from utils import logging_module

logger = logging_module.get_logger(__name__)


class SimplificationDataModule(LightningDataModule):
    """
  Creating a custom dataset for reading the dataset and
  loading it into the dataloader to pass it to the neural network for finetuning the model

  """

    def __init__(
            self,
            model_name: str,
            data_path: Path,
            model_features: Dict,
            max_seq_length: int = 256,
            train_batch_size: int = 8,
            eval_batch_size: int = 8,
            **kwargs):
        super().__init__()
        self.dataset = None
        self.model_name = model_name
        self.data_path = data_path
        self.model_features = model_features
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

    def load_data(self):
        train_original_data = pd.read_csv(Path(self.data_path) / "wikilarge_chunk.train.complex",
                                          sep="\t", header=None, names=["original_text"])

        train_simple_data = pd.read_csv(Path(self.data_path) / "wikilarge_chunk.train.simple",
                                        sep="\t", header=None, names=["simple_text"])

        valid_original_data = pd.read_csv(Path(self.data_path) / "wikilarge_chunk.valid.complex",
                                          sep="\t", header=None, names=["original_text"])

        valid_simple_data = pd.read_csv(Path(self.data_path) / "wikilarge_chunk.valid.simple",
                                        sep="\t", header=None, names=["simple_text"])

        test_original_data = pd.read_csv(Path(self.data_path) / "wikilarge_chunk.test.complex",
                                         sep="\t", header=None, names=["original_text"])

        test_simple_data = pd.read_csv(Path(self.data_path) / "wikilarge_chunk.test.simple",
                                       sep="\t", header=None, names=["simple_text"])

        train_data = pd.concat([train_original_data, train_simple_data], axis=1)
        valid_data = pd.concat([valid_original_data, valid_simple_data], axis=1)
        test_data = pd.concat([test_original_data, test_simple_data], axis=1)

        train_test_valid_dataset = datasets.DatasetDict({
            'train': datasets.Dataset.from_pandas(train_data),
            'valid': datasets.Dataset.from_pandas(valid_data),
            'test': datasets.Dataset.from_pandas(test_data)
        })

        return train_test_valid_dataset

    def setup(self, stage: Optional[str] = None):
        self.dataset = self.load_data()
        self.add_features()
        self.tokenize_dataset()

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size)

    def val_dataloader(self):
        return DataLoader(self.dataset["valid"], batch_size=self.eval_batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)

    def add_features(self):
        for feature, kwargs in self.model_features.items():
            logger.info(f"Calculating feature: {feature}")
            self.dataset = self.dataset.map(getattr(features, feature)().get_ratio,
                                            fn_kwargs=kwargs)
            logger.info(f"Feature: {feature} calculated.")

        self.dataset = self.dataset.map(lambda example: {'original_text_preprocessed':
                                                             "simplify: " +
                                                             example['original_text_preprocessed'] +
                                                             example['original_text']})

    def tokenize_dataset(self):

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self._tokenize_batch,
                batched=True
            )

        columns = ["input_ids", "labels", "attention_mask", "target_mask"]
        self.dataset.set_format(type="torch", columns=columns, output_all_columns=True)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self._replace_pad_token_id,
            )
        return self.dataset

    def _tokenize_batch(self, batch):

        input_encodings = self.tokenizer(batch["original_text_preprocessed"], max_length=256,
                                         truncation=True, padding="max_length")

        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(batch["simple_text"], max_length=256,
                                              truncation=True, padding="max_length")

        return {"original_text": batch["original_text"],
                "input_ids": input_encodings["input_ids"],
                "attention_mask": input_encodings["attention_mask"],
                "simple_text": batch["simple_text"],
                "labels": target_encodings["input_ids"],
                "target_mask": target_encodings["attention_mask"]}

    def _replace_pad_token_id(self, batch):

        labels = batch["labels"]
        # Huggingface’s loss functions are defined to exclude the ID -100 during loss calculations.
        # Therefore, we need to convert all padding token IDs in labels to -100.
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch
