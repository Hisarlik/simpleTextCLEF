import os
from source.utils import logging_module
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
import torch
from pathlib import Path
import pandas as pd
from easse.sari import corpus_sari
from transformers import (
    AdamW,
    AutoTokenizer,
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup
)
from source.utils import storage

logger = logging_module.get_logger(__name__)


class LitProgressBar(TQDMProgressBar):

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description('running validation ...')
        return bar


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")

        metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                logger.info("{} = {}\n".format(key, str(metrics[key])))


class T5SimplificationModel(pl.LightningModule):

    def __init__(self, **kwarg):
        """ Simplification Pytorch lightning module """
        super(T5SimplificationModel, self).__init__()

        self.save_hyperparameters()
        self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name).to(self.hparams.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name, use_fast=True)

        self.total_steps = None
        self.predictions = []

    def setup(self, stage=None) -> None:
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        if stage == "fit":
            train_loader = self.trainer.datamodule.train_dataloader()

            # Calculate total steps
            tb_size = self.hparams.train_batch_size * max(1, self.trainer.gpus)
            ab_size = tb_size * self.trainer.accumulate_grad_batches
            self.total_steps = int((len(train_loader.dataset) / ab_size) * float(self.trainer.max_epochs))
            logger.debug(f"Total steps: {self.total_steps}")

    def forward(
            self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None
    ):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            decoder_attention_mask=batch['target_mask'],
        )
        loss = outputs.loss
        # logger.info(f"train_loss: {loss}")
        self.log('train_loss', loss, on_epoch=False, prog_bar=True, logger=True,
                 batch_size=batch['input_ids'].size(dim=0))
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.sari_validation_step(batch)
        logger.info(f"valid_loss: {loss}")
        self.log('val_loss', loss, batch_size=batch['input_ids'].size(dim=0))
        return torch.tensor(loss, dtype=float)

    def test_step(self, batch, batch_idx):
        beam_outputs = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            do_sample=False,
            max_length=self.hparams.max_seq_length,
            num_beams=8,
            top_k=120,
            top_p=0.98,
            early_stopping=True,
            num_return_sequences=1
        ).to(self.device)

        predictions = self.tokenizer.batch_decode(beam_outputs,
                                                  skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=True)
        self.predictions.extend(predictions)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.learning_rate,
                          eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

    def optimizer_step(self, epoch=None, batch_idx=None, optimizer=None, optimizer_idx=None, optimizer_closure=None,
                       on_tpu=None, using_native_amp=None, using_lbfgs=None):
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def sari_validation_step(self, batch):
        beam_outputs = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            do_sample=False,
            max_length=self.hparams.max_seq_length,
            num_beams=8,
            top_k=120,
            top_p=0.98,
            early_stopping=True,
            num_return_sequences=1
        ).to(self.device)

        predictions = self.tokenizer.batch_decode(beam_outputs,
                                                  skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=True)

        test = [v for k, v in batch.items() if "valid" in k]
        score = corpus_sari(batch["original_text"], predictions, test)

        return 1 - score / 100
