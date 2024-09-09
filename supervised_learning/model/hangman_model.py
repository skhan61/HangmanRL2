from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics.aggregation import MaxMetric, MeanMetric
from torchmetrics.classification import MultilabelAveragePrecision
# from src.model.components import NDCG

from torchvision.ops import sigmoid_focal_loss

# MultilabelCoverageError, \
#     BinaryHingeLoss, MultilabelRankingAveragePrecision, MultilabelRankingLoss,
    
# from src.model.loss import FocalLoss
# from sklearn.base import BaseEstimator
from typing import Type
# from src.datamodule import encode_character
# from src.model.components import GuessAccuracy
# from src.model.components import MAP
from src.utils import *
import numpy as np
# from src.model.components import ListNetLoss
# # torch.set_float32_matmul_precision("medium")

class HangmanModel(LightningModule):

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(config)
        # self.hparams = config
        self.nn_model = self.hparams['nn_model'](config)

        self.initialize_loss()

    # def forward(self, states, lengths, features):
    def forward(self, batch):
        # Extracting components from the batch
        inputs = batch['inputs'].to(self.device)
        lengths = batch['lengths'].to(self.device)
        miss_chars = batch['miss_chars'].to(self.device)
        # Feature extraction and classification
        logits = self.nn_model(inputs, lengths, miss_chars)
        return logits

    def next_guess(self, single_batch, mask):
        self.eval()
        with torch.no_grad():
            # Extracting components from the batch
            inputs = single_batch['inputs'].to(self.device)
            lengths = single_batch['lengths'].to(self.device)
            miss_chars = single_batch['miss_chars'].to(self.device)

            # Feature extraction and classification
            logits = self.nn_model(inputs, lengths, miss_chars)

            # Ensure mask is set to a large negative value for guessed indices
            large_negative = torch.finfo(logits.dtype).min
            mask = (mask == 1) * large_negative  # Apply large negative value where mask is 1

            mask = mask.to(self.device)  # Ensure mask is also on the right device

            # Apply the mask by subtracting it from the logits
            masked_logits = logits + mask
            # print(f"masked_logits: ", masked_logits)

            # Compute probabilities with softmax
            probs = F.softmax(masked_logits, dim=1)
            # print(f"probs: ", probs)

            # Find the indices of the highest probability characters not masked out
            _, idxs = torch.max(probs, dim=1)
            
            # Convert index to character assuming the batch size is 1
            char = chr(idxs.item() + 97)
            
            return char

    def model_step(self, batch):
        # Get predictions from a PyTorch-based feature extractor
        logits = self(batch)  # prediction is likelihood
        if 'labels' in batch:
            labels = batch['labels'].to(self.device).long() 
            sample_weights = ((1/batch['lengths']) \
                              / torch.sum(1 / batch['lengths'])).to(self.device)
            sample_weights = sample_weights.unsqueeze(1)

            # Convert pos_weight to a tensor if it's not already and move to the correct device
            if isinstance(self.hparams['pos_weight'], (list, np.ndarray)):
                pos_weight = torch.tensor(self.hparams['pos_weight'], \
                                          dtype=torch.float).to(self.device)
            else:
                pos_weight = self.hparams['pos_weight'].to(self.device)

            # criterion = nn.BCEWithLogitsLoss(weight=sample_weights, \
            #                             pos_weight=pos_weight)
            
            criterion = nn.BCEWithLogitsLoss(weight=sample_weights, reduction='sum')
            # print(logits.device)
            # print(labels.device)

            loss = criterion(logits, (labels > 0).float() \
                if labels.max() > 1 else labels.float())
            # miss penalty calculation
            # Calculate miss penalty
            outputs = F.log_softmax(logits, dim=1)
            miss_penalty = torch.sum(outputs * batch['miss_chars'], dim=(0, 1)) \
                / outputs.shape[0]
            return loss, miss_penalty, logits, labels
        return logits

    def initialize_loss(self):
        """Initialize components related to loss computation."""
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.train_miss_penalty = MeanMetric()
        self.val_miss_penalty = MeanMetric()

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # Reset all metrics to clear previous values, including validation 
        # # set to train
        self.train() # TODO: check if this is necessary
        # # reset losses
        self.train_loss.reset()
        self.val_loss.reset()

        # reset miss penalty
        self.train_miss_penalty.reset()
        self.val_miss_penalty.reset()

    # def on_train_epoch_start(self) -> None:
    #     epoch = self.trainer.current_epoch
    #     self.trainer.datamodule.train_dataset.cur_epoch = epoch
    #     # self.trainer.train_dataloader.dataset.cur_epoch = epoch
    #     # if epoch % self.trainer.reload_dataloaders_every_n_epochs == 0:
    #     #     print(f"Epoch {epoch}: Reloading data loaders...")

    def training_step(self, batch, batch_idx):
        loss, miss_penalty, logits, labels = self.model_step(batch)
        self.train_loss(loss)
        self.train_miss_penalty(miss_penalty)

        self.log("train/train_loss", \
                 self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        self.log("train/train_miss_penalty", \
                 self.train_miss_penalty, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    # def on_validation_epoch_start(self):
    #     epoch = self.trainer.current_epoch
    #     # self.val_dataloader.dataset.cur_epoch = epoch
    #     self.trainer.datamodule.val_dataset.cur_epoch = epoch
    #     # if epoch % self.trainer.reload_dataloaders_every_n_epochs == 0:
    #     #     print(f"Epoch {epoch}: Reloading data loaders...")

    def validation_step(self, batch, batch_idx):
        loss, miss_penalty, logits, labels = self.model_step(batch)
        
        self.val_loss(loss)
        self.val_miss_penalty(miss_penalty)

        self.log("val/val_loss", \
                 self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        self.log("val/val_miss_penalty", \
                 self.val_miss_penalty, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        # optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())

        optimizer = self.hparams['optimizer_config']['type'](params=self.trainer.model.parameters(), \
                                                             **self.hparams['optimizer_config']['params'])
        
        # print(optimizer)

        if 'scheduler_config' in self.hparams and self.hparams['scheduler_config'] is not None:
            # print()
            # print('here')
            # print()
            # # Retrieve scheduler class and parameters from hparams
            scheduler_config = self.hparams['scheduler_config']
            scheduler = scheduler_config['type'](optimizer, **scheduler_config['params'])

            # print(scheduler)

            # Return optimizer and scheduler configuration
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": scheduler_config.get('monitor', 'val/val_loss'),
                    "interval": scheduler_config.get('interval', 'epoch'),
                    "frequency": scheduler_config.get('frequency', 1),
                },
            }
        
        return {"optimizer": optimizer}

