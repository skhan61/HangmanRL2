from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import Accuracy, F1Score, MaxMetric, MeanMetric, MinMetric
from torchmetrics.classification import MultilabelAUROC, \
    MultilabelAveragePrecision, MultilabelF1Score, \
        MultilabelPrecision, MultilabelRecall, MultilabelPrecisionAtFixedRecall, \
        MultilabelSpecificity, MultilabelRankingAveragePrecision, MultilabelRankingLoss


# MultilabelCoverageError, \
#     BinaryHingeLoss, MultilabelRankingAveragePrecision, MultilabelRankingLoss,
    
# from src.model.loss import FocalLoss
# from sklearn.base import BaseEstimator
from typing import Type
from src.datamodule import encode_character
from src.model.components import GuessAccuracy

from src.utils import *

# # torch.set_float32_matmul_precision("medium")

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from sklearn.metrics import classification_report

class HangmanModel(LightningModule):

    def __init__(
        self,
        pos_weight: torch.Tensor,
        weight: torch.Tensor,
        thresholds: torch.Tensor,
        feature_extractor: nn.Module,
        feature_extractor_params: dict,
        classifier: nn.Module,
        classifier_params: dict,
        optimizer_config: dict,
        scheduler_config: dict,
        compile: bool = False,
    ):
        super().__init__()

        # self.save_hyperparameters(ignore=["feature_extractor_class", \
        #                     "classifier_class"])

        self.save_hyperparameters()
        
        # self.feature_extractor = feature_extractor
        self.pos_weight = pos_weight # for BCEWithLogitLoss
        self.weight = weight # class weight, for MultilabelSoftMarginLoss

        self.feature_extractor = feature_extractor(**feature_extractor_params)
        self.classifier = classifier(**classifier_params)

        self.feature_extractor = self.feature_extractor.to(self.device)
        self.classifier = self.classifier.to(self.device)

        self.optimal_threshold = thresholds if thresholds is not None else torch.full((26,), 0.5)
        
        self.initialize_metrics()
        self.initialize_loss()


    # def forward(self, states, lengths, features):
    def forward(self, batch):
        # Extracting components from the batch
        states = batch['states'].to(self.device)
        lengths = batch['lengths'].to(self.device)
        features = batch['features'].to(self.device)

        # Feature extraction and classification
        extracted_features = self.feature_extractor(states, lengths, features)
        logits = self.classifier(extracted_features)
        logits = logits.to(self.device)  # Ensure logits are on the correct device
        
        # Calculating probabilities
        # Calculating probabilities and ensuring they are on the correct device
        probabilities = torch.sigmoid(logits).to(self.device)
        # print(probabilities)

        # print(probabilities.device)
        self.optimal_threshold = self.optimal_threshold.to(self.device)
        # print(self.optimal_threshold)
        # Apply thresholds to compute binary predictions, ensure threshold tensor is also on the correct device
        predictions = (probabilities >= self.optimal_threshold).float()  # Convert to 0/1 based on threshold
        
        # predictions = (probabilities * self.optimal_threshold).float()  # Convert to 0/1 based on threshold
        
        # print(predictions)
        return logits, probabilities, predictions

    def initialize_metrics(self):
        # """Initialize metrics for training and validation."""
        # # AUROC score
        self.train_auroc = MultilabelAUROC(num_labels=26, average='macro')
        self.val_auroc = MultilabelAUROC(num_labels=26, average='macro')
        self.val_auroc_best = MaxMetric()

        # # Avg Precision
        self.train_ap = MultilabelAveragePrecision(num_labels=26, average='macro')
        self.val_ap = MultilabelAveragePrecision(num_labels=26, average='macro')
        self.val_ap_best = MaxMetric()

        ## MultilabelRankingLoss
        self.train_mrl = MultilabelRankingLoss(num_labels=26, average='marco')
        self.val_mrl = MultilabelRankingLoss(num_labels=26, average='macro')
        self.val_mrl_best = MinMetric()

    def initialize_loss(self):
        """Initialize components related to loss computation."""
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # Reset all metrics to clear previous values, including validation 
        # # set to train
        self.train()
        
        # # reset losses
        self.train_loss.reset()
        self.val_loss.reset()
        
        # # reset ap
        self.train_ap.reset()
        self.val_ap.reset()
        self.val_ap_best.reset()

        # Reset AUROC metrics
        self.train_auroc.reset()
        self.val_auroc.reset()
        self.val_auroc_best.reset()

        # # Reset MultilabelRankingLoss
        self.train_mrl.reset()
        self.val_mrl.reset()
        self.val_mrl_best.reset()


    def validation_step(self, batch, batch_idx):
        loss, probabilities, predictions, labels = self.model_step(batch)
        self.val_loss(loss)
        self.val_auroc(predictions, labels.int())
        self.val_ap(predictions, labels.int())
        self.val_f1(predictions.int(), labels.int())
        self.val_precision(predictions.int(), labels.int())
        self.val_recall(predictions.int(), labels.int())
        self.val_specificity(predictions.int(), labels.int())  # Add specificity calculation

        # self.val_precision_at_fixed_recall(probabilities, labels.int())

        self.log("val/val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/val_auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/val_ap", self.val_ap, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/val_precision", self.val_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/val_recall", self.val_recall, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val/val_precision_at_fixed_recall", self.val_precision_at_fixed_recall, \
        #                             on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/val_specificity", self.val_specificity, on_step=False, on_epoch=True, prog_bar=True)  # Log specificity

        return probabilities

    def training_step(self, batch, batch_idx):
        loss, probabilities, predictions, labels = self.model_step(batch)
        self.train_loss(loss)
        self.train_auroc(predictions, labels.int())
        self.train_ap(predictions, labels.int())
        self.train_f1(predictions.int(), labels.int())
        self.train_precision(predictions.int(), labels.int())
        self.train_recall(predictions.int(), labels.int())
        # self.train_precision_at_fixed_recall(probabilities, labels.int())
        self.train_specificity(predictions.int(), labels.int())  # Add specificity calculation

        self.log("train/train_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/train_auroc", self.train_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/train_ap", self.train_ap, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/train_f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/train_precision", self.train_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/train_recall", self.train_recall, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train/train_precision_at_fixed_recall", self.train_precision_at_fixed_recall, \
        #                             on_step=False, on_epoch=True, prog_bar=True)
        self.log("tran/train_specificity", self.train_specificity, on_step=False, on_epoch=True, prog_bar=True)  # Log specificity

        return loss

    def on_validation_epoch_end(self):
        auroc = self.val_auroc.compute()
        ap = self.val_ap.compute()
        f1 = self.val_f1.compute()
        precision = self.val_precision.compute()
        recall = self.val_recall.compute()
        # p_at_fixed_r = self.val_precision_at_fixed_recall.compute()
        specificity = self.val_specificity.compute()  # Compute specificity for the current epoch


        self.val_auroc_best(auroc)
        self.val_ap_best(ap)
        self.val_f1_best(f1)
        self.val_precision_best(precision)
        self.val_recall_best(recall)
        # self.val_precision_at_fixed_recall_best(p_at_fixed_r)
        self.val_specificity_best(specificity)  # Update the best specificity metric

        self.log("val/val_auroc_best", self.val_auroc_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/val_ap_best", self.val_ap_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/val_f1_best", self.val_f1_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/val_precision_best", self.val_precision_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/val_recall_best", self.val_recall_best.compute(), sync_dist=True, prog_bar=True)
        # self.log("val/val_precision_at_fixed_recall", self.val_precision_at_fixed_recall_best.compute(), \
        #          sync_dist=True, prog_bar=True)
        self.log("val/val_specificity_best", self.val_specificity_best.compute(), sync_dist=True, prog_bar=True)  # Log best specificity

        print(f"Best val auroc scores: {self.val_auroc_best.compute()}")      
        print(f"Best ap scores: {self.val_ap_best.compute()}")      
        print(f"Best f1 scores: {self.val_f1_best.compute()}")      
        print(f"Best precision scores: {self.val_precision_best.compute()}")   
        print(f"Best recall scores: {self.val_recall_best.compute()}")  
        # print(f"Best precision_at_fixed_recall scores: {self.val_precision_at_fixed_recall_best.compute()}")  
        print(f"Best specificity scores: {self.val_specificity_best.compute()}")  # Print the best specificity score

    def model_step(self, batch):
        # Get predictions from a PyTorch-based feature extractor
        logits, probabilities, predictions = self(batch)  # prediction is likelihood
        # logits = logits.to(self.device)  # Ensure logits are on the correct device
        # predictions = predictions.to(self.device)  # Ensure probabilities are on the correct device

        # Check if labels are available in the batch
        if 'labels' in batch:
            labels = batch['labels'].to(self.device).float()

            # # Ensure the position weight is on the correct device before creating the criterion
            # pos_weight_device = self.pos_weight.to(self.device)  # Move position weight to the same device

            # # # Reshape sample_weights to be compatible with the loss function
            # # sample_weights = batch['sample_weights'].to(self.device)
            # # sample_weights = sample_weights.unsqueeze(1)

            criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(self.device))
            # Ensure the position weight is on the correct device before creating the criterion
            
            # weight = self.weight.to(self.device)  # Move position weight to the same device

            # # # # # # Create criterion with position weight
            # criterion = nn.MultiLabelSoftMarginLoss(weight=self.weight.to(self.device))

            # Compute the loss
            loss = criterion(logits, labels)

            return loss, probabilities, predictions, labels

        return probabilities, predictions

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

        if 'scheduler_config' in self.hparams and self.hparams['scheduler_config'] is not None:
            # Retrieve scheduler class and parameters from hparams
            scheduler_config = self.hparams['scheduler_config']
            scheduler = scheduler_config['type'](optimizer, **scheduler_config['params'])

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
    
    # def initialize_metrics(self):
    #     """Initialize metrics for training and validation."""
    #     self.train_f1 = MultilabelF1Score(num_labels=26, average='macro', multidim_average='global')
    #     self.val_f1 = MultilabelF1Score(num_labels=26, average='macro', multidim_average='global')
    #     self.train_acc = MultilabelAccuracy(num_labels=26, average='macro', multidim_average='global')
    #     self.val_acc = MultilabelAccuracy(num_labels=26, average='macro', multidim_average='global')
    #     self.train_auroc = MultilabelAUROC(num_labels=26, average='macro')
    #     self.val_auroc = MultilabelAUROC(num_labels=26, average='macro')

    #     # Metrics for tracking the best validation scores
    #     self.val_f1_best = MaxMetric()
    #     self.val_acc_best = MaxMetric()
    #     self.val_auroc_best = MaxMetric()

#     def initialize_loss(self):
#         """Initialize components related to loss computation."""
#         self.train_loss = MeanMetric()
#         self.val_loss = MeanMetric()

# # if __name__ == "__main__":
# #     _ = MNISTLitModule(None, None, None, None)


#     # def validation_step(self, batch, batch_idx):
#     #     loss, preds, labels = self.model_step(batch)

#     #     # # Debug: Check the types of predictions and labels
#     #     # print("Type of preds:", type(preds))
#     #     # print("Type of labels:", type(labels))
#     #     # print("First few preds:", preds[:5])  # print first few predictions to check
#     #     # print("First few labels:", labels[:5])  # print first few labels to check

#     #     # Ensure predictions are converted to PyTorch tensors if they are not already
#     #     if not isinstance(preds, torch.Tensor):
#     #         preds = torch.tensor(preds, dtype=torch.float32, device=self.device)

#     #     # Ensure labels are also tensors (they should already be)
#     #     if not isinstance(labels, torch.Tensor):
#     #         labels = torch.tensor(labels, dtype=torch.int64, device=self.device)

#     #     # Metrics calculation
#     #     self.val_loss(loss)
#     #     self.val_f1(preds, labels.int())
#     #     self.val_acc(preds, labels.int())
#     #     self.val_auroc(preds.float(), labels.int())

#     #     self.log("val/val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
#     #     self.log("val/val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
#     #     self.log("val/val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
#     #     self.log("val/val_auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)

#     #     return {'val_loss': loss, 'val_f1': self.val_f1, 'val_acc': self.val_acc, 'val_auroc': self.val_auroc}

    # def on_train_epoch_end(self) -> None:
    #     auroc = self.train_auroc.compute()
    #     # self.val_auroc_best(auroc)
        
    #     self.log("train/val_auroc_best", self.val_auroc.compute(), sync_dist=True, prog_bar=True)
    #     print(f"Best val win rate {self.val_auroc_best.compute()}")      

    # return super().on_train_epoch_end()


#     def on_train_epoch_end(self) -> None:
#         "Lightning hook that is called when a training epoch ends."
#         pass

#     def test_step(
#         self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
#     ) -> None:
#         """Perform a single test step on a batch of data from the test set.

#         :param batch: A batch of data (a tuple) containing the input tensor of images and target
#             labels.
#         :param batch_idx: The index of the current batch.
#         """
#         pass

#     def on_test_epoch_end(self) -> None:
#         """Lightning hook that is called when a test epoch ends."""
#         pass

#     def setup(self, stage: str) -> None:
#         """Lightning hook that is called at the beginning of fit (train + validate), validate,
#         test, or predict.

#         This is a good hook when you need to build models dynamically or adjust something about
#         them. This hook is called on every process when using DDP.

#         :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
#         """
#         if self.hparams.compile and stage == "fit":
#             self.feature_extractor = torch.compile(self.feature_extractor)




    # def adjust_predictions(self, predictions, thresholds):
    #     """ Adjust predictions based on dynamic thresholds to prevent all-0 or all-1 predictions. """
    #     min_positive = 0.1  # At least 10% of predictions should be positive
    #     max_negative = 0.9  # At most 90% of predictions should be negative

    #     # Calculate current proportions
    #     positive_proportion = predictions.mean(dim=0)

    #     if torch.any(positive_proportion < min_positive):
    #         thresholds -= 0.05  # Decrease threshold to increase positives
    #     if torch.any(positive_proportion > max_negative):
    #         thresholds += 0.05  # Increase threshold to decrease positives

    #     thresholds = torch.clamp(thresholds, 0.0, 1.0)  # Keep thresholds within [0, 1]
    #     adjusted_predictions = (predictions >= thresholds).int()
    #     return adjusted_predictions

    # def forward(self, batch):
    #     # Extracting components from the batch
    #     states = batch['states'].to(self.device)
    #     lengths = batch['lengths'].to(self.device)
    #     features = batch['features'].to(self.device)

    #     # Feature extraction and classification
    #     extracted_features = self.feature_extractor(states, lengths, features)
    #     logits = self.classifier(extracted_features)
    #     logits = logits.to(self.device)  # Ensure logits are on the correct device
        
    #     # Calculating probabilities
    #     probabilities = torch.sigmoid(logits).to(self.device)

    #     # Apply thresholds to compute binary predictions, ensure threshold tensor is also on the correct device
    #     predictions = (probabilities >= self.optimal_threshold).float()

    #     # Adjust predictions if needed to prevent all-0 or all-1 scenarios
    #     adjusted_predictions = self.adjust_predictions(predictions, self.optimal_threshold)

    #     return logits, probabilities, adjusted_predictions

