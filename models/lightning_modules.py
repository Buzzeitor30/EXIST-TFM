import torch
from transformers import (
    get_linear_schedule_with_warmup,
)
import os
import pandas as pd
import lightning as L
from models.pooling_layer import PoolingLayer
from models.text_model import TextEncoder
from models.vision_model import VisionEncoder
from models.multimodal_model import MultiModalModel
from models.projection_head import ProjectionHead
from models.classifier_head import ClassifierHead
from transformers import AutoModel
from lightning.pytorch.utilities import grad_norm


class BaseModuleLightning(L.LightningModule):
    def __init__(self, lr, loss_fn, metrics_collection, dropout_rate):
        super(BaseModuleLightning, self).__init__()
        self.save_hyperparameters(
            ignore=[
                "metrics_collection",
                "loss_fn",
            ]
        )
        # Loss function
        self._loss_fn = loss_fn
        # Metrics
        self.learning_rate = lr
        self.train_metrics_collection = metrics_collection.clone(postfix="_train")
        self.val_metrics_collection = metrics_collection.clone(postfix="_val")
        self.dropout_rate = dropout_rate
        self.preds = {}

    def _common_step(self, batch, postifx="train"):
        """
        Performs a common step in the training/validation process.

        Args:
            batch (dict): A dictionary containing the input batch data.
            postfix (str): A string indicating the current step (either "train" or "val").

        Returns:
            tuple: A tuple containing the loss, output, and logits.

        """

        logits = self(batch)
        if logits.ndim == 2 and batch["label"].ndim == 1:
            logits = logits.squeeze(1)
        loss = self._loss_fn(logits, batch["label"])
        if postifx == "train":
            output = self.train_metrics_collection(logits, batch["label"])
        else:
            output = self.val_metrics_collection(logits, batch["label"])
        return loss, output, logits

    def training_step(self, batch):
        loss, output, _ = self._common_step(batch)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        for metric, value in output.items():
            self.log(
                metric,
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        return loss

    def validation_step(self, batch):
        loss, output, logits = self._common_step(batch, "validation")
        self.log_dict(
            {"val_loss": loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        for metric, value in output.items():
            self.log(
                metric,
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        return loss

    def predict_step(self, batch):
        logits = self(batch).squeeze(1)
        ids = batch["id"]
        return {"id": ids, "value": logits}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": get_linear_schedule_with_warmup(
                    optimizer,
                    int(0.1 * self.trainer.estimated_stepping_batches),
                    self.trainer.estimated_stepping_batches,
                ),
                "interval": "step",
                "frequency": 1,
                "name": "Linear",
            },
        }

    def build_classifier(self, hidden_size, output_neurons):
        return ClassifierHead(hidden_size, output_neurons)

    def freeze_text(self):
        try:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
        except AttributeError:
            pass

    def freeze_vision(self):
        try:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
        except AttributeError:
            pass


class TextEncoderL(BaseModuleLightning):
    def __init__(
        self,
        hf_text_model,
        learning_rate,
        loss_fn,
        output_neurons_for_each_classifier,
        metrics_collection,
        dropout_rate,
    ):
        super(TextEncoderL, self).__init__(
            learning_rate, loss_fn, metrics_collection, dropout_rate
        )
        # Model definition
        self.text_encoder = TextEncoder(hf_text_model)
        self.hidden_size = self.text_encoder.text_config.hidden_size
        self.pooling_layer = PoolingLayer(
            self.hidden_size, self.hidden_size, self.dropout_rate
        )
        self.classifier = self.build_classifier(
            self.hidden_size, output_neurons_for_each_classifier
        )

    def forward(self, batch):
        x = self.text_encoder(batch)[:, 0, :]
        x = self.pooling_layer(x)
        return self.classifier(x)
    

class VisionEncoderL(BaseModuleLightning):
    def __init__(
        self,
        hf_vision_model,
        learning_rate,
        loss_fn,
        output_neurons_for_each_classifier,
        metrics_collection,
        dropout_rate,
    ):
        super(VisionEncoderL, self).__init__(
            learning_rate, loss_fn, metrics_collection, dropout_rate
        )
        # Model definition
        self.vision_encoder = VisionEncoder(hf_vision_model)
        self.hidden_size = self.vision_encoder.vision_transformer.config.hidden_size
        self.classifier = self.build_classifier(
            self.hidden_size, output_neurons_for_each_classifier
        )

    def forward(self, batch):
        x = self.vision_encoder(batch["image"]).last_hidden_state[:, 0, :]  # Extract token <CLS>
        return self.classifier(x)
    

class EarlyFusionModelL(BaseModuleLightning):
    def __init__(
        self,
        hf_text_model,
        hf_vision_model,
        projection_dim,
        learning_rate,
        loss_fn,
        output_neurons_for_each_classifier,
        metrics_collection,
        dropout_rate,
    ):
        super(EarlyFusionModelL, self).__init__(
            learning_rate, loss_fn, metrics_collection, dropout_rate
        )
        # Model definition
        # Encoders
        self.text_encoder = TextEncoder(hf_text_model)
        self.vision_encoder = VisionEncoder(hf_vision_model)
        # Define projection head
        text_hidden_size = self.text_encoder.text_config.hidden_size
        vision_hidden_size = self.vision_encoder.vision_transformer.config.hidden_size
        self.text_projection_layer = torch.nn.Sequential(
            torch.nn.Linear(text_hidden_size, projection_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(self.dropout_rate),
        )
        self.vision_projection_layer = torch.nn.Sequential(
            torch.nn.Linear(vision_hidden_size, projection_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(self.dropout_rate),
        )
        # Define classifier
        self.hidden_size = projection_dim * 2
        self.classifier = self.build_classifier(
            self.hidden_size, output_neurons_for_each_classifier
        )

    def forward(self, batch):
        text_hidden_state = self.text_encoder(batch)[:, 0, :]  # It's already token CLS
        vision_hidden_state = self.vision_encoder(batch["image"]).last_hidden_state[:, 0, :]  # Extract token CLS

        text_projected = self.text_projection_layer(text_hidden_state)
        vision_projected = self.vision_projection_layer(vision_hidden_state)

        x = torch.cat((text_projected, vision_projected), dim=1)
        return self.classifier(x)


class MultiModalCrossAttentionL(BaseModuleLightning):
    def __init__(
        self,
        hf_text_model,
        hf_vision_model,
        hf_multimodal_model,
        learning_rate,
        loss_fn,
        output_neurons_for_each_classifier,
        metrics_collection,
        dropout_rate,
    ):
        super(MultiModalCrossAttentionL, self).__init__(
            learning_rate, loss_fn, metrics_collection, dropout_rate
        )
        # Transformers
        self.text_encoder = TextEncoder(hf_text_model)
        self.vision_encoder = VisionEncoder(hf_vision_model)
        self.multimodal = MultiModalModel(hf_multimodal_model)

        self.hidden_size = self.multimodal.multimodal_config.hidden_size
        # Classifier head
        self.pooling_layer = PoolingLayer(
            self.hidden_size,
            self.hidden_size,
            self.dropout_rate,
        )

        self.classifier = self.build_classifier(
            self.hidden_size, output_neurons_for_each_classifier
        )

    def forward(self, batch):
        text_hidden = self.text_encoder(batch)
        vision_hidden = self.vision_encoder(batch)
        # Flava MultiModal assumes input as (B, Image Num patches + text_seq_len, Hidden size)
        input_to_multimodal = torch.cat(
            (vision_hidden, text_hidden.unsqueeze(1)), dim=1
        )
        x = self.multimodal(input_to_multimodal)[:, 0, :]  # token CLS
        return self.classifier(x)
