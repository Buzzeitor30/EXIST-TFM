from typing import Any, Sequence
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from torch import sigmoid, softmax, argmax, flatten
import pandas as pd
import os


class CustomWriterForAnyTask(BasePredictionWriter):
    def __init__(self, output_dir, write_interval, filename):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.df = {"id": [], "value": []}
        self.soft_df = {"id": [], "value": []}
        self.hard_df = {"id": [], "value": []}
        self.filename = filename

    def on_predict_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self.df = pd.DataFrame(self.df)
        self.df["test_case"] = "EXIST2024"
        """self.df.to_json(
            os.path.join(self.output_dir, self.filename + ".json"), orient="records"
        )"""
        self.soft_df = pd.DataFrame(self.soft_df)
        self.soft_df["test_case"] = "EXIST2024"
        self.soft_df.to_json(
            os.path.join(self.output_dir, self.filename + "_soft.json"),
            orient="records",
        )
        self.hard_df = pd.DataFrame(self.hard_df)
        self.hard_df["test_case"] = "EXIST2024"
        self.hard_df.to_json(
            os.path.join(self.output_dir, self.filename + "_hard.json"),
            orient="records",
        )


class EXISTT4Writer(CustomWriterForAnyTask):
    def __init__(self, output_dir, write_interval, filename, approach):
        super().__init__(output_dir, write_interval, filename)
        self.approach = approach

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        preds = sigmoid(outputs["value"]).tolist()
        """if self.approach == "hard":
            preds = sigmoid(outputs["value"]).tolist()
            preds = [{"YES": pred, "NO": 1 - pred} for pred in preds]
        else:
            preds = softmax(outputs["value"], dim=1).tolist()
            preds = [{"YES": pred[0], "NO": pred[1]} for pred in preds]"""

        preds = [{"YES": pred, "NO": 1 - pred} for pred in preds]
        self.df["id"].extend(list(map(str, outputs["id"].tolist())))
        self.df["value"].extend(preds)
        # Id is the same for both soft and hard
        self.soft_df["id"].extend(list(map(str, outputs["id"].tolist())))
        self.hard_df["id"].extend(list(map(str, outputs["id"].tolist())))
        # Soft we store the probabilities, hard we store the max
        self.soft_df["value"].extend(preds)

        self.hard_df["value"].extend(["YES" if x["YES"] > 0.5 else "NO" for x in preds])


class EXISTT5Writer(CustomWriterForAnyTask):
    def __init__(self, output_dir, write_interval, filename, approach):
        super().__init__(output_dir, write_interval, filename)
        self.approach = approach

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        preds_softmax = softmax(outputs["value"], dim=1)
        preds = [
            {
                "NO": pred[0].item(),
                "DIRECT": pred[1].item(),
                "JUDGEMENTAL": pred[2].item(),
            }
            for pred in preds_softmax
        ]
        preds_hard = list(
            map(
                lambda x: "NO" if x == 0 else "DIRECT" if x == 1 else "JUDGEMENTAL",
                argmax(preds_softmax, dim=1).tolist(),
            )
        )

        self.df["id"].extend(list(map(str, outputs["id"].tolist())))
        self.df["value"].extend(preds)

        self.soft_df["id"].extend(list(map(str, outputs["id"].tolist())))
        self.soft_df["value"].extend(preds)

        self.hard_df["id"].extend(list(map(str, outputs["id"].tolist())))
        self.hard_df["value"].extend(preds_hard)


class EXISTT6Writer(CustomWriterForAnyTask):
    def __init__(self, output_dir, write_interval, filename, approach):
        super().__init__(output_dir, write_interval, filename)
        self.approach = approach

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:

        preds_sigmoid = sigmoid(outputs["value"]).tolist()
        preds = [
            (
                {
                    "NO": pred[0],
                    "IDEOLOGICAL-INEQUALITY": pred[1],
                    "MISOGYNY-NON-SEXUAL-VIOLENCE": pred[2],
                    "OBJECTIFICATION": pred[3],
                    "SEXUAL-VIOLENCE": pred[4],
                    "STEREOTYPING-DOMINANCE": pred[5],
                }
            )
            for pred in preds_sigmoid
        ]

        self.df["id"].extend(list(map(str, outputs["id"].tolist())))
        self.soft_df["id"].extend(list(map(str, outputs["id"].tolist())))
        self.hard_df["id"].extend(list(map(str, outputs["id"].tolist())))
        self.df["value"].extend(preds)
        self.soft_df["value"].extend(preds)
        # TODO: Hard is not implemented
        if self.approach == "soft":
            preds_hard = [[k for k, v in pred.items() if v >= 0.3] for pred in preds ]
        else:
            preds_hard = [[k for k, v in pred.items() if v >= 0.5] for pred in preds ]
        #Just in case...
        for idx, pred in enumerate(preds_hard):
            if pred == []:
                preds_hard[idx] = ["NO"]
        self.hard_df["value"].extend(preds_hard)
