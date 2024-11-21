from torchmetrics import F1Score, Metric
import torch


class CustomF1Score(F1Score):
    def __init__(self, task_id, approach, task, average, num_classes):
        super(CustomF1Score, self).__init__(
            task=task, average=average, num_classes=num_classes, num_labels=num_classes
        )
        self.task_id = task_id
        self.approach = approach

    def update(self, preds: dict, target: dict) -> None:
        preds = preds[f"task{self.task_id}_{self.approach}"]
        target = target[f"task{self.task_id}_{self.approach}"]
        if preds.shape > 1:
            if preds.shape[1] > 1:
                preds = torch.argmax(preds, dim=1)

        if target.shape > 1:
            if target.shape[1] > 1:
                target = torch.argmax(target, dim=1)

        super(CustomF1Score, self).update(preds, target)

    def compute(self):
        return super(CustomF1Score, self).compute()


class CustomCrossEntropy(Metric):
    def __init__(self, task_id, **kwargs):
        super().__init__(**kwargs)
        self.task_id = task_id
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def update(self, preds, target):
        self.preds += preds
        self.target += target

    def compute(self):
        with torch.no_grad():
            preds = torch.stack(self.preds)
            target = torch.stack(self.target)
            return self.cross_entropy(preds, target)


class CrossEntropyLog2(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Update the state with the new logits and targets.
        Args:
            logits (torch.Tensor): Predictions from the model (logits).
            targets (torch.Tensor): Ground truth labels.
        """
        loss = self.cross_entropy_log2(logits, targets)
        self.loss += loss * logits.size(0)
        self.total += logits.size(0)

    def compute(self):
        """
        Compute the final metric value.
        Returns:
            torch.Tensor: The computed cross-entropy loss.
        """
        return self.loss / self.total

    def reset(self):
        """
        Reset the state variables to their default values.
        """
        self.loss = torch.tensor(0.0)
        self.total = torch.tensor(0)

    @staticmethod
    def cross_entropy_log2(logits: torch.Tensor, targets: torch.Tensor):
        """
        Compute the cross-entropy loss using base-2 logarithms.
        Args:
            logits (torch.Tensor): Predictions from the model (logits).
            targets (torch.Tensor): Ground truth labels.
        Returns:
            torch.Tensor: The cross-entropy loss.
        """
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1) / torch.log(
            torch.tensor(2.0)
        )
        return torch.nn.functional.nll_loss(log_probs, targets, reduction="mean")
