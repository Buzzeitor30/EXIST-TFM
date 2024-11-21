from torch import nn
from torch import tensor


class CustomBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def __init__(self, task_ids, **kwargs):
        super(CustomBCEWithLogitsLoss, self).__init__(**kwargs)
        self.task_id = task_ids[0]

    def forward(self, input, target, prefix):
        input = input[f"task{self.task_id}"]
        label = target[f"task{self.task_id}"]
        mask = target.get(
            f"task{self.task_id}_mask",
            tensor([1 for _ in range(label.shape[1])], device=input.device),
        )
        loss = super().forward(input, label) * mask
        return {f"{prefix}_loss": loss.mean()}


class CustomCELoss(nn.CrossEntropyLoss):
    def __init__(self, task_ids, **kwargs):
        super(CustomCELoss, self).__init__(**kwargs)
        self.task_id = task_ids[0]

    def forward(self, input, target, prefix):
        input = input[f"task{self.task_id}"]
        label = target[f"task{self.task_id}"]
        loss = super().forward(input, label)
        return {f"{prefix}_loss": loss}


class CustomMultiTaskLoss(nn.Module):
    def __init__(self, task_id, **kwargs):
        super(CustomMultiTaskLoss, self).__init__(**kwargs)
        self.task_id = task_id  # for compatibility reasons

        self.task4_loss = CustomCELoss("4")
        self.task5_loss = CustomCELoss("5")
        self.task6_loss = CustomBCEWithLogitsLoss("6", **{"reduction": "none"})

    def forward(self, input, target, prefix):

        loss_task_4 = self.task4_loss(input, target, prefix)[f"{prefix}_loss"]
        loss_task_5 = self.task5_loss(input, target, prefix)[f"{prefix}_loss"]
        loss_task_6 = self.task6_loss(input, target, prefix)[f"{prefix}_loss"]

        loss = loss_task_4 + loss_task_5 + loss_task_6
        return {
            f"{prefix}_loss": loss,
            f"{prefix}_task4_loss": loss_task_4,
            f"{prefix}_task5_loss": loss_task_5,
            f"{prefix}_task6_loss": loss_task_6,
        }
