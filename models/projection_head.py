import torch
import torch.nn.functional as F


class ProjectionHead(torch.nn.Sequential):
    def __init__(
        self,
        transformer_hidden_size: int,
        projection_size: int,
        dropout_prob: float = 0.15,
    ):
        super(ProjectionHead, self).__init__(
            torch.nn.Linear(transformer_hidden_size, projection_size),
            L2NormalizationLayer(),
        )


class L2NormalizationLayer(torch.nn.Module):
    def __init__(self, dim=1, eps=1e-12):
        super(L2NormalizationLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)
