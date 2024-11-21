import torch


class PoolingLayer(torch.nn.Sequential):
    def __init__(self, in_neurons, out_neurons, dropout_prob=0.15):
        """
        Initialize the PoolingLayer.

        Args:
            in_neurons (int): Number of input neurons.
            out_neurons (int): Number of output neurons.
            dropout_prob (float, optional): Dropout probability. Defaults to 0.15.
        """
        super().__init__(
            torch.nn.Dropout(dropout_prob),
            torch.nn.Linear(in_neurons, out_neurons),
            torch.nn.Tanh(),
            torch.nn.Dropout(dropout_prob),
        )
