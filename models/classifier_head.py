import torch


class ClassifierHead(torch.nn.Sequential):
    def __init__(self, in_neurons, out_neurons) -> None:
        super().__init__(torch.nn.Linear(in_neurons, out_neurons))
