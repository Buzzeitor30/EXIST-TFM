import torch
from transformers import AutoConfig, AutoModel
import timm


class VisionEncoder(torch.nn.Module):
    def __init__(self, hf_model_name: str):
        super(VisionEncoder, self).__init__()
        self.vision_transformer = AutoModel.from_pretrained(hf_model_name, add_pooling_layer=False)

    def forward(self, batch):
        x = self.vision_transformer(batch["pixel_values"])
        return x

    def set_trainable(self, trainable=True):
        for p in self.vision_transformer.parameters():
            p.requires_grad = trainable
