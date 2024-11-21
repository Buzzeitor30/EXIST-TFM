import torch
import transformers


class MultiModalModel(torch.nn.Module):
    def __init__(self, hf_model_name: str):
        super(MultiModalModel, self).__init__()
        self.multimodal_config = transformers.FlavaMultimodalConfig.from_pretrained(
            hf_model_name
        )
        self.multimodal_model = transformers.FlavaMultimodalModel.from_pretrained(
            hf_model_name, add_pooling_layer=False
        )

    def forward(self, x):
        x = self.multimodal_model(x).last_hidden_state
        return x
