import torch
from transformers import AutoConfig, AutoModel


class TextEncoder(torch.nn.Module):
    def __init__(self, hf_text_model) -> None:
        super(TextEncoder, self).__init__()
        self.text_config = AutoConfig.from_pretrained(hf_text_model)
        self.text_encoder = AutoModel.from_pretrained(
            hf_text_model, add_pooling_layer=False
        )

    def forward(self, batch):
        x = self.text_encoder(**batch["text"]).last_hidden_state # Return token <CLS>
        return x
