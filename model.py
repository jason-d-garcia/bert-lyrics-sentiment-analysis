# model.py
from __future__ import annotations

import torch
from torch import nn
from transformers import BertModel


class SentimentClassifier(nn.Module):
    def __init__(self, n_classes: int, pretrained_name: str = "bert-base-cased", dropout_p: float = 0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_name)
        self.drop = nn.Dropout(p=dropout_p)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs["pooler_output"]  # [batch, hidden]
        dropped = self.drop(pooled)
        return self.out(dropped)           # [batch, n_classes]
