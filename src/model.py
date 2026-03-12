import torch
import torch.nn as nn
from transformers import BertModel
from src.config import MODEL_NAME


class BiasClassifier(nn.Module):
    def __init__(self):
        super(BiasClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        x = self.fc(x)

        return torch.sigmoid(x)