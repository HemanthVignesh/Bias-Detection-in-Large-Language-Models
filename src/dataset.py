import torch
from torch.utils.data import Dataset
from src.tokenizer_utils import tokenize_text

class BiasDataset(Dataset):
    def __init__(self, dataframe):
        self.texts = dataframe['text']
        self.labels = dataframe['label']

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = tokenize_text(text)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }