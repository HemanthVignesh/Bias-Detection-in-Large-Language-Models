from transformers import BertTokenizer
from src.config import MODEL_NAME, MAX_LENGTH

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def tokenize_text(text):
    return tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )