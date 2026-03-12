from src.data_loader import load_sample_data
from src.tokenizer_utils import tokenize_text

df = load_sample_data()

print("Dataset:")
print(df)

sample = df['text'][0]

encoded = tokenize_text(sample)

print("\nOriginal Text:")
print(sample)

print("\nTokenized Output:")
print(encoded)
print(encoded['input_ids'].shape)
print(encoded['attention_mask'].shape)