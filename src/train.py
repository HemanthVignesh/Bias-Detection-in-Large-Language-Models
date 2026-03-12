import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_loader import load_sample_data
from src.dataset import BiasDataset
from src.model import BiasClassifier

# Load dataset
df = load_sample_data()

print("Dataset Loaded:")
print(df)

# Dataset and loader
dataset = BiasDataset(df)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Model
model = BiasClassifier()

# Loss function
criterion = nn.BCELoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# Training
epochs = 3

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}")

    for batch in loader:
        optimizer.zero_grad()

        outputs = model(
            batch['input_ids'],
            batch['attention_mask']
        ).view(-1)

        labels = batch['label']

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        print("Loss:", loss.item())