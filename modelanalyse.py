import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import pickle 

df = pd.read_csv("dataset_amazon.csv")
df = df[['review', 'label']]
df.dropna(inplace=True)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    return text

df['review'] = df['review'].apply(clean_text)

def tokenize(text):
    return text.split()

all_words = []
for text in df['review']:
    all_words.extend(tokenize(text))

word_counts = Counter(all_words)
vocab = {word: i+1 for i, (word, _) in enumerate(word_counts.most_common(10000))}

max_len = 50

def encode(text):
    return [vocab.get(word, 0) for word in tokenize(text)]

def pad(seq):
    if len(seq) < max_len:
        seq += [0] * (max_len - len(seq))
    return seq[:max_len]

df['encoded'] = df['review'].apply(encode)
df['padded'] = df['encoded'].apply(pad)

X = np.array(df['padded'].tolist())
y = np.array(df['label'].tolist())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

class ReviewDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_data = ReviewDataset(X_train, y_train)
test_data = ReviewDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

class SentimentModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, 128)
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.fc = nn.Linear(128, 3)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out

model = SentimentModel(len(vocab))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    total_loss = 0
    for X_batch, y_batch in train_loader:
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

correct = 0
total = 0

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)

        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

print("Accuracy:", correct / total)

def predict(text):
    text = clean_text(text)
    seq = encode(text)
    seq = pad(seq)

    tensor = torch.tensor([seq], dtype=torch.long)
    output = model(tensor)

    pred = torch.argmax(output, dim=1).item()
    labels = {0: "Negative", 1: "Positive",2: "Neutral"}
    return labels[round(pred)]
#model save 
torch.save(model,'model_analyzer.pth')

