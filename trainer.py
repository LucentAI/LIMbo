from torch.utils.data import DataLoader, TensorDataset
from torchtext.data.utils import get_tokenizer
from datasets import load_dataset
import torch.optim as optim
import torch.nn as nn
import torch
import os

if torch.cuda.is_available():
    print("CUDA available.")
else:
    print("CUDA not available.")

# 1. Load and Tokenize Dataset
print("Loading dataset...")
dataset = load_dataset("stingning/ultrachat", cache_dir='./cache/', split="train")
print("Tokenizing dataset...")
tokenizer = get_tokenizer("basic_english")
print("Tokenizer set to basic_english.")
tokenized_dialogues = [tokenizer(' '.join(item['data'])) for item in dataset]
print("Tokenized dataset.")
vocab = {}
start_time = time.time()

for i, dialogue in enumerate(tokenized_dialogues):
    if time.time() - start_time > 1:
        print(f"Loading vocabulary [Dialogue {i+1}/{len(tokenized_dialogues)}]")
        start_time = time.time()
    for j, token in enumerate(dialogue):
        if token not in vocab:
            vocab[token] = len(vocab)
tokenized_dialogues = [[vocab[token] for token in dialogue] for dialogue in tokenized_dialogues]

# 2. Prepare Data for Training
tokenized_tensors = [torch.tensor(dialogue) for dialogue in tokenized_dialogues]
tokenized_tensors = torch.nn.utils.rnn.pad_sequence(tokenized_tensors, batch_first=True)
train_dataset = TensorDataset(tokenized_tensors)
train_dataloader = DataLoader(train_dataset, batch_size=4)

# 3. Define Model
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

# 4. Initialize and Train Model
vocab_size = len(vocab)
embed_dim = 128
hidden_dim = 256
model = SimpleRNN(vocab_size, embed_dim, hidden_dim).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):
    for i, data in enumerate(train_dataloader, 0):
        inputs = data[0].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, vocab_size), inputs.view(-1))
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item()}")