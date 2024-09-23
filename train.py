import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ntlk_utils import tokenize, stem, bag_of_words
from model import NeuralNet

# Load intents
with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

# Preparing data
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!', '.', ',', '/', ':']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = np.array([bag_of_words(pattern_sentence, all_words) for pattern_sentence, tag in xy])
Y_train = np.array([tags.index(tag) for _, tag in xy])

# Define the dataset class
class ChatDataset(Dataset):
    def __init__(self, X_train, Y_train):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

def main():
    dataset = ChatDataset(X_train, Y_train)
    batch_size = 8
    hidden_size = 8
    output_size = len(tags)
    input_size = len(X_train[0])
    learning_rate = 0.001
    num_epochs = 1000
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print("Running on CUDA")
    else:
        print("Running on CPU")

    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for words, labels in train_loader:
            words = words.to(device)
            labels = labels.to(device).long()

            outputs = model(words)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}/1000, Loss: {loss.item():.4f}')
    print(f'Final loss: {loss.item():.4f}')

    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "all_words": all_words,
        "tags": tags,
    }
    FILE = "data.pth"
    torch.save(data, FILE)
    print(f'Training complete. File saved to {FILE}')

if __name__ == '__main__':
    main()
