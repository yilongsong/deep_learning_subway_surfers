

# convnet.py

# yilong song
# Apr 6, 2023


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms

import os

import random

# ConvNet model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Convolutional layer 1 with 32 filters, a kernel size of 3x3
        self.conv1 = nn.Conv2d(4, 32, 3)
        # Max pooling layer 1 with pool size of 2x2
        self.pool = nn.MaxPool2d(2, 2)
        # Convolutional layer 2 with 64 filters, a kernel size of 3x3
        self.conv2 = nn.Conv2d(32, 64, 3)
        # Convolutional layer 3 with 128 filters, a kernel size of 3x3
        self.conv3 = nn.Conv2d(64, 128, 3)
        # Fully connected (dense) layer 1 with 256 units
        self.fc1 = nn.Linear(128 * 5 * 9, 256)
        # Dropout layer with a rate of 0.5 to reduce overfitting
        self.dropout = nn.Dropout(0.5)
        # Fully connected (dense) layer 2 with 128 units
        self.fc2 = nn.Linear(256, 128)
        # Output layer
        self.fc3 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 5 * 9)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Training

model = ConvNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise use CPU
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # You can adjust the learning rate (lr) as needed

# Load data
X = []
y = []

# Up: [1, 0, 0, 0]
# Down: [0, 1, 0, 0]
# Left: [0, 0, 1, 0]
# Right: [0, 0, 0, 1]

for folder in os.listdir('dataset'):
    if folder == '.DS_Store':
        continue
    for file in os.listdir('dataset/'+folder):
        if file == '.DS_Store':
            continue
        x = Image.open('dataset/'+folder+'/'+file)
        transform = transforms.ToTensor()
        x = transform(x)
        X.append(x)
        if folder == 'up':
            y.append([1,0,0,0])
        if folder == 'down':
            y.append([0,1,0,0])
        if folder == 'left':
            y.append([0,0,1,0])
        if folder == 'right':
            y.append([0,0,0,1])


# Shuffle X and y
indices = list(range(len(X)))
random.shuffle(indices)

X_shuffled = [X[i] for i in indices]
y_shuffled = [y[i] for i in indices]

X_train = X_shuffled[:int(len(X)*0.7)]
y_train = y_shuffled[:int(len(X)*0.7)]
X_test = X_shuffled[int(len(X)*0.7):]
y_test = y_shuffled[int(len(X)*0.7):]

X_train = torch.stack(X_train).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)

# Hyperparameters
num_epochs = 100
batch_size = 100
num_batches = len(X_train) // batch_size

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for i in range(num_batches):
        # Get a batch of training data
        batch_X = X_train[i * batch_size : (i + 1) * batch_size]
        batch_y = y_train[i * batch_size : (i + 1) * batch_size]

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item()

    # Print epoch loss
    epoch_loss = running_loss / num_batches
    print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, epoch_loss))

# Evaluation
model.eval()  # Set the model to evaluation mode
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

with torch.no_grad():
    # Forward pass for testing data
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    total = y_test.size(0)
    correct = (predicted == y_test).sum().item()
    accuracy = 100 * correct / total
    print("Test Accuracy: {:.2f}%".format(accuracy))

torch.save(model.state_dict(), 'convnet_trained.pth')