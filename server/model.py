import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms

transform = transforms.Compose([
  transforms.Resize((28, 28)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

dataset = datasets.ImageFolder(root="data/drowsiness-dataset/train", transform=transform) # assumes that each subfolder is a class

train_N = int(0.8 * len(dataset))
valid_N = len(dataset) - train_N

train_dataset, val_dataset = random_split(dataset, [train_N, valid_N])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=32)

print(dataset.class_to_idx)
batch = next(iter(train_loader))
batch
batch[0].shape # 32 28x28 graph papers, 3 colour channels (0-255)
batch[1].shape # 32 papers

n_classes = 4
kernel_size = 3
flattened_img_size = 75 * 3 * 3

class NeuralNetwork(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv_layer = nn.Sequential( # won't have much branching logic
      nn.Conv2d(3, 25, kernel_size, stride=1, padding=1), #3 img channels, 25 * 28 * 28
      nn.BatchNorm2d(25), # centre around 0
      nn.ReLU(),
      nn.MaxPool2d(2, stride=2),  # 25 x 14 x 14
      
      nn.Conv2d(25, 50, kernel_size, stride=1, padding=1), # 50 x 14 x 14
      nn.BatchNorm2d(50),
      nn.ReLU(),
      nn.Dropout(.2),
      nn.MaxPool2d(2, stride=2), # 50 x 7 x 7
      
      nn.Conv2d(50, 75, kernel_size, stride=1, padding=1), # 75 x 7 x 7
      nn.BatchNorm2d(75),
      nn.ReLU(),
      nn.Dropout(.2),
      nn.MaxPool2d(2, stride=2), # 75 x 3 x 3
    )
    
    self.fc_layer = nn.Sequential(
      nn.Flatten(),
      nn.Linear(flattened_img_size, 512),
      nn.Dropout(0.3),
      nn.ReLU(),
      nn.Linear(512, n_classes)
    )

  def forward(self, x):
    x = self.conv_layer(x)
    x = self.fc_layer(x)
    return x

model = NeuralNetwork()

# training loop

epochs = 8
loss_func = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

def get_batch_accuracy(output, y, N):
  pred = output.argmax(dim=1, keepdim=True) # getting highest score
  correct = pred.eq(y.view_as(pred)).sum().item() # using label
  return correct / N

def train():
  loss = 0
  accuracy = 0
  
  model.train()
  
  for x, y in train_loader:
    output = model(x)
    optimizer.zero_grad() # reset gradients each time
    batch_loss = loss_func(output, y)
    batch_loss.backward() # back propogate
    optimizer.step()
    
    loss += batch_loss.item()
    accuracy += get_batch_accuracy(output, y, train_N)
  
  print('Train minus Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy * 100))
  return loss, accuracy * 100
    
    
def validate():
  loss = 0
  accuracy = 0
  
  model.eval()
  
  with torch.no_grad():
    for x, y in valid_loader:
      output = model(x)
      
      loss += loss_func(output, y).item()
      accuracy += get_batch_accuracy(output, y, valid_N)
      
  print('Valid minus Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy * 100))
  return loss, accuracy * 100

      
def plot(train_accuracies, valid_accuracies, train_losses, valid_losses):
  plt.figure(figsize=(12, 5))

  # accuracy plot
  plt.subplot(1, 2, 1)
  plt.plot(train_accuracies, label='Train Accuracy')
  plt.plot(valid_accuracies, label='Validation Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy (%)')
  plt.title('Accuracy over Epochs')
  plt.legend()

  # lossp lot
  plt.subplot(1, 2, 2)
  plt.plot(train_losses, label='Train Loss')
  plt.plot(valid_losses, label='Validation Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Loss over Epochs')
  plt.legend()

  plt.tight_layout()
  plt.show()
  
# main training loop

if __name__ == "__main__":
  model = NeuralNetwork()

  train_losses, valid_losses = [], []
  train_accuracies, valid_accuracies = [], []

  for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    train_loss, train_acc = train()
    valid_loss, valid_acc = validate()

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_accuracies.append(train_acc)
    valid_accuracies.append(valid_acc)

  plot(train_accuracies, valid_accuracies, train_losses, valid_losses)