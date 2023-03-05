import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from tqdm import tqdm

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def qunatize_data(train_dataset, test_dataset):
    denom_train = np.full([len(train_dataset.data), 28, 28], 2**5)
    train_dataset.data = train_dataset.data / denom_train
    train_dataset.data = train_dataset.data.type(torch.uint8)

    denom_test = np.full([len(test_dataset.data), 28, 28], 2**5)
    test_dataset.data = test_dataset.data / denom_test
    test_dataset.data = test_dataset.data.type(torch.uint8)
    
    train_dataset.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Resize([8,8])])
    test_dataset.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Resize([8,8])])
    return train_dataset, test_dataset


class linear_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    # reshape/flatten input tensor when it is passed to model
    def forward(self, x):
        out = self.linear(x)
        return out

#Path settings
p = Path('.').parent.absolute()
dataset_path = p / 'dataset'

#Quantize to [0,7] range and 8x8 images?
quantize_test = False

#load dataset
train_dataset = torchvision.datasets.FashionMNIST(dataset_path, train=True, download=True)
test_dataset = torchvision.datasets.FashionMNIST(dataset_path, train=False)

#Transformation of the dataset
print('Min Pixel Value: {} \nMax Pixel Value: {}'.format(train_dataset.data.min(), train_dataset.data.max()))
print('Quantized test: {}'.format(quantize_test))
if quantize_test:
    train_dataset, test_dataset = qunatize_data(train_dataset, test_dataset)
    input_size = 8*8
    print('New Min Pixel Value: {} \nNew Max Pixel Value: {}'.format(train_dataset.data.min(), train_dataset.data.max()))
else:
    train_dataset.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    test_dataset.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    input_size = 28*28

#Show image example
for images, labels in train_dataset:
    plt.imshow(images[0],cmap='gray')
    plt.show()
    break

#Hyperparameters
batch_size = 100
epochs = 10
learning_rate = 0.001
output_size = 10

#Prepaer data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

#initialize model
model = linear_model()

loss_func = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print(list(model.parameters()))

for epoch in range(int(epochs)):
    for i, (images, labels) in tqdm(enumerate(train_loader)):
        images = Variable(images.view(-1, input_size))
        labels = Variable(labels)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
    # calculate Accuracy
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, input_size))
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total+= labels.size(0)
        correct+= (predicted == labels).sum()
    accuracy = 100 * correct/total
    print("Epoch: {}. Loss: {}. Accuracy: {}.".format(epoch, loss.item(), accuracy))

print(list(model.parameters()))