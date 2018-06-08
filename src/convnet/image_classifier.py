#############
# http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#############
import itertools
import os
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

# Transforms are common image transforms. They can be chained together using Compose
transform = transforms.Compose([
    transforms.Resize(size=(42, 42)),
    transforms.ToTensor()
])

# Load individual image data
trainset = torchvision.datasets.ImageFolder('./imagery/training_data/processed', transform=transform)
trainset_loader = torch.utils.data.DataLoader(trainset, shuffle=False)

# Testset
testset = torchvision.datasets.ImageFolder('./imagery/test_data/processed/eddies', transform=transform)
testset_loader = torch.utils.data.DataLoader(testset, shuffle=False)

classes = tuple(trainset.classes)

###########################################
# Convolutional Neural Network Definition
###########################################

# Network Architecture
# Convolution 1 > Relu > MaxPool2d
# Convolution 2 > Relu > MaxPool2d
# Linear 1 > Relu
# Linear 2 > Relu
# Linear 3

class Net(nn.Module):

    def __init__(self, num_classes=4):
        super(Net, self).__init__()
        # 1st Convolution Layer (3 image input channels, 6 output channels, 3x3 convolution kernel)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)

        # Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2)

        # 2nd Convolution Layer
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)

        # Apply linear transformations to the incoming data
        # torch.nn.Linear(in_features, out_features, bias=True)
        self.fc1 = nn.Linear(in_features=16 * 9 * 9, out_features=num_classes)

    def forward(self, input):
        ### input.size() >>> [4, 3, 42, 42]
        output = Func.relu(self.conv1(input))
        ### output.size() >>> [4, 6, 40, 40]
        output = self.pool(output)
        ### output.size() >>> [4, 6, 20, 20]

        output = Func.relu(self.conv2(output))
        ### output.size() >>> [4, 16, 18, 18]
        output = self.pool(output)
        ### output.size() >>> [4, 16, 9, 9]

        output = output.view(-1, 16 * 9 * 9)
        ### output.size() >>> [4, 1296]
        output = Func.relu(self.fc1(output))
        return output

net = Net()

###########################################
# Loss Function
###########################################

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

###########################################
# Training Stage
###########################################

def train():

    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0

        for i, (images, labels) in enumerate(trainset_loader, 0):

            # Wrap images and labels into Variables
            images, labels = Variable(images), Variable(labels)

            # Clear all accumulated gradients
            optimizer.zero_grad()

            # Predict classes using images from the test set
            outputs = net(images)

             # Compute the loss based on the predictions and actual labels
            loss = criterion(outputs, labels)
            
            # Backpropagate the loss
            loss.backward()

            # Adjust parameters according to the computed gradients
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]

            if i % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')


#####################################
# Test the network on the test data
#####################################

def predict():

    dataiter = itertools.islice(iter(testset_loader), 100)

    for item in dataiter:
        images, labels = item

        print(f"GroundTruth: {classes[labels[0]]}")

        ##########################################################
        # Feed the images into the CNN and check what it predicts
        ##########################################################

        outputs = net(Variable(images))

        _, predicted = torch.max(outputs.data, 1)

        print(f"Predicted: {classes[predicted[0]]}")


################################
# Load existing net or train
################################

if os.path.exists('./data/models/gear-cnn'):
    print('Use already existing network')
    net.load_state_dict(torch.load('./data/models/gear-cnn'))
    predict()
else:
    print('Start training network')
    train()
    torch.save(net.state_dict(), './data/models/gear-cnn')


################################
# Print some test data
################################

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
