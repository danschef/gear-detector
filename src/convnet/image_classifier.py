#############
# http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#############
import itertools
import os

from net import Net
from image_helper import save_image

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

# Transforms are common image transforms. They can be chained together using Compose
TRANSFORM = transforms.Compose([
    transforms.Resize(size=(42, 42)),
    transforms.ToTensor()
])

TRAINING_DATA_PATH = './imagery/training_data/processed'
TEST_DATA_PATH = './imagery/test_data/processed/eddies'
TRAINSET = torchvision.datasets.ImageFolder(TRAINING_DATA_PATH, transform=TRANSFORM)
TESTSET = torchvision.datasets.ImageFolder(TEST_DATA_PATH, transform=TRANSFORM)
CLASSES = tuple(TRAINSET.classes)

# Training set
def trainset_loader():
    return torch.utils.data.DataLoader(TRAINSET, shuffle=True)

# Testing set
def testset_loader():
    return torch.utils.data.DataLoader(TESTSET, shuffle=False)

###########################################
# Training Stage
###########################################

def train(net):

    ###########################################
    # Loss Function
    ###########################################

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0

        for i, (images, labels) in enumerate(trainset_loader(), 0):

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
# Prediction stage
#####################################

def predict(net):

    dataiter = itertools.islice(iter(testset_loader()), 5000)
    # dataiter = iter(testset_loader())
    prediction_cnt = {'aquafarm': 0, 'cloud': 0, 'vessel': 0, 'water': 0}


    for idx, item in enumerate(dataiter):
        if idx % 100 == 0:
            print('.', end='', flush=True)

        images, _labels = item

        # print(f"GroundTruth: {CLASSES[labels[0]]}")

        ##########################################################
        # Feed the images into the CNN and check what it predicts
        ##########################################################

        outputs = net(Variable(images))

        _, predicted = torch.max(outputs.data, 1)

        if CLASSES[predicted[0]] == 'aquafarm':
            save_image(idx, images)

        prediction_cnt[CLASSES[predicted[0]]] += 1

    print(f"\nPredicted: {prediction_cnt}")

################################################################
# Train network or use existing one for prediction
################################################################

def main():
    net = Net()

    if os.path.exists('./data/models/gear-cnn'):
        print('Use trained network for prediction')
        net.load_state_dict(torch.load('./data/models/gear-cnn'))
        predict(net)
    else:
        print(f"Train network using data from {TRAINING_DATA_PATH}")
        train(net)
        torch.save(net.state_dict(), './data/models/gear-cnn')

main()
