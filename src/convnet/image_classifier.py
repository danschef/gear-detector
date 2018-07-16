import configparser
import os
import sys
from time import gmtime, strftime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from net import Net

from image_helper import CLASSES
from image_helper import save_image
from image_helper import test_set_loader
from image_helper import train_set_loader
from image_helper import validation_set_loader

CONFIG = configparser.ConfigParser()
CONFIG.read('./src/config.ini')

###########################################
# Training Stage
###########################################

def train(net, cycles=50, learning_rate=0.001):

    datetime = strftime("%Y%m%d_%H%M", gmtime())
    logfile = f"{CONFIG['CNN Paths']['accuracy_log_path']}/{datetime}.log"

    ###########################################
    # Loss Function
    ###########################################

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(cycles):  # loop over the dataset multiple times

        running_loss = 0.0

        for i, (images, labels) in enumerate(train_set_loader(), 0):

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
                print('[%d, %5d] loss: %.3f, accuracy: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100, validate(logfile, net)))
                running_loss = 0.0

    print('Finished Training')

#####################################
# Validation stage
#####################################

def validate(logfile, net):
    dataiter = iter(validation_set_loader())

    hits = 0.0

    for idx, item in enumerate(dataiter):
        images, labels = item

        outputs = net(Variable(images))

        _, predicted = torch.max(outputs.data, 1)

        if  (labels == predicted[0]).all():
            hits += 1

    accuracy = hits / (idx + 1)
    log_accuracy(logfile, accuracy)

    return accuracy

def log_accuracy(filename, accuracy):
    with open(filename, "a") as file:
        file.write(str(accuracy)+ '\n')

#####################################
# Prediction stage
#####################################

def predict(net):

    dataiter = iter(test_set_loader())
    prediction_cnt = {'cloud': 0, 'land': 0, 'nets': 0, 'rock': 0, 'vessel': 0, 'water': 0}

    for idx, item in enumerate(dataiter):
        if idx > int(CONFIG['CNN Prediction']['batch_size']):
            break
        if idx % 100 == 0:
            print('.', end='', flush=True)

        images, _labels = item

        ##########################################################
        # Feed the images into the CNN and check what it predicts
        ##########################################################

        outputs = net(Variable(images))

        _, predicted = torch.max(outputs.data, 1)

        # Save images from prediction for visual check
        if CLASSES[predicted[0]] == 'nets':
            save_image(dataiter.dataset.imgs[idx][0])

        prediction_cnt[CLASSES[predicted[0]]] += 1

    print(f"\nPredicted: {prediction_cnt}")

################################################################
# Train network or use existing one for prediction
################################################################

def main(mode=''):
    image_bands = int(CONFIG['CNN Training']['image_bands'])
    training_cycles = int(CONFIG['CNN Training']['training_cycles'])
    learning_rate = float(CONFIG['CNN Training']['learning_rate'])

    if len(sys.argv) > 1:
        mode = sys.argv[1]

    net = Net(in_channels=image_bands)

    if mode == 'predict' and os.path.exists(CONFIG['CNN Paths']['model_path']):
        print('Use trained network for prediction')
        net.load_state_dict(torch.load(CONFIG['CNN Paths']['model_path']))
        predict(net)
    elif mode == 'train':
        print(f"Start network training for {training_cycles} cycles")
        train(net, training_cycles, learning_rate)
        torch.save(net.state_dict(), CONFIG['CNN Paths']['model_path'])
    else:
        print('No mode provided.')

main()
