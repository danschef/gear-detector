import configparser
import os
import sys
from time import localtime, strftime, mktime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from net import Net

from geo_helper import store_image_bounds
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

def train(net, epochs=50, learning_rate=0.001):
    start_time = strftime('%H:%M:%S', localtime())
    print(f"Started training at: {start_time}")

    datetime = strftime("%Y%m%d_%H%M", localtime())
    logfile = f"{CONFIG['CNN Paths']['accuracy_log_path']}/{datetime}.log"

    ###########################################
    # Loss Function
    ###########################################

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(epochs):  # loop over the dataset multiple times

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
            running_loss += loss.item()

            if i % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f, accuracy: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100, validate(logfile, net)))
                running_loss = 0.0

    end_time = strftime('%H:%M:%S', localtime())
    print(f"Finished Training: {end_time}")

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
    print(f"Prediction started at: {strftime('%H:%M:%S', localtime())}")

    dataiter = iter(test_set_loader())
    prediction_cnt = {
        'cloud': 0,
        'edge': 0,
        'land': 0,
        'nets': 0,
        'rock': 0,
        'vessel': 0,
        'water': 0
    }

    datetime = strftime("%Y%m%d_%H%M", localtime())
    prediction_log = f"{CONFIG['CNN Paths']['predicted_geodata_path']}/{datetime}.json"
    prediction_img_folder = f"{CONFIG['CNN Paths']['predicted_imagery_path']}/{datetime}"

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
            image_path = dataiter._dataset.imgs[idx][0]
            save_image(image_path, prediction_img_folder)
            store_image_bounds(image_path, prediction_log)

        prediction_cnt[CLASSES[predicted[0]]] += 1

    print(f"\nPrediction ended at: {strftime('%H:%M:%S', localtime())}")
    print(f"\nPredicted: {prediction_cnt}")

def model_full_path(path, checkpoint):
    return f"{path}_{checkpoint}.pt"


################################################################
# Train network or use existing one for prediction
################################################################

def main(mode=''):
    image_bands = int(CONFIG['CNN Training']['image_bands'])
    training_epochs = int(CONFIG['CNN Training']['epochs'])
    resume_epochs = int(CONFIG['CNN Resume Training']['epochs'])
    learning_rate = float(CONFIG['CNN Training']['learning_rate'])
    batch_size = CONFIG['CNN Prediction']['batch_size']

    if len(sys.argv) > 1:
        mode = sys.argv[1]

    net = Net(in_channels=image_bands)
    model_path = CONFIG['CNN Paths']['model_path']
    checkpoint = CONFIG['CNN Prediction']['checkpoint']

    # Use network for prediction
    if mode == 'predict' and os.path.exists(model_full_path(model_path, checkpoint)):
        print(f"Use trained network {checkpoint} for prediction of max {batch_size} images")

        # Load existing model
        model = torch.load(model_full_path(model_path, checkpoint))
        net.load_state_dict(model)
        predict(net)

    # Start training
    elif mode == 'train':
        print(f"Start network training for {training_epochs} epochs")
        train(net, training_epochs, learning_rate)

        # Save model after training
        checkpoint = strftime("%Y%m%d_%H%M", localtime())
        torch.save(net.state_dict(), model_full_path(model_path, checkpoint))

    # Resume training
    elif mode == 'resume':
        checkpoint = CONFIG['CNN Resume Training']['checkpoint']
        print(f"Resume training on Model {checkpoint} for {resume_epochs} epochs")

        # Load existing model and resume training
        model = torch.load(model_full_path(model_path, checkpoint))
        net.load_state_dict(model)
        train(net, resume_epochs, learning_rate)
        torch.save(net.state_dict(), model_full_path(model_path, checkpoint))
    else:
        print('No mode provided.')

main()
