import torch.nn as nn
import torch.nn.functional as Func

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

    def forward(self, data):
        ### data.size() >>> [4, 3, 42, 42]
        output = Func.relu(self.conv1(data))
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
