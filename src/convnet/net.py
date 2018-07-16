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

    def __init__(self, in_channels=4, num_classes=6):
        super(Net, self).__init__()
        # 1st Convolution Layer (4 image input channels, 8 output channels, 3x3 convolution kernel)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3)

        # Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2)

        # 2nd Convolution Layer
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)

        # Apply linear transformations to the incoming data
        # torch.nn.Linear(in_features, out_features, bias=True)
        self.fc1 = nn.Linear(in_features=16 * 9 * 9, out_features=num_classes)

    def forward(self, data):
        ### data.size() >>> [1, 4, 42, 42]
        output = Func.relu(self.conv1(data))
        ### output.size() >>> [1, 8, 40, 40]
        output = self.pool(output)
        ### output.size() >>> [1, 8, 20, 20]
        output = Func.relu(self.conv2(output))
        ### output.size() >>> [1, 16, 18, 18]
        output = self.pool(output)
        ### output.size() >>> [1, 16, 9, 9]
        output = output.view(-1, 16 * 9 * 9)
        ### output.size() >>> [1, 1296]
        output = Func.relu(self.fc1(output))

        return output
