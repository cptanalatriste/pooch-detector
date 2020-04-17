import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(Net, self).__init__()

        conv1_out_channels = 6
        kernel_size = 5
        self.conv1_layer = nn.Conv2d(in_channels=input_channels, out_channels=conv1_out_channels,
                                     kernel_size=kernel_size)
        self.maxpool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
        conv2_out_channels = 16
        self.conv2_layer = nn.Conv2d(in_channels=conv1_out_channels, out_channels=conv2_out_channels,
                                     kernel_size=kernel_size)
        self.linear1_in_features = conv2_out_channels * 53 * 53
        linear1_out_features = 400
        self.linear1_layer = nn.Linear(in_features=self.linear1_in_features, out_features=linear1_out_features)

        linear2_out_features = 120
        self.linear2_layer = nn.Linear(in_features=linear1_out_features, out_features=linear2_out_features)
        self.linear3_layer = nn.Linear(in_features=linear2_out_features, out_features=num_classes)

        self.current_val_loss = None

    def forward(self, x):
        x = self.maxpool_layer(F.relu(self.conv1_layer(x)))
        x = self.maxpool_layer(F.relu(self.conv2_layer(x)))

        x = x.view(-1, self.linear1_in_features)
        x = F.relu(self.linear1_layer(x))
        x = F.relu(self.linear2_layer(x))

        x = self.linear3_layer(x)
        return x

    def update_model_parameters(self, features, labels, criterion, optimiser):
        network_output = self.forward(features)

        loss = criterion(network_output, labels)
        loss.backward()
        optimiser.step()

        return loss.item()

    def calculate_loss(self, criterion, features, labels):
        with torch.no_grad():
            network_output = self.forward(features)
            loss = criterion(network_output, labels)

        return loss.item()

    def save_if_improved(self, epoch_val_loss, file_path):
        if self.current_val_loss is None or self.current_val_loss > epoch_val_loss:

            print("Saving model at ", file_path)
            torch.save(self.state_dict(), file_path)
            self.current_val_loss = epoch_val_loss


def reset_gradients(optimiser):
    optimiser.zero_grad()
