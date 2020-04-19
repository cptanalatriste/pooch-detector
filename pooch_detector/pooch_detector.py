import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import transforms

from pooch_detector import vgg_utils, toolbox


class TransferLearningNet:

    def __init__(self, num_classes):
        self.model = models.vgg16(pretrained=True)
        self.model.current_val_loss = None

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.classifier_inputs = self.model.classifier[0].in_features

        linear1_layer_output = 4096
        linear2_layer_output = 1000

        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=self.classifier_inputs, out_features=linear1_layer_output),
            nn.ReLU(),
            nn.Linear(in_features=linear1_layer_output, out_features=linear2_layer_output),
            nn.ReLU(),
            nn.Linear(in_features=linear2_layer_output, out_features=num_classes))

    def parameters(self):
        return self.model.classifier.parameters()

    def forward(self, x):
        return self.model.forward(x)

    def state_dict(self):
        return self.model.state_dict()

    def cuda(self):
        self.model.cuda()
        return self

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def load_model(self, file_path):
        toolbox.load_model(self.model, file_path)

    def save_if_improved(self, epoch_val_loss, file_path):
        toolbox.save_if_improved(self.model, epoch_val_loss, file_path)

    def calculate_loss(self, criterion, features, labels):
        return toolbox.calculate_loss(self.model, criterion, features, labels)

    def update_model_parameters(self, features, labels, criterion, optimiser):
        return toolbox.update_model_parameters(self.model, features, labels, criterion, optimiser)


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
        return toolbox.update_model_parameters(self, features, labels, criterion, optimiser)

    def load_model(self, file_path):
        toolbox.load_model(self, file_path)

    def save_if_improved(self, epoch_val_loss, file_path):
        toolbox.save_if_improved(self, epoch_val_loss, file_path)

    def calculate_loss(self, criterion, features, labels):
        return toolbox.calculate_loss(self, criterion, features, labels)


def reset_gradients(optimiser):
    optimiser.zero_grad()


def get_training_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(vgg_utils.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=vgg_utils.normalization_means,
                             std=vgg_utils.normalization_stds)

    ])
