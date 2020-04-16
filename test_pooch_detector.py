import copy
import unittest

from torch import nn, optim

import pooch_detector
import toolbox
import vgg_utils
from pooch_detector import Net


class TestPoochDetector(unittest.TestCase):

    def setUp(self):
        self.vgg16_transform = vgg_utils.get_vgg16_transformation()
        self.batch_size = 4

        input_channels = 3
        num_classes = 133
        self.model = Net(input_channels=input_channels, num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def test_smoke_training(self):
        optimizer_scratch = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        train_dataloader = toolbox.get_data_loader(root_folder="dogImages/train",
                                                   transform=self.vgg16_transform, batch_size=self.batch_size)

        parameters_before_training = copy.deepcopy(self.model.state_dict())
        data_iterator = iter(train_dataloader)
        images, training_labels = data_iterator.next()
        self.assertEqual(len(images), self.batch_size)

        pooch_detector.reset_gradients(optimiser=optimizer_scratch)
        loss_value = self.model.update_model_parameters(features=images,
                                                        labels=training_labels,
                                                        criterion=self.criterion,
                                                        optimiser=optimizer_scratch)

        parameters_after_training = copy.deepcopy(self.model.state_dict())

        self.assertIsInstance(loss_value, float)
        self.assertFalse(toolbox.compare_model_parameters(parameters_before_training, parameters_after_training))

    def test_smoke_validation(self):
        valid_dataloader = toolbox.get_data_loader(root_folder="dogImages/valid",
                                                   transform=self.vgg16_transform, batch_size=self.batch_size)
        parameters_before_validation = copy.deepcopy(self.model.state_dict())
        data_iterator = iter(valid_dataloader)
        images, valid_labels = data_iterator.next()
        self.assertEqual(len(images), self.batch_size)

        loss_value = self.model.calculate_loss(self.criterion, features=images, labels=valid_labels)
        parameters_after_validation = copy.deepcopy(self.model.state_dict())

        self.assertIsInstance(loss_value, float)
        self.assertTrue(toolbox.compare_model_parameters(parameters_before_validation, parameters_after_validation))
