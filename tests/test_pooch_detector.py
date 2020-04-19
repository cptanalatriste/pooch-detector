import copy
import os
import unittest

from torch import nn, optim

from .context import pooch_detector


class TestPoochDetector(unittest.TestCase):

    def setUp(self):
        self.batch_size = 4

        input_channels = 3
        self.num_classes = 133
        self.scratch_net = pooch_detector.Net(input_channels=input_channels, num_classes=self.num_classes)
        self.transfer_net = pooch_detector.TransferLearningNet(num_classes=self.num_classes)

        self.criterion = nn.CrossEntropyLoss()

        training_transform = pooch_detector.get_training_transform()
        self.train_dataloader = pooch_detector.toolbox.get_data_loader(root_folder="dogImages/train",
                                                                       transform=training_transform,
                                                                       batch_size=self.batch_size)

    def test_transfer_net_structure(self):

        vgg16_classifier_inputs = 25088
        self.assertEqual(self.transfer_net.classifier_inputs, vgg16_classifier_inputs)

        num_outputs = self.transfer_net.model.classifier[-1].out_features
        self.assertEqual(num_outputs, self.num_classes)

    def test_transfer_net_training(self):

        optimizer_transfer = optim.SGD(self.transfer_net.parameters(), lr=0.001, momentum=0.9)

        parameters_before_training = copy.deepcopy(self.transfer_net.state_dict())
        data_iterator = iter(self.train_dataloader)
        images, training_labels = data_iterator.next()
        self.assertEqual(len(images), self.batch_size)

        pooch_detector.reset_gradients(optimiser=optimizer_transfer)
        loss_value = self.transfer_net.update_model_parameters(features=images,
                                                               labels=training_labels,
                                                               criterion=self.criterion,
                                                               optimiser=optimizer_transfer)

        parameters_after_training = copy.deepcopy(self.transfer_net.state_dict())

        self.assertIsInstance(loss_value, float)
        self.assertFalse(
            pooch_detector.toolbox.compare_model_parameters(parameters_before_training, parameters_after_training))

    def test_scratch_net_training(self):

        optimizer_scratch = optim.SGD(self.scratch_net.parameters(), lr=0.001, momentum=0.9)

        parameters_before_training = copy.deepcopy(self.scratch_net.state_dict())
        data_iterator = iter(self.train_dataloader)
        images, training_labels = data_iterator.next()
        self.assertEqual(len(images), self.batch_size)

        pooch_detector.reset_gradients(optimiser=optimizer_scratch)
        loss_value = self.scratch_net.update_model_parameters(features=images,
                                                              labels=training_labels,
                                                              criterion=self.criterion,
                                                              optimiser=optimizer_scratch)

        parameters_after_training = copy.deepcopy(self.scratch_net.state_dict())

        self.assertIsInstance(loss_value, float)
        self.assertFalse(
            pooch_detector.toolbox.compare_model_parameters(parameters_before_training, parameters_after_training))

    def test_smoke_validation(self):

        testing_transform = pooch_detector.vgg_utils.get_input_transform()
        valid_dataloader = pooch_detector.toolbox.get_data_loader(root_folder="dogImages/valid",
                                                                  transform=testing_transform,
                                                                  batch_size=self.batch_size)
        parameters_before_validation = copy.deepcopy(self.scratch_net.state_dict())
        data_iterator = iter(valid_dataloader)
        images, valid_labels = data_iterator.next()
        self.assertEqual(len(images), self.batch_size)

        loss_value = self.scratch_net.calculate_loss(self.criterion, features=images, labels=valid_labels)
        parameters_after_validation = copy.deepcopy(self.scratch_net.state_dict())

        self.assertIsInstance(loss_value, float)
        self.assertTrue(
            pooch_detector.toolbox.compare_model_parameters(parameters_before_validation, parameters_after_validation))

    def test_save_if_improved(self):
        self.scratch_net.current_val_loss = None
        first_epoch_loss = 1.0
        file_path = "../test.pth"

        if os.path.exists(file_path):
            os.remove(file_path)

        self.scratch_net.save_if_improved(epoch_val_loss=first_epoch_loss, file_path=file_path)

        self.assertEqual(first_epoch_loss, self.scratch_net.current_val_loss)
        self.assertTrue(os.path.exists(file_path))

        second_epoch_loss = 10.0
        new_test_path = "test_new.pth"
        if os.path.exists(new_test_path):
            os.remove(new_test_path)

        self.scratch_net.save_if_improved(epoch_val_loss=second_epoch_loss, file_path=new_test_path)
        self.assertEqual(first_epoch_loss, self.scratch_net.current_val_loss)
        self.assertFalse(os.path.exists(new_test_path))
