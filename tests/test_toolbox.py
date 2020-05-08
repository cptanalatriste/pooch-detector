import unittest

import torchvision.models as models

from .context import pooch_detector


class TestToolbox(unittest.TestCase):

    def test_calculate_percentage(self):
        predicate = lambda item: item > 0
        test_input = [-2, -1, 0, 1, 2]
        output = pooch_detector.toolbox.calculate_percentage(input=test_input,
                                                             predicate=predicate)
        self.assertAlmostEqual(output, 0.4)

    def test_apply_vgg16(self):
        dog_image = "dogImages/train/001.Affenpinscher/Affenpinscher_00001.jpg"
        vgg16_model = models.vgg16(pretrained=True)

        index = pooch_detector.toolbox.apply_classification_model(classification_model=vgg16_model,
                                                                  preprocess_function=pooch_detector.vgg_utils.preprocess_for_vgg16,
                                                                  image_path=dog_image)
        self.assertIsInstance(index, int)
        self.assertEqual(pooch_detector.vgg_utils.get_imagenet_class(index), "affenpinscher")

    def test_get_classname_map(self):
        root_folder = "dogImages/train"
        class_names = pooch_detector.toolbox.get_class_names(root_folder)

        self.assertEqual("Affenpinscher", class_names[0])
        self.assertEqual("Akita", class_names[3])
        self.assertEqual("American foxhound", class_names[6])
