import unittest
import torchvision.models as models

import toolbox


class TestToolbox(unittest.TestCase):

    def test_calculate_percentage(self):
        predicate = lambda item: item > 0
        test_input = [-2, -1, 0, 1, 2]
        output = toolbox.calculate_percentage(input=test_input,
                                              predicate=predicate)
        self.assertAlmostEqual(output, 0.4)

    def test_apply_vgg16(self):
        dog_image = "dogImages/train/001.Affenpinscher/Affenpinscher_00001.jpg"
        vgg16_model = models.vgg16(pretrained=True)

        _, label = toolbox.apply_classification_model(classification_model=vgg16_model,
                                                      preprocess_function=toolbox.preprocess_for_vgg16,
                                                      image_path=dog_image)

        self.assertEqual(label, "affenpinscher")
