import json

from PIL import Image
from torchvision.transforms import transforms

image_size = 224
num_channels = 3


def get_imagenet_class(index):
    with open("imagenet_class_index.json") as json_file:
        imagenet_label_dict = json.load(json_file)

    label = imagenet_label_dict[str(index)][1]

    return label


def get_vgg16_transformation():
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def preprocess_for_vgg16(image_path):
    image_file = Image.open(image_path)
    num_images = 1

    transformation = get_vgg16_transformation()
    return transformation(image_file).view(num_images, num_channels, image_size, image_size)
