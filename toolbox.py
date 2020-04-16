import torch
from torchvision.transforms import transforms
import cv2
import json


def calculate_percentage(input, predicate):
    count = [predicate(item) for item in input].count(True)
    return float(count) / len(input)


def apply_classification_model(classification_model,
                               preprocess_function,
                               image_path):
    preprocessed_image = preprocess_function(image_path)

    with torch.no_grad():
        classification_model.eval()
        log_probabilities = classification_model.forward(preprocessed_image)

    probabilities = torch.exp(log_probabilities)
    _, indices = torch.topk(probabilities, k=1, dim=1)

    index = indices.item()

    with open("imagenet_class_index.json") as json_file:
        imagenet_label_dict = json.load(json_file)

    label = imagenet_label_dict[str(index)][1]

    return index, label


def preprocess_for_vgg16(image_path):
    image_file = cv2.imread(image_path)

    num_images = 1
    image_size = 224
    num_channels = 3
    transformation = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return transformation(image_file).view(num_images, num_channels, image_size, image_size)
