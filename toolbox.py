import torch
import torchvision.datasets as datasets


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

    return index


def get_data_loader(root_folder, transform, batch_size, num_workers=4):
    dataset_folder = datasets.ImageFolder(root=root_folder, transform=transform)
    return torch.utils.data.DataLoader(dataset_folder, shuffle=True, batch_size=batch_size, num_workers=num_workers)


def compare_model_parameters(parameters, more_parameters):
    for first_parameter, second_parameter in zip(parameters, more_parameters):
        if first_parameter.data.ne(second_parameter.data).sum() > 0:
            return False
    return True


def compare_model_parameters(parameters, more_parameters):
    """
    Taken from: https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351/5
    :param parameters:
    :param more_parameters:
    :return:
    """
    models_differ = 0
    for key_item_1, key_item_2 in zip(parameters.items(), more_parameters.items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
                return False
    if models_differ == 0:
        return True
