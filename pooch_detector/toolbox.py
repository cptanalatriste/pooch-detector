import os

import torch
import torchvision.datasets as datasets


def calculate_percentage(input, predicate):
    count = [predicate(item) for item in input].count(True)
    return float(count) / len(input)


def apply_classification_model(classification_model,
                               preprocess_function,
                               image_path):
    preprocessed_image = preprocess_function(image_path)
    if torch.cuda.is_available():
        preprocessed_image = preprocessed_image.cuda()

    with torch.no_grad():
        classification_model.eval()
        log_probabilities = classification_model.forward(preprocessed_image)

    probabilities = torch.exp(log_probabilities)
    _, indices = torch.topk(probabilities, k=1, dim=1)

    index = indices.item()

    return index


def get_class_names(root_folder):
    dataset_folder = datasets.ImageFolder(root=root_folder)
    return [raw_classname[4:].replace("_", " ") for raw_classname in dataset_folder.classes]


def get_data_loader(root_folder, transform, batch_size, num_workers=0):
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


def calculate_loss(model, criterion, features, labels):
    with torch.no_grad():
        network_output = model.forward(features)
        loss = criterion(network_output, labels)

    return loss.item()


def save_if_improved(model, epoch_val_loss, file_path):
    if model.current_val_loss is None or model.current_val_loss > epoch_val_loss:
        print("Saving model at ", file_path)
        torch.save(model.state_dict(), file_path)
        model.current_val_loss = epoch_val_loss


def update_model_parameters(model, features, labels, criterion, optimiser):
    network_output = model.forward(features)

    loss = criterion(network_output, labels)
    loss.backward()
    optimiser.step()

    return loss.item()


def load_model(model, file_path):
    if os.path.exists(file_path):
        if torch.cuda.is_available():
            map_location = torch.device('cuda')
        else:
            map_location = torch.device('cpu')

        print("Loading parameters from ", file_path)
        state_dict = torch.load(file_path, map_location=map_location)
        model.load_state_dict(state_dict)
