import torch


def accuracy_fn(y_pred: torch.Tensor, y: torch.Tensor) -> float:
    """
        Calculates the accuracy between predicted and ground truth labels
    :param y_pred: Y predicted labels
    :param y: Y ground truth labels
    :return: Accuracy between predicted and ground truth labels
    """
    correct = torch.eq(y_pred, y).sum().item()
    acc = correct / len(y_pred) * 100
    return acc
