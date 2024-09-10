import torch

def number_of_correct(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    return predictions.argmax(dim=-1).eq(targets).sum().item()
