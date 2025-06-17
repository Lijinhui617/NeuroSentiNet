import torch

def apply_edo_loss(predictions, targets, sentiment_weights):
    """
    Compute weighted cross-entropy loss using sentiment weights.
    """
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    base_loss = loss_fn(predictions, targets)
    weighted_loss = base_loss * sentiment_weights
    return torch.mean(weighted_loss)
