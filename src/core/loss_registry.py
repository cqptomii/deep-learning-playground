from torch import nn

_LOSS_REGISTRY = {
    "cross_entropy": nn.CrossEntropyLoss,
}

def get_loss_fn(loss_fn : str):
    if loss_fn not in _LOSS_REGISTRY:
        available = ", ".join(sorted(_LOSS_REGISTRY))
        raise ValueError(f"Unknown loss '{loss_fn}'. Available losses: {available}.")
    return _LOSS_REGISTRY[loss_fn]()
