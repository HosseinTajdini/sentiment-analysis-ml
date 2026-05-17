import torch
import torch.nn as nn

def accuracy_fn(y_true, y_pred):
    """Calculate accuracy"""
    return (y_true == y_pred).sum().item() / len(y_true)

def create_optim_loss(model, lr, weight_tensor=None, device=None):
    """Create optimizer and loss function"""
    optim = torch.optim.AdamW(params=model.parameters(), lr=lr)
    
    if weight_tensor is not None:
        loss_fn = nn.CrossEntropyLoss(weight=weight_tensor.to(device))
    else:
        loss_fn = nn.CrossEntropyLoss()
        
    return optim, loss_fn