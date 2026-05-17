from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np

def create_dataloaders(texts_tensor, labels_tensor, batch_size=64, 
                       test_size=0.2, random_state=42, num_workers=4):
    """Create train and test dataloaders"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts_tensor, labels_tensor, 
        test_size=test_size, 
        random_state=random_state,
        stratify=labels_tensor
    )
    
    # Compute class weights
    classes = np.unique(labels_tensor.numpy())
    class_weights = compute_class_weight(
        'balanced',
        classes=classes,
        y=labels_tensor.numpy()
    )
    weight_tensor = torch.tensor(class_weights, dtype=torch.float)
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2
    )
    
    return train_loader, test_loader, weight_tensor