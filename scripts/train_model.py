#!/usr/bin/env python
import sys
import os
import yaml
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from app.data.preprocessing import DataPreprocessor
from app.data.dataloader import create_dataloaders
from app.models.base_model import ModelFactory
from app.training.train import Trainer
from app.training.utils import create_optim_loss, accuracy_fn
import torch
import pickle

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='Train sentiment analysis model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--model', type=str, choices=['lstm', 'cnn', 'gru'], help='Model type')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.model:
        config['model']['type'] = args.model
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    
    print(f"Model type: {config['model']['type']}")
    print(f"Model params: {config['model']['params']}")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Preprocess data
    print("Loading and preprocessing data...")
    preprocessor = DataPreprocessor(
        max_len=config['training']['max_len'],
        tokenizer_cache_dir=config['tokenizer']['cache_dir']
    )
    
    df = preprocessor.load_and_clean_data(config['data']['raw_path'])
    texts_tensor, labels_tensor = preprocessor.tokenize_data(df)
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, test_loader, weight_tensor = create_dataloaders(
        texts_tensor, labels_tensor,
        batch_size=config['training']['batch_size'],
        test_size=config['training']['test_size'],
        random_state=config['training']['random_state'],
        num_workers=config['training']['num_workers']
    )
    
    # Create model using factory
    print("Creating model...")
    tokenizer = preprocessor.get_tokenizer()
    model = ModelFactory.create_model(
        model_type=config['model']['type'],
        vocab_size=tokenizer.n_vocab,
        output_dim=2,
        **config['model']['params']
    )
    
    # Print model info
    print(f"Model info: {model.get_model_info() if hasattr(model, 'get_model_info') else 'N/A'}")
    
    # Create optimizer and loss
    optimizer, loss_fn = create_optim_loss(
        model, 
        config['training']['learning_rate'],
        weight_tensor,
        device
    )
    
    # Train model
    print("Starting training...")
    trainer = Trainer(model, device)
    results = trainer.train(
        train_loader, test_loader, optimizer, loss_fn,
        accuracy_fn, epochs=config['training']['epochs']
    )
    
    # Save model
    os.makedirs(config['data']['model_dir'], exist_ok=True)
    model_filename = f"sentiment_model_{config['model']['type']}.pth"
    model_path = f"{config['data']['model_dir']}/{model_filename}"
    torch.save(model.state_dict(), model_path)
    
    # Save config used for training
    config_path_saved = f"{config['data']['model_dir']}/training_config.yaml"
    with open(config_path_saved, 'w') as f:
        yaml.dump(config, f)
    
    print(f"Model saved to {model_path}")
    print(f"Training config saved to {config_path_saved}")
    print("Training completed!")

if __name__ == "__main__":
    main()