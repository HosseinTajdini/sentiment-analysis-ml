import torch
import tiktoken
import pickle
import yaml
from pathlib import Path
from .base_model import ModelFactory

class SentimentPredictor:
    def __init__(self, model_path, config_path, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.tokenizer = None
        self.config = None
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.load_model(model_path)
        
    def load_model(self, model_path):
        # Initialize tokenizer
        import os
        cache_dir = self.config['tokenizer'].get('cache_dir', 'C:/tiktoken_cache')
        os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Get model type and params
        model_type = self.config['model']['type']
        model_params = self.config['model']['params']
        
        # Create model using factory
        self.model = ModelFactory.create_model(
            model_type=model_type,
            vocab_size=self.tokenizer.n_vocab,
            output_dim=2,
            **model_params
        )
        
        # Load weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, text, max_len=128):
        """Predict sentiment of a single text"""
        tokens = self.tokenizer.encode(text)
        padded = tokens[:max_len] + [0] * (max_len - len(tokens))
        
        input_tensor = torch.tensor([padded]).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1).max().item()
            
        sentiment = "HAPPY" if prediction == 0 else "SAD"
        return {
            "sentiment": sentiment,
            "label_id": prediction,
            "confidence": confidence,
            "model_type": self.config['model']['type']
        }