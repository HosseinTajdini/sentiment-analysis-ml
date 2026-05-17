from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseSentimentModel(nn.Module, ABC):
    """Base class for all sentiment models"""
    
    @abstractmethod
    def forward(self, x):
        """Forward pass"""
        pass
    
    @abstractmethod
    def get_model_info(self):
        """Return model information"""
        pass

class ModelFactory:
    """Factory to create different model types"""
    
    @staticmethod
    def create_model(model_type, vocab_size, **kwargs):
        """
        Create model based on type
        
        Args:
            model_type: 'lstm', 'cnn', 'bert', etc.
            vocab_size: vocabulary size
            **kwargs: model specific parameters
        """
        if model_type == 'lstm':
            from .lstm_model import SentimentLSTM
            return SentimentLSTM(
                vocab_size=vocab_size,
                embed_dim=kwargs.get('embed_dim', 64),
                hidden_dim=kwargs.get('hidden_dim', 64),
                output_dim=kwargs.get('output_dim', 2),
                n_layers=kwargs.get('n_layers', 3),
                dropout=kwargs.get('dropout', 0.5)
            )
        # elif model_type == 'cnn':
        #     from .cnn_model import SentimentCNN
        #     return SentimentCNN(
        #         vocab_size=vocab_size,
        #         embed_dim=kwargs.get('embed_dim', 64),
        #         num_filters=kwargs.get('num_filters', 100),
        #         filter_sizes=kwargs.get('filter_sizes', [3,4,5]),
        #         output_dim=kwargs.get('output_dim', 2),
        #         dropout=kwargs.get('dropout', 0.5)
        #     )
        else:
            raise ValueError(f"Unknown model type: {model_type}")