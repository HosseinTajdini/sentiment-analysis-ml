import pandas as pd
import torch
import tiktoken
import os

class DataPreprocessor:
    def __init__(self, max_len=128, tokenizer_cache_dir=None):
        self.max_len = max_len
        if tokenizer_cache_dir:
            os.environ["TIKTOKEN_CACHE_DIR"] = tokenizer_cache_dir
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
    def load_and_clean_data(self, file_path):
        """Load and clean the raw data"""
        raw_df = pd.read_csv(file_path)
        
        # Remove unnamed column
        if 'Unnamed: 0' in raw_df.columns:
            raw_df.drop(columns=['Unnamed: 0'], inplace=True)
        
        # Remove duplicates
        raw_df.drop_duplicates(inplace=True)
        
        # Convert label_id to int
        raw_df['label_id'] = raw_df['label_id'].apply(lambda x: int(x))
        
        # Strip comments
        raw_df['comment'] = raw_df['comment'].str.strip()
        
        return raw_df
    
    def tokenize_data(self, df):
        """Convert comments to tokens"""
        processed_df = pd.DataFrame({
            'tokens': df['comment'].apply(lambda x: self.tokenizer.encode(x)),
            'label': df['label_id']
        }).reset_index(drop=True)
        
        # Pad sequences
        padded_tokens = [tokens[:self.max_len] + [0] * (self.max_len - len(tokens))
                        for tokens in processed_df['tokens']]
        
        texts_tensor = torch.tensor(padded_tokens)
        labels_tensor = torch.tensor(processed_df['label'].values)
        
        return texts_tensor, labels_tensor
    
    def save_processed_data(self, texts_tensor, labels_tensor, output_dir):
        """Save processed tensors"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        torch.save(texts_tensor, f'{output_dir}/texts.pt')
        torch.save(labels_tensor, f'{output_dir}/labels.pt')
        
    def get_tokenizer(self):
        return self.tokenizer