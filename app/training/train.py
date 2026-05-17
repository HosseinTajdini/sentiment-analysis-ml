import time
import torch
from tqdm.auto import tqdm
from timeit import default_timer as timer

class Trainer:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
    def train_step(self, dataloader, optimizer, loss_fn, accuracy_fn):
        """Single training step"""
        self.model.train()
        train_loss, train_acc = 0, 0
        start_time = time.time()
        
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)
            
            y_pred = self.model(X)
            loss = loss_fn(y_pred, y)
            train_acc += accuracy_fn(y, y_pred.argmax(dim=1))
            train_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch % 200 == 0:
                print(f"  Batch {batch}, time: {time.time() - start_time:.1f}s")
                
        train_acc /= len(dataloader)
        train_loss /= len(dataloader)
        
        return train_loss, train_acc
    
    def test_step(self, dataloader, loss_fn, accuracy_fn):
        """Single test step"""
        self.model.eval()
        test_loss, test_acc = 0, 0
        
        with torch.inference_mode():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                y_pred = self.model(X)
                
                loss = loss_fn(y_pred, y)
                test_loss += loss.item()
                test_acc += accuracy_fn(y, y_pred.argmax(dim=1))
                
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        
        return test_loss, test_acc
    
    def train(self, train_dataloader, test_dataloader, optimizer, 
              loss_fn, accuracy_fn, epochs=5):
        """Full training loop"""
        start = timer()
        
        results = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": []
        }
        
        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = self.train_step(
                train_dataloader, optimizer, loss_fn, accuracy_fn
            )
            test_loss, test_acc = self.test_step(
                test_dataloader, loss_fn, accuracy_fn
            )
            
            print(f"Epoch {epoch} | Train loss: {train_loss:.4f} | "
                  f"Train acc: {train_acc:.3f} | Test loss: {test_loss:.4f} | "
                  f"Test acc: {test_acc:.3f}")
            
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)
            
        end = timer()
        print(f"Training completed in {end-start:.2f} seconds")
        
        return results