import threading
import torch
from typing import Tuple, List

class OnlineLearningBuffer:
    """
    A thread-safe in-memory buffer for asynchronous continual learning.
    Stores training sequences and targets, yielding them in batches.
    
    Hard constraints implemented:
    - Thread-safety via threading.Lock()
    - Strict memory bounding via max_capacity
    - Defensive dropping of NaN/Inf tensors
    """
    
    def __init__(self, max_capacity: int = 10000):
        self.max_capacity = max_capacity
        self.lock = threading.Lock()
        
        self.sequences: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        
    def add_sample(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Safely appends a new sequence and target to the buffer.
        If the buffer reaches max_capacity, the oldest samples are dropped to prevent OOM.
        
        Args:
            x: Tensor of shape (seq_len, num_features)
            y: Tensor of shape (1,)
        """
        # Defensive validation: Trust no input
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("Inputs must be torch Tensors.")
            
        if torch.isnan(x).any() or torch.isinf(x).any() or torch.isnan(y).any() or torch.isinf(y).any():
            return  # Drop corrupted inputs silently to preserve service continuity
            
        with self.lock:
            # Force CPU detachment to prevent GPU memory leaks from autograd graphs
            self.sequences.append(x.detach().cpu())
            self.targets.append(y.detach().cpu())
            
            # Enforce strict memory bounds
            if len(self.sequences) > self.max_capacity:
                excess = len(self.sequences) - self.max_capacity
                self.sequences = self.sequences[excess:]
                self.targets = self.targets[excess:]
                
    def get_batch(self, min_size: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the buffer contents if the buffer has reached `min_size`.
        Upon yielding, the buffer is cleared.
        
        Args:
            min_size: Minimum number of samples required to trigger a batch yield.
            
        Returns:
            x_batch: Tensor of shape (batch_size, seq_len, num_features) or empty tensor
            y_batch: Tensor of shape (batch_size, 1) or empty tensor
        """
        with self.lock:
            if len(self.sequences) < min_size:
                return torch.empty(0), torch.empty(0)
                
            x_batch = torch.stack(self.sequences)
            y_batch = torch.stack(self.targets)
            
            # Clear buffer safely under lock
            self.sequences.clear()
            self.targets.clear()
            
            return x_batch, y_batch
            
    def __len__(self) -> int:
        with self.lock:
            return len(self.sequences)
