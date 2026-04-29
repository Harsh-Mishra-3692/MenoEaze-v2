import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Tuple, Any

# Define the expected 11 features matching the dataset generator
FEATURE_KEYS = [
    "age",
    "bmi",
    "hot_flash_score",
    "night_sweats_score",
    "sleep_quality",
    "mood_score",
    "fatigue_score",
    "anxiety_score",
    "physical_activity",
    "stress_level",
    "caffeine_intake"
]

def build_sequences(user_logs: List[Dict[str, Any]], seq_length: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts a chronological list of daily user logs into structured 
    tensor sequences and scalar targets for the ML engine.
    
    Handles missing days via safe interpolation/padding and discards 
    sequences that cannot strictly form the (seq_length, num_features) shape.
    
    Args:
        user_logs: Chronological list of daily logs (dicts).
        seq_length: Number of consecutive days needed to form a sequence (default 5).
        
    Returns:
        sequences: torch.Tensor of shape (batch_size, seq_length, 11)
        targets: torch.Tensor of shape (batch_size, 1)
    """
    if not user_logs or len(user_logs) < seq_length + 1:
        return torch.empty((0, seq_length, len(FEATURE_KEYS))), torch.empty((0, 1))
        
    # Convert to DataFrame to safely interpolate missing values
    df = pd.DataFrame(user_logs)
    
    # Ensure all required feature columns exist; if entirely missing, we cannot form the sequence
    for col in FEATURE_KEYS + ["severity"]:
        if col not in df.columns:
            return torch.empty((0, seq_length, len(FEATURE_KEYS))), torch.empty((0, 1))
            
    # Interpolate missing days (forward fill followed by backward fill)
    df[FEATURE_KEYS + ["severity"]] = df[FEATURE_KEYS + ["severity"]].ffill().bfill()
    
    sequences = []
    targets = []
    
    for i in range(len(df) - seq_length):
        seq_df = df.iloc[i : i + seq_length]
        target_row = df.iloc[i + seq_length]
        
        # Extract features and target
        seq_features = seq_df[FEATURE_KEYS].values.astype(np.float32)
        target_val = float(target_row["severity"])
        
        # Defensive check: Ensure valid shape and no NaNs/Infs
        if seq_features.shape != (seq_length, len(FEATURE_KEYS)):
            continue
            
        if np.isnan(seq_features).any() or np.isinf(seq_features).any():
            continue
            
        if np.isnan(target_val) or np.isinf(target_val):
            continue
            
        sequences.append(seq_features)
        targets.append([target_val])
        
    if not sequences:
        return torch.empty((0, seq_length, len(FEATURE_KEYS))), torch.empty((0, 1))
        
    # Convert list of valid arrays into strict tensors
    seq_tensor = torch.tensor(np.array(sequences), dtype=torch.float32)
    target_tensor = torch.tensor(np.array(targets), dtype=torch.float32)
    
    # Final defensive assertion before returning
    assert seq_tensor.shape[1:] == (seq_length, len(FEATURE_KEYS)), f"Shape violation: {seq_tensor.shape}"
    
    return seq_tensor, target_tensor
