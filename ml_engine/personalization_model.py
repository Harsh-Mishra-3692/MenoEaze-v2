# personalization_model.py

import torch
import torch.nn as nn


class PersonalizationModel(nn.Module):
    """
    Input:
        [base_severity, mean_error, trend, variability]
    Output:
        correction (delta to add)
    """

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)
