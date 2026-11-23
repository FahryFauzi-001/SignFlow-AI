import os
import torch
import torch.nn as nn


class SignBiGRU(nn.Module):
    def __init__(self, input_dim: int = 98, hidden_dim: int = 256, num_layers: int = 2, num_classes: int = 66):
        super(SignBiGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, 32, 98)
        out, _ = self.gru(x)
        # Temporal Average Pooling: (batch_size, hidden_dim * 2)
        out = torch.mean(out, dim=1)
        return self.fc(out)


def load_trained_model(weights_path: str, num_classes: int, device: torch.device) -> SignBiGRU:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model file not found at: {weights_path}. Ensure 'model.pt' is placed inside models/.")
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    if "fc.1.weight" in state_dict:
        num_classes = state_dict["fc.1.weight"].shape[0]
    model = SignBiGRU(input_dim=98, hidden_dim=256, num_layers=2, num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
