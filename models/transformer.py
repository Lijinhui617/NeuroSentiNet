import torch
import torch.nn as nn

class SentimentAwareTransformer(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, num_heads=4, num_classes=6):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        out = self.fc(x)
        return out
