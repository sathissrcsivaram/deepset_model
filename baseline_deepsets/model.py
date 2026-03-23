import torch
import torch.nn as nn


class DeepSetsHeatmap(nn.Module):
    def __init__(self, input_dim=12, embed_dim=128, heatmap_size=(50, 50)):
        super().__init__()

        height, width = heatmap_size
        self.height = height
        self.width = width

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim * 3, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.GELU(),
        )
        self.x_head = nn.Linear(512, width)
        self.y_head = nn.Linear(512, height)

    def forward(self, x):
        encoded = self.encoder(x)
        pooled_mean = encoded.mean(dim=1)
        pooled_max = encoded.amax(dim=1)
        pooled_std = torch.sqrt(encoded.var(dim=1, unbiased=False) + 1e-6)
        pooled = torch.cat([pooled_mean, pooled_max, pooled_std], dim=1)
        hidden = self.decoder(pooled)
        x_logits = self.x_head(hidden)
        y_logits = self.y_head(hidden)
        return y_logits.unsqueeze(2) + x_logits.unsqueeze(1)

