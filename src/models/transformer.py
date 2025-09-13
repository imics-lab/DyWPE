"""
Time Series Transformer Implementation
"""

import torch
import torch.nn as nn
from .embeddings import TimeSeriesPatchEmbeddingLayer


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_timesteps,
        in_channels,
        patch_size,
        embedding_dim,
        num_transformer_layers=6,
        num_heads=8,
        dim_feedforward=128,
        dropout=0.1,
        num_classes=2
    ):
        super().__init__()
        
        self.patch_embedding = TimeSeriesPatchEmbeddingLayer(
            in_channels=in_channels,
            patch_size=patch_size,
            embedding_dim=embedding_dim,
            input_timesteps=input_timesteps
        )

        self.num_patches = -(-input_timesteps // patch_size)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=num_transformer_layers
        )

        self.ff_layer = nn.Linear(embedding_dim, dim_feedforward)
        self.classifier = nn.Linear(dim_feedforward, num_classes)

    def set_positional_encoder(self, pos_encoder):
        self.patch_embedding.set_positional_encoder(pos_encoder)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)
        class_token_output = x[:, 0, :]
        x = self.ff_layer(class_token_output)
        output = self.classifier(x)
        return output


def create_model_with_dywpe(
    input_timesteps,
    in_channels,
    patch_size=8,
    embedding_dim=128,
    num_transformer_layers=4,
    num_heads=4,
    dim_feedforward=128,
    dropout=0.2,
    num_classes=14,
    max_level=3,
    wavelet='db4'
):
    """
    Factory function that creates a transformer with your original DyWPE implementation.
    """
    from ..core.dywpe import DyWPE
    
    model = TimeSeriesTransformer(
        input_timesteps=input_timesteps,
        in_channels=in_channels,
        patch_size=patch_size,
        embedding_dim=embedding_dim,
        num_transformer_layers=num_transformer_layers,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_classes=num_classes
    )
    
    dywpe = DyWPE(
        d_model=embedding_dim,
        d_x=in_channels,
        max_level=max_level,
        wavelet=wavelet
    )
    
    model.set_positional_encoder(dywpe)
    return model