"""
Time Series Patch Embedding Layer

This module contains only the patch embedding logic, keeping it modular.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.dywpe import DyWPE
from ..core.position_encodings import tAPE, LearnedPositionalEncoding, FixedPositionalEncoding, ConvSPE, TemporalPositionalEncoding, RelativePositionalEncoding, RotaryPositionalEncoding


def get_pos_encoder(pos_encoding, d_model, dropout=0.1, max_len=5000):
    """Factory function to get positional encoder"""
    encoders = {
        'fixed': FixedPositionalEncoding,
        'learned': LearnedPositionalEncoding,
        'rope': RotaryPositionalEncoding,
        'relative': RelativePositionalEncoding,
        'T-PE': TemporalPositionalEncoding,
        'dywpe': DyWPE,
        'tape': tAPE,
        'convspe': ConvSPE,
    }

    if pos_encoding not in encoders:
        available = ', '.join(encoders.keys())
        raise ValueError(f"Unknown positional encoding: {pos_encoding}. Available: {available}")

    # Handle special cases that need additional parameters
    if pos_encoding == 'tape':
        return encoders[pos_encoding](d_model, dropout, max_len, scale_factor=1.0)
    elif pos_encoding in ['tupe', 'convspe']:
        return encoders[pos_encoding](d_model, dropout, max_len, num_heads=8)
    else:
        return encoders[pos_encoding](d_model, dropout, max_len)



class TimeSeriesPatchEmbeddingLayer(nn.Module):
    """
    Time Series Patch Embedding Layer
    
    Implementation for converting time series into patch embeddings.
    """
    def __init__(self, in_channels, patch_size, embedding_dim, input_timesteps, pos_encoding='dywpe'):
        super().__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.in_channels = in_channels

        # Calculate the number of patches
        self.num_patches = -(-input_timesteps // patch_size)  # Ceiling division
        self.padding = (self.num_patches * patch_size) - input_timesteps

        # Convolutional layer for patch creation
        self.conv_layer = nn.Conv1d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Class token embeddings
        self.class_token_embeddings = nn.Parameter(
            torch.randn((1, 1, embedding_dim), requires_grad=True)
        )

        # Store positional encoding type for later assignment
        self.position_embeddings = get_pos_encoder(
            pos_encoding, embedding_dim, dropout=0.1, max_len=input_timesteps
        )
    def set_positional_encoder(self, pos_encoder):
        """Set the positional encoder after initialization."""
        self.position_embeddings = pos_encoder

    def forward(self, x):
        """
        Forward pass matching your original implementation.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, in_channels)
            
        Returns:
            Embedded patches with positional encoding
        """
        # Apply padding if necessary
        if self.padding > 0:
            x = nn.functional.pad(x, (0, 0, 0, self.padding))

        # Convert to (batch, channels, time) for conv1d
        x = x.permute(0, 2, 1)
        
        # Apply convolution to create patches
        conv_output = self.conv_layer(x)
        
        # Convert back to (batch, time, channels)
        conv_output = conv_output.permute(0, 2, 1)

        # Add class token
        batch_size = x.shape[0]
        class_tokens = self.class_token_embeddings.expand(batch_size, -1, -1)
        output = torch.cat((class_tokens, conv_output), dim=1)
        
        # Apply positional encoding if available
        if self.position_embeddings is not None:
            output = self.position_embeddings(output)

        return output