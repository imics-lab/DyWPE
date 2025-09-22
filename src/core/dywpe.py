# dywpe.py
import torch
import torch.nn as nn
try:
    import pytorch_wavelets as DWT
except ImportError:
    raise ImportError("Please install pytorch_wavelets: pip install pytorch_wavelets")

class DyWPE(nn.Module):
    """
    Dynamic Wavelet Positional Encoding (DyWPE)
    
    This module generates a signal-aware positional encoding by analyzing the
    input time series signal using the Discrete Wavelet Transform (DWT).
    Instead of relying on fixed indices, it derives positional information
    from the signal's multi-scale dynamics.

    Args:
        d_model (int): The hidden dimension of the transformer model.
        d_x (int): The number of channels in the input time series.
        max_level (int): The maximum level of decomposition for the DWT.
                         Should be chosen carefully based on sequence length.
                         max_level <= log2(L).
        wavelet (str): The name of the wavelet to use (e.g., 'db4', 'haar').
    """
    def __init__(self, d_model: int, d_x: int, max_level: int, wavelet: str = 'db4'):
        super().__init__()
        
        if max_level < 1:
            raise ValueError("max_level must be at least 1.")
            
        self.d_model = d_model
        self.d_x = d_x
        self.max_level = max_level
        
        # DWT and IDWT layers from the pytorch_wavelets library
        self.dwt = DWT.DWT1D(wave=wavelet, J=self.max_level, mode='symmetric')
        self.idwt = DWT.IDWT1D(wave=wavelet, mode='symmetric')

        # Learnable projection to create a single representative channel for DWT
        self.channel_proj = nn.Linear(d_x, 1)

        # Learnable embeddings for each scale (J details + 1 approximation)
        # These act as learnable "prototypes" for each temporal scale.
        num_scales = self.max_level + 1
        self.scale_embeddings = nn.Parameter(torch.randn(num_scales, d_model))

        # Gated modulation mechanism to combine scale embeddings with signal coefficients
        self.gate_w_g = nn.Linear(d_model, d_model)
        self.gate_w_v = nn.Linear(d_model, d_model)

    def _gated_modulation(self, scale_embedding, coeffs):
        """
        Modulates a scale embedding with its corresponding wavelet coefficients.
        Args:
            scale_embedding (torch.Tensor): Shape (d_model,)
            coeffs (torch.Tensor): Shape (B, L_coeffs)
        Returns:
            torch.Tensor: Modulated coefficients, shape (B, L_coeffs, d_model)
        """
        # Project the scale embedding through the gating layers
        gate_g = torch.sigmoid(self.gate_w_g(scale_embedding))
        gate_v = torch.tanh(self.gate_w_v(scale_embedding))
        
        # Combine gates and broadcast with coefficients
        # coeffs.unsqueeze(-1) -> (B, L_coeffs, 1)
        # gate_g * gate_v -> (d_model,)
        modulated_coeffs = coeffs.unsqueeze(-1) * (gate_g * gate_v)
        
        return modulated_coeffs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for DyWPE.
        Args:
            x (torch.Tensor): Input time series tensor of shape (B, L, d_x).
        Returns:
            torch.Tensor: The generated positional encoding of shape (B, L, d_model).
        """
        B, L, _ = x.shape

        # Step 2: Project multivariate signal to a single channel for DWT analysis
        # (B, L, d_x) -> (B, L, 1) -> (B, 1, L)
        x_mono = self.channel_proj(x).permute(0, 2, 1)
        
        # Step 2: Decompose signal into wavelet coefficients
        # Yl: low-pass (approx), Yh: high-pass (details) list
        # Yl shape: (B, 1, L_approx)
        # Yh is a list of J tensors, each of shape (B, 1, L_detail_j)
        Yl, Yh = self.dwt(x_mono)

        # Step 3: Modulate learnable scale embeddings with coefficients
        # Modulate approximation coefficients (coarsest scale)
        # self.scale_embeddings is for the approximation level
        Yl_mod = self._gated_modulation(self.scale_embeddings, Yl.squeeze(1))

        # Modulate detail coefficients for each level
        Yh_mod = []
        for i in range(self.max_level):
            # scale_embeddings[i+1] corresponds to detail level J-i
            level_embedding = self.scale_embeddings[i + 1]
            level_coeffs = Yh[i].squeeze(1)
            
            modulated_detail_coeffs = self._gated_modulation(level_embedding, level_coeffs)
            Yh_mod.append(modulated_detail_coeffs)
        
        # Step 4: Reconstruct to get the final positional encoding
        # The IDWT layer expects inputs of shape (B, C, L), so we permute d_model to the channel dim
        # Yl_mod shape: (B, L_approx, d_model) -> (B, d_model, L_approx)
        # Yh_mod elements: (B, L_detail, d_model) -> (B, d_model, L_detail)
        Yl_mod_p = Yl_mod.permute(0, 2, 1)
        Yh_mod_p = [h.permute(0, 2, 1) for h in Yh_mod]
        
        # The inverse transform reconstructs the embeddings to the original length L
        # Output shape: (B, d_model, L)
        pos_encoding = self.idwt((Yl_mod_p, Yh_mod_p))
        
        # Permute back to standard transformer format (B, L, d_model)
        pos_encoding = pos_encoding.permute(0, 2, 1)
        
        # Ensure output length matches input length, handle potential off-by-one DWT issues
        if pos_encoding.shape[1] != L:
            pos_encoding = nn.functional.pad(pos_encoding, (0, 0, 0, L - pos_encoding.shape[1]))

        return pos_encoding

