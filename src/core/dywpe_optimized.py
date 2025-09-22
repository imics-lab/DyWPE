# dywpe_optimized.py
import torch
import torch.nn as nn
try:
    import pytorch_wavelets as DWT
except ImportError:
    raise ImportError("Please install pytorch_wavelets: pip install pytorch_wavelets")

class DyWPEOptimized(nn.Module):
    """
    Optimized Dynamic Wavelet Positional Encoding (DyWPE)
    
    This optimized version vectorizes the gating mechanism and fuses linear layers
    to improve computational speed and GPU utilization without changing the
    mathematical output of the method.

    Args:
        d_model (int): The hidden dimension of the transformer model.
        d_x (int): The number of channels in the input time series.
        max_level (int): The maximum level of decomposition for the DWT.
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
        num_scales = self.max_level + 1
        self.scale_embeddings = nn.Parameter(torch.randn(num_scales, d_model))

        # --- OPTIMIZATION 2: Fuse gating layers ---
        # Fuse the two linear layers into one for a single, more efficient matmul.
        self.gate_proj = nn.Linear(d_model, 2 * d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the optimized DyWPE.
        Args:
            x (torch.Tensor): Input time series tensor of shape (B, L, d_x).
        Returns:
            torch.Tensor: The generated positional encoding of shape (B, L, d_model).
        """
        B, L, _ = x.shape

        # Project multivariate signal to a single channel for DWT analysis
        x_mono = self.channel_proj(x).permute(0, 2, 1)
        
        # Decompose signal into wavelet coefficients
        Yl, Yh = self.dwt(x_mono)

        # --- OPTIMIZATION 1: Vectorize the Gating Calculation ---
        # Compute all gating signals at once in a single batched operation
        # before any looping.
        
        # Project all scale embeddings through the fused gate layer
        # (num_scales, d_model) -> (num_scales, 2 * d_model)
        gated_output = self.gate_proj(self.scale_embeddings)
        
        # Split the output into the two gate components
        # (num_scales, 2 * d_model) -> 2 x (num_scales, d_model)
        gates_g_raw, gates_v_raw = gated_output.chunk(2, dim=-1)
        
        # Apply activations
        all_gates_g = torch.sigmoid(gates_g_raw)
        all_gates_v = torch.tanh(gates_v_raw)
        
        # Combine the gates
        # Shape: (num_scales, d_model)
        combined_gates = all_gates_g * all_gates_v

        # --- Apply modulation in a much lighter loop ---
        
        # Modulate approximation coefficients (coarsest scale)
        # (B, L_approx) -> (B, L_approx, 1) * (d_model,) -> (B, L_approx, d_model)
        Yl_squeezed = Yl.squeeze(1)
        Yl_mod = Yl_squeezed.unsqueeze(-1) * combined_gates[0]

        # Modulate detail coefficients for each level
        Yh_mod = []
        # This loop is now much faster as it only contains efficient tensor broadcasting
        for i in range(self.max_level):
            level_coeffs = Yh[i].squeeze(1)
            level_gate = combined_gates[i + 1]
            
            modulated_detail_coeffs = level_coeffs.unsqueeze(-1) * level_gate
            Yh_mod.append(modulated_detail_coeffs)
        
        # Reconstruct to get the final positional encoding
        Yl_mod_p = Yl_mod.permute(0, 2, 1)
        Yh_mod_p = [h.permute(0, 2, 1) for h in Yh_mod]
        
        pos_encoding = self.idwt((Yl_mod_p, Yh_mod_p))
        
        # Permute back to standard transformer format (B, L, d_model)
        pos_encoding = pos_encoding.permute(0, 2, 1)
        
        # Ensure output length matches input length, handle potential off-by-one DWT issues
        if pos_encoding.shape[1] != L:
            pos_encoding = nn.functional.pad(pos_encoding, (0, 0, 0, L - pos_encoding.shape[1]))

        return pos_encoding
