"""
Signal-Awareness Ablation Study

Tests whether signal-awareness (P = f(x, θ)) is better than static approaches (P = f(θ)).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict

from ..core.dywpe import DyWPE


class StaticWaveletPE(nn.Module):
    """
    Static Wavelet PE - same multi-scale framework but no signal dependency.
    This tests P = f(θ) vs P = f(x, θ).
    """
    
    def __init__(self, d_model: int, max_level: int = 3, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.max_level = max_level
        self.dropout = nn.Dropout(dropout)
        
        # Same scale embeddings as DyWPE
        num_scales = self.max_level + 1
        self.scale_embeddings = nn.Parameter(torch.randn(num_scales, d_model))
        
        # CRITICAL DIFFERENCE: Fixed learnable coefficients instead of signal-derived
        approx_length = max_len // (2 ** self.max_level)
        self.static_approx_coeffs = nn.Parameter(torch.randn(1, approx_length))
        
        self.static_detail_coeffs = nn.ParameterList([
            nn.Parameter(torch.randn(1, max_len // (2 ** (level + 1))))
            for level in range(self.max_level)
        ])
        
        # Same gating mechanism as DyWPE
        self.gate_w_g = nn.Linear(d_model, d_model)
        self.gate_w_v = nn.Linear(d_model, d_model)
        
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        with torch.no_grad():
            # Initialize scale embeddings
            for scale_idx in range(self.max_level + 1):
                freq_factor = 2.0 ** (-scale_idx)
                positions = torch.arange(self.d_model, dtype=torch.float32)
                div_term = torch.exp(positions * -(np.log(10000.0) / self.d_model))
                
                if self.d_model % 2 == 0:
                    self.scale_embeddings.data[scale_idx, 0::2] = torch.sin(freq_factor * div_term[0::2])
                    self.scale_embeddings.data[scale_idx, 1::2] = torch.cos(freq_factor * div_term[1::2])
                else:
                    self.scale_embeddings.data[scale_idx, 0::2] = torch.sin(freq_factor * div_term[0::2])
                    if len(div_term[1::2]) > 0:
                        self.scale_embeddings.data[scale_idx, 1::2] = torch.cos(freq_factor * div_term[1::2])
            
            self.scale_embeddings.data = F.normalize(self.scale_embeddings.data, dim=-1)
            
            # Initialize static coefficients
            for coeffs in [self.static_approx_coeffs] + list(self.static_detail_coeffs):
                nn.init.normal_(coeffs, mean=0.0, std=0.1)
    
    def _gated_modulation(self, scale_embedding: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        gate_g = torch.sigmoid(self.gate_w_g(scale_embedding))
        gate_v = torch.tanh(self.gate_w_v(scale_embedding))
        combined_gate = gate_g * gate_v
        return coeffs.unsqueeze(-1) * combined_gate.unsqueeze(0).unsqueeze(0)
    
    def _interpolate_to_length(self, tensor: torch.Tensor, target_length: int) -> torch.Tensor:
        if tensor.size(1) == target_length:
            return tensor
        
        try:
            tensor_reshaped = tensor.permute(0, 2, 1)
            interpolated = F.interpolate(
                tensor_reshaped, size=target_length,
                mode='linear', align_corners=False
            )
            return interpolated.permute(0, 2, 1)
        except:
            current_len = tensor.size(1)
            if current_len < target_length:
                repeat_factor = (target_length + current_len - 1) // current_len
                repeated = tensor.repeat(1, repeat_factor, 1)
                return repeated[:, :target_length, :]
            else:
                return tensor[:, :target_length, :]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        CRITICAL: This ignores the input signal content entirely!
        Uses only fixed learned coefficients.
        """
        B, L, _ = x.shape
        
        try:
            # ABLATION: Use static coefficients instead of signal-derived ones
            static_approx = self.static_approx_coeffs.expand(B, -1)
            static_details = [coeff.expand(B, -1) for coeff in self.static_detail_coeffs]
            
            # Apply same gated modulation as DyWPE
            approx_mod = self._gated_modulation(self.scale_embeddings[0], static_approx)
            
            detail_mods = []
            for i, static_detail in enumerate(static_details):
                if i + 1 < len(self.scale_embeddings):
                    level_embedding = self.scale_embeddings[i + 1]
                    modulated_detail = self._gated_modulation(level_embedding, static_detail)
                    detail_mods.append(modulated_detail)
            
            # Same reconstruction process as DyWPE
            combined_encoding = self._interpolate_to_length(approx_mod, L)
            
            for detail_mod in detail_mods:
                detail_interpolated = self._interpolate_to_length(detail_mod, L)
                combined_encoding = combined_encoding + detail_interpolated
            
            # Scale normalization
            combined_encoding = combined_encoding * (self.d_model ** -0.5)
            
        except Exception as e:
            print(f"Static Wavelet PE failed: {e}")
            combined_encoding = torch.randn(B, L, self.d_model, device=x.device) * 0.01
        
        return self.dropout(combined_encoding)


def run_signal_awareness_ablation(
    dataset_name: str,
    model_params: Dict,
    pe_params: Dict,
    train_loader,
    valid_loader,
    test_loader,
    num_epochs: int = 50,
    device: str = 'cuda'
):
    """
    Run signal-awareness ablation study.
    
    This is a placeholder - implement full study if needed.
    """
    print("Signal-awareness ablation study would compare:")
    print("1. DyWPE (Signal-Aware): P = f(x, θ)")
    print("2. Static Wavelet PE: P = f(θ)")
    print("Implementation pending...")
    
    return {"status": "not_implemented"}