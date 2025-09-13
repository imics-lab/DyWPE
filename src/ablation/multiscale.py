"""
Multi-Scale Ablation Study

Tests whether hierarchical wavelet decomposition is essential.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict

from ..core.dywpe import DyWPE


class SingleScaleDyWPE(nn.Module):
    """
    Single-Scale Dynamic PE - keeps signal-awareness but removes multi-scale decomposition.
    Uses only J=1 decomposition level.
    """
    
    def __init__(self, d_model: int, d_x: int, wavelet: str = 'db4', dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_x = d_x
        self.wavelet = wavelet
        self.max_level = 1  # ABLATION: Force single scale
        self.dropout = nn.Dropout(dropout)
        
        # Channel projection
        self.channel_proj = nn.Linear(d_x, 1)
        
        # Only 2 scales: approximation + 1 detail level
        num_scales = 2
        self.scale_embeddings = nn.Parameter(torch.randn(num_scales, d_model))
        
        # Same gating mechanism as full DyWPE
        self.gate_w_g = nn.Linear(d_model, d_model)
        self.gate_w_v = nn.Linear(d_model, d_model)
        
        # Simple wavelet filters (Haar-like)
        self.register_buffer('lowpass_filter', torch.tensor([0.7071, 0.7071]))
        self.register_buffer('highpass_filter', torch.tensor([0.7071, -0.7071]))
        
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        with torch.no_grad():
            for scale_idx in range(2):
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
    
    def _safe_dwt(self, x):
        try:
            batch_size, seq_len = x.shape
            
            if seq_len < len(self.lowpass_filter):
                padding_needed = len(self.lowpass_filter) - seq_len
                x = F.pad(x, (0, padding_needed), mode='replicate')
                seq_len = x.shape[1]
            
            if seq_len % 2 == 1:
                x = F.pad(x, (0, 1), mode='replicate')
                seq_len += 1
            
            x_reshaped = x.unsqueeze(1)
            
            filter_len = len(self.lowpass_filter)
            padding = filter_len // 2
            
            lowpass = self.lowpass_filter.view(1, 1, -1)
            highpass = self.highpass_filter.view(1, 1, -1)
            
            approx = F.conv1d(x_reshaped, lowpass, stride=2, padding=padding)
            detail = F.conv1d(x_reshaped, highpass, stride=2, padding=padding)
            
            return approx.squeeze(1), detail.squeeze(1)
            
        except Exception as e:
            print(f"DWT failed: {e}, using fallback")
            half_len = seq_len // 2
            approx = x[:, ::2][:, :half_len]
            detail = x[:, 1::2][:, :half_len]
            return approx, detail
    
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
        """Signal-aware but single-scale decomposition."""
        batch_size, seq_len, _ = x.shape
        
        try:
            # Signal-aware: use actual input signal
            x_mono = self.channel_proj(x).squeeze(-1)
            
            # ABLATION: Only single-level decomposition (J=1)
            approx, detail = self._safe_dwt(x_mono)
            
            # Modulate with signal coefficients (keeping signal-awareness)
            approx_mod = self._gated_modulation(self.scale_embeddings[0], approx)
            detail_mod = self._gated_modulation(self.scale_embeddings[1], detail)
            
            # Combine scales
            combined_encoding = self._interpolate_to_length(approx_mod, seq_len)
            detail_interpolated = self._interpolate_to_length(detail_mod, seq_len)
            combined_encoding = combined_encoding + detail_interpolated
            
            # Scale normalization
            combined_encoding = combined_encoding * (self.d_model ** -0.5)
            
        except Exception as e:
            print(f"Single-Scale Dynamic PE failed: {e}")
            combined_encoding = torch.randn(batch_size, seq_len, self.d_model, device=x.device) * 0.01
        
        return self.dropout(combined_encoding)


class GatedConvolutionalPE(nn.Module):
    """
    Alternative single-scale approach using learnable convolutional filters
    instead of fixed wavelets.
    """
    
    def __init__(self, d_model: int, d_x: int, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_x = d_x
        self.dropout = nn.Dropout(dropout)
        
        # Channel projection
        self.channel_proj = nn.Linear(d_x, 1)
        
        # Learnable convolutional filter
        self.conv_filter = nn.Conv1d(1, d_model, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        
        # Gating mechanism
        self.gate_w_g = nn.Linear(d_model, d_model)
        self.gate_w_v = nn.Linear(d_model, d_model)
        
        # Global embedding for modulation
        self.global_embedding = nn.Parameter(torch.randn(d_model))
        
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        with torch.no_grad():
            nn.init.normal_(self.global_embedding, mean=0.0, std=0.02)
            nn.init.normal_(self.conv_filter.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Signal-aware single-scale approach with learned filters."""
        batch_size, seq_len, _ = x.shape
        
        try:
            # Signal-aware: use actual input signal
            x_mono = self.channel_proj(x).unsqueeze(1)  # (B, 1, L)
            
            # Apply learned convolutional filter
            conv_features = self.conv_filter(x_mono)  # (B, d_model, L)
            conv_features = conv_features.permute(0, 2, 1)  # (B, L, d_model)
            
            # Gated modulation with global embedding
            gate_g = torch.sigmoid(self.gate_w_g(self.global_embedding))
            gate_v = torch.tanh(self.gate_w_v(self.global_embedding))
            combined_gate = gate_g * gate_v
            
            # Apply gating to conv features
            gated_features = conv_features * combined_gate.unsqueeze(0).unsqueeze(0)
            
            # Scale normalization
            final_encoding = gated_features * (self.d_model ** -0.5)
            
        except Exception as e:
            print(f"Gated Convolutional PE failed: {e}")
            final_encoding = torch.randn(batch_size, seq_len, self.d_model, device=x.device) * 0.01
        
        return self.dropout(final_encoding)


def run_multiscale_ablation(
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
    Run multi-scale ablation study.
    
    This is a placeholder - implement full study if needed.
    """
    print("Multi-scale ablation study would compare:")
    print("1. DyWPE (Multi-Scale): Full hierarchical decomposition")
    print("2. Single-Scale DyWPE: J=1 decomposition only")
    print("3. Gated Conv PE: Alternative single-scale approach")
    print("Implementation pending...")
    
    return {"status": "not_implemented"}