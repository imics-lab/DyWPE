# Core DyWPE Ablation Studies: Signal-Awareness vs Multi-Scale
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class StaticWaveletPE(nn.Module):
    """
    ABLATION 1: Static Wavelet Positional Encoding (SWPE)

    This removes signal-awareness from DyWPE while keeping the multi-scale framework.
    Instead of using actual signal coefficients, it uses fixed learnable parameters
    for each scale. This tests whether signal-awareness is truly necessary.

    Key difference: P = f(θ) instead of P = f(x, θ)
    """

    def __init__(self, d_model, wavelet='bior2.2', max_len=5000, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.wavelet = wavelet
        self.max_level = min(3, int(np.log2(max_len)) - 1) if max_len > 8 else 1
        self.dropout = nn.Dropout(p=dropout)

        # Same scale embeddings as DyWPE
        num_scales = self.max_level + 1
        self.scale_embeddings = nn.Parameter(torch.randn(num_scales, d_model))

        # CRITICAL DIFFERENCE: Fixed learnable coefficients instead of signal-derived ones
        # These replace the dynamic wavelet coefficients with static learned patterns
        approx_length = max_len // (2 ** self.max_level)
        self.static_approx_coeffs = nn.Parameter(torch.randn(1, approx_length))

        self.static_detail_coeffs = nn.ParameterList([
            nn.Parameter(torch.randn(1, max_len // (2 ** (level + 1))))
            for level in range(self.max_level)
        ])

        # Same gating mechanism as DyWPE
        self.gate_w_g = nn.Linear(d_model, d_model)
        self.gate_w_v = nn.Linear(d_model, d_model)

        # Initialize wavelet filters (same as DyWPE for fair comparison)
        self._initialize_wavelet_filters(wavelet)

    def _initialize_wavelet_filters(self, wavelet):
        """Same wavelet filters as DyWPE for consistency."""
        # [Same implementation as WaveletDyWPE - shortened for brevity]
        wavelet_filters = {
            'bior2.2': ([0.35355339059327373, 0.7071067811865476, 0.35355339059327373, 0.0],
                       [0.0, -0.35355339059327373, 0.7071067811865476, -0.35355339059327373]),
            'db4': ([0.23037781330885523, 0.7148465705525415, 0.6308807679295904, -0.02798376941698385,
                    -0.18703481171888114, 0.030841381835986965, 0.032883011666982945, -0.010597401784997278],
                   [-0.010597401784997278, -0.032883011666982945, 0.030841381835986965, 0.18703481171888114,
                    -0.02798376941698385, -0.6308807679295904, 0.7148465705525415, -0.23037781330885523]),
            # Add other wavelets as needed
        }

        if wavelet in wavelet_filters:
            low_filter, high_filter = wavelet_filters[wavelet]
        else:
            low_filter, high_filter = wavelet_filters['bior2.2']

        self.register_buffer('lowpass_filter', torch.tensor(low_filter, dtype=torch.float32))
        self.register_buffer('highpass_filter', torch.tensor(high_filter, dtype=torch.float32))

    def _gated_modulation(self, scale_embedding, coeffs):
        """Same gating mechanism as DyWPE."""
        batch_size, seq_len = coeffs.shape

        gate_g = torch.sigmoid(self.gate_w_g(scale_embedding))
        gate_v = torch.tanh(self.gate_w_v(scale_embedding))

        combined_gate = gate_g * gate_v
        modulated_coeffs = coeffs.unsqueeze(-1) * combined_gate.unsqueeze(0).unsqueeze(0)

        return modulated_coeffs

    def _interpolate_to_length(self, tensor, target_length):
        """Same interpolation as DyWPE."""
        try:
            if tensor.size(1) == target_length:
                return tensor

            if tensor.size(1) == 0 or target_length == 0:
                return torch.zeros(tensor.size(0), target_length, tensor.size(2),
                                 device=tensor.device, dtype=tensor.dtype)

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

    def forward(self, x):
        """
        CRITICAL: This ignores the input signal content entirely!
        Uses only fixed learned coefficients.
        """
        B, L, _ = x.shape

        try:
            # ABLATION: Use static coefficients instead of signal-derived ones
            # Expand static coefficients to batch size
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

        except Exception as e:
            print(f"Static Wavelet PE failed: {e}")
            combined_encoding = torch.randn(B, L, self.d_model, device=x.device) * 0.01

        output = x + combined_encoding
        return self.dropout(output)


class SingleScaleDynamicPE(nn.Module):
    """
    ABLATION 2A: Single-Scale Dynamic PE (J=1 DyWPE)

    This keeps signal-awareness but removes multi-scale decomposition.
    Uses only one level of wavelet decomposition (J=1).
    Tests whether multi-scale representation is necessary.
    """

    def __init__(self, d_model, wavelet='bior2.2', max_len=5000, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.wavelet = wavelet
        self.max_level = 1  # ABLATION: Force single scale
        self.dropout = nn.Dropout(p=dropout)

        # Only 2 scales: approximation + 1 detail
        num_scales = 2
        self.scale_embeddings = nn.Parameter(torch.randn(num_scales, d_model))

        # Same gating as DyWPE
        self.gate_w_g = nn.Linear(d_model, d_model)
        self.gate_w_v = nn.Linear(d_model, d_model)

        self._initialize_wavelet_filters(wavelet)

    def _initialize_wavelet_filters(self, wavelet):
        """Same as StaticWaveletPE."""
        wavelet_filters = {
            'bior2.2': ([0.35355339059327373, 0.7071067811865476, 0.35355339059327373, 0.0],
                       [0.0, -0.35355339059327373, 0.7071067811865476, -0.35355339059327373]),
            'db4': ([0.23037781330885523, 0.7148465705525415, 0.6308807679295904, -0.02798376941698385,
                    -0.18703481171888114, 0.030841381835986965, 0.032883011666982945, -0.010597401784997278],
                   [-0.010597401784997278, -0.032883011666982945, 0.030841381835986965, 0.18703481171888114,
                    -0.02798376941698385, -0.6308807679295904, 0.7148465705525415, -0.23037781330885523]),
        }

        if wavelet in wavelet_filters:
            low_filter, high_filter = wavelet_filters[wavelet]
        else:
            low_filter, high_filter = wavelet_filters['bior2.2']

        self.register_buffer('lowpass_filter', torch.tensor(low_filter, dtype=torch.float32))
        self.register_buffer('highpass_filter', torch.tensor(high_filter, dtype=torch.float32))

    def _safe_dwt(self, x):
        """Same DWT as full DyWPE."""
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

    def _gated_modulation(self, scale_embedding, coeffs):
        """Same as DyWPE."""
        batch_size, seq_len = coeffs.shape

        gate_g = torch.sigmoid(self.gate_w_g(scale_embedding))
        gate_v = torch.tanh(self.gate_w_v(scale_embedding))

        combined_gate = gate_g * gate_v
        modulated_coeffs = coeffs.unsqueeze(-1) * combined_gate.unsqueeze(0).unsqueeze(0)

        return modulated_coeffs

    def _interpolate_to_length(self, tensor, target_length):
        """Same as DyWPE."""
        try:
            if tensor.size(1) == target_length:
                return tensor

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

    def forward(self, x):
        """Signal-aware but single-scale decomposition."""
        B, L, _ = x.shape

        try:
            # Signal-aware: use actual input signal
            x_mono = x.mean(dim=-1)  # (B, L)

            # ABLATION: Only single-level decomposition
            approx, detail = self._safe_dwt(x_mono)

            # Modulate with signal coefficients (keeping signal-awareness)
            approx_mod = self._gated_modulation(self.scale_embeddings[0], approx)
            detail_mod = self._gated_modulation(self.scale_embeddings[1], detail)

            # Combine scales
            combined_encoding = self._interpolate_to_length(approx_mod, L)
            detail_interpolated = self._interpolate_to_length(detail_mod, L)
            combined_encoding = combined_encoding + detail_interpolated

        except Exception as e:
            print(f"Single-Scale Dynamic PE failed: {e}")
            combined_encoding = torch.randn(B, L, self.d_model, device=x.device) * 0.01

        output = x + combined_encoding
        return self.dropout(output)


class GatedConvolutionalPE(nn.Module):
    """
    ABLATION 2B: Gated Convolutional PE

    Alternative single-scale approach using learnable convolutional filters
    instead of fixed wavelets. Still signal-aware but bypasses DWT entirely.
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1, kernel_size=7):
        super().__init__()

        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        # Learnable convolutional filter (replaces fixed wavelet)
        self.conv_filter = nn.Conv1d(1, d_model, kernel_size=kernel_size,
                                   padding=kernel_size//2, bias=False)

        # Gating mechanism (similar to DyWPE)
        self.gate_w_g = nn.Linear(d_model, d_model)
        self.gate_w_v = nn.Linear(d_model, d_model)

        # Global embedding for modulation
        self.global_embedding = nn.Parameter(torch.randn(d_model))

    def forward(self, x):
        """Signal-aware single-scale approach with learned filters."""
        B, L, _ = x.shape

        try:
            # Signal-aware: use actual input signal
            x_mono = x.mean(dim=-1).unsqueeze(1)  # (B, 1, L)

            # Apply learned convolutional filter
            conv_features = self.conv_filter(x_mono)  # (B, d_model, L)
            conv_features = conv_features.permute(0, 2, 1)  # (B, L, d_model)

            # Gated modulation with global embedding
            gate_g = torch.sigmoid(self.gate_w_g(self.global_embedding))
            gate_v = torch.tanh(self.gate_w_v(self.global_embedding))
            combined_gate = gate_g * gate_v

            # Apply gating to conv features
            gated_features = conv_features * combined_gate.unsqueeze(0).unsqueeze(0)

        except Exception as e:
            print(f"Gated Convolutional PE failed: {e}")
            gated_features = torch.randn(B, L, self.d_model, device=x.device) * 0.01

        output = x + gated_features
        return self.dropout(output)


# Import the original DyWPE for comparison
# from your_previous_code import WaveletDyWPE  # Your original implementation


class AblationTransformer(nn.Module):
    """
    Modified transformer that can use any of the PE variants for ablation studies.
    """

    def __init__(self, input_timesteps, in_channels, patch_size, embedding_dim,
                 num_transformer_layers=4, num_heads=4, dim_feedforward=128,
                 dropout=0.2, num_classes=14, pe_type='dywpe', wavelet='bior2.2'):
        super().__init__()

        self.pe_type = pe_type

        # Create patch embedding with specified PE type
        self.patch_embedding = self._create_patch_embedding_with_pe(
            in_channels, patch_size, embedding_dim, input_timesteps, pe_type, wavelet
        )

        # Standard transformer components
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads,
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers, num_layers=num_transformer_layers
        )

        self.ff_layer = nn.Linear(embedding_dim, dim_feedforward)
        self.classifier = nn.Linear(dim_feedforward, num_classes)

    def _create_patch_embedding_with_pe(self, in_channels, patch_size,
                                       embedding_dim, input_timesteps, pe_type, wavelet):
        """Create patch embedding with specified PE type."""

        class PatchEmbeddingWithAblationPE(nn.Module):
            def __init__(self, in_channels, patch_size, embedding_dim, input_timesteps, pe_type, wavelet):
                super().__init__()
                self.patch_size = patch_size
                self.embedding_dim = embedding_dim
                self.pe_type = pe_type

                # Patch processing
                self.num_patches = -(-input_timesteps // patch_size)
                self.padding = (self.num_patches * patch_size) - input_timesteps

                self.conv_layer = nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=embedding_dim,
                    kernel_size=patch_size,
                    stride=patch_size,
                )

                # Class token
                self.class_token_embeddings = nn.Parameter(
                    torch.randn((1, 1, embedding_dim), requires_grad=True)
                )

                # Select PE type
                max_len = self.num_patches + 1
                if pe_type == 'dywpe':
                    self.position_embeddings = WaveletDyWPE(
                        d_model=embedding_dim, wavelet=wavelet, max_len=max_len
                    )
                elif pe_type == 'static_wavelet':
                    self.position_embeddings = StaticWaveletPE(
                        d_model=embedding_dim, wavelet=wavelet, max_len=max_len
                    )
                elif pe_type == 'single_scale':
                    self.position_embeddings = SingleScaleDynamicPE(
                        d_model=embedding_dim, wavelet=wavelet, max_len=max_len
                    )
                elif pe_type == 'gated_conv':
                    self.position_embeddings = GatedConvolutionalPE(
                        d_model=embedding_dim, max_len=max_len
                    )
                else:
                    raise ValueError(f"Unknown PE type: {pe_type}")

            def forward(self, x):
                if self.padding > 0:
                    x = F.pad(x, (0, 0, 0, self.padding))

                x = x.permute(0, 2, 1)
                conv_output = self.conv_layer(x)
                conv_output = conv_output.permute(0, 2, 1)

                batch_size = x.shape[0]
                class_tokens = self.class_token_embeddings.expand(batch_size, -1, -1)
                output = torch.cat((class_tokens, conv_output), dim=1)
                output = self.position_embeddings(output)

                return output

        return PatchEmbeddingWithAblationPE(in_channels, patch_size, embedding_dim,
                                          input_timesteps, pe_type, wavelet)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)
        class_token_output = x[:, 0, :]
        x = self.ff_layer(class_token_output)
        output = self.classifier(x)
        return output


class CoreAblationStudy:
    """
    Framework to run the two core ablation studies.
    """

    def __init__(self, results_dir: str = "core_ablation_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # Define ablation experiments
        self.experiments = {
            # Ablation 1: Signal-Awareness
            'dywpe': {'pe_type': 'dywpe', 'name': 'DyWPE (Full)', 'category': 'Signal-Awareness'},
            'static_wavelet': {'pe_type': 'static_wavelet', 'name': 'Static Wavelet PE', 'category': 'Signal-Awareness'},

            # Ablation 2: Multi-Scale
            'single_scale': {'pe_type': 'single_scale', 'name': 'Single-Scale DyWPE', 'category': 'Multi-Scale'},
            'gated_conv': {'pe_type': 'gated_conv', 'name': 'Gated Conv PE', 'category': 'Multi-Scale'},
        }

    def run_single_experiment(self, experiment_name: str, model_params: Dict,
                             train_loader, valid_loader, test_loader,
                             num_epochs: int = 50, device: str = 'cuda') -> Dict:
        """Run a single ablation experiment."""

        config = self.experiments[experiment_name]
        print(f"\nTesting {config['name']} ({config['category']} Ablation)")
        print("-" * 60)

        try:
            # Create model with specified PE type
            model = AblationTransformer(
                pe_type=config['pe_type'],
                wavelet='bior2.2',  # Use best wavelet from previous study
                **model_params
            ).to(device)

            # Standard training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', patience=10, factor=0.5
            )

            # Training loop
            train_losses, val_losses = [], []
            train_accs, val_accs = [], []

            best_val_acc = 0.0
            best_epoch = 0
            epochs_without_improvement = 0

            start_time = time.time()

            for epoch in range(num_epochs):
                # Training phase
                model.train()
                train_loss, train_correct, total_train = 0, 0, 0

                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    train_loss += loss.item()
                    train_correct += (outputs.argmax(1) == labels).sum().item()
                    total_train += labels.size(0)

                # Validation phase
                model.eval()
                val_loss, val_correct, total_val = 0, 0, 0

                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        val_loss += loss.item()
                        val_correct += (outputs.argmax(1) == labels).sum().item()
                        total_val += labels.size(0)

                # Calculate metrics
                train_acc = train_correct / total_train
                val_acc = val_correct / total_val
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(valid_loader)

                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
                train_accs.append(train_acc)
                val_accs.append(val_acc)

                scheduler.step(val_acc)

                # Track best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    best_model_state = model.state_dict().copy()
                else:
                    epochs_without_improvement += 1

                # Print progress
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1:3d}: Train Acc: {train_acc:.4f}, "
                          f"Val Acc: {val_acc:.4f}, Best: {best_val_acc:.4f}")

                # Early stopping
                if epochs_without_improvement >= 15:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            training_time = time.time() - start_time

            # Test evaluation
            test_acc = None
            if test_loader is not None:
                model.load_state_dict(best_model_state)
                model.eval()

                test_correct, total_test = 0, 0
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        test_correct += (outputs.argmax(1) == labels).sum().item()
                        total_test += labels.size(0)

                test_acc = test_correct / total_test

            test_acc_str = f"{test_acc:.4f}" if test_acc is not None else "N/A"
            print(f"Results: Val Acc: {best_val_acc:.4f}, Test Acc: {test_acc_str}, Time: {training_time:.1f}s")

            return {
                'experiment': experiment_name,
                'name': config['name'],
                'category': config['category'],
                'pe_type': config['pe_type'],
                'best_val_acc': best_val_acc,
                'test_acc': test_acc,
                'training_time': training_time,
                'convergence_epoch': best_epoch,
                'stability': np.std(val_accs[-10:]) if len(val_accs) >= 10 else np.std(val_accs),
                'val_accs': val_accs,
                'train_losses': train_losses,
                'success': True
            }

        except Exception as e:
            print(f"ERROR with {experiment_name}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'experiment': experiment_name,
                'name': config['name'],
                'category': config['category'],
                'success': False,
                'error': str(e)
            }

    def run_complete_ablation(self, dataset_name: str, model_params: Dict,
                             train_loader, valid_loader, test_loader,
                             num_epochs: int = 50, device: str = 'cuda') -> Dict:
        """Run both core ablation studies."""

        print(f"Core DyWPE Ablation Studies on {dataset_name}")
        print("=" * 80)
        print("Testing the two fundamental claims of DyWPE:")
        print("1. Signal-awareness is crucial (DyWPE vs Static Wavelet PE)")
        print("2. Multi-scale decomposition is necessary (Multi-scale vs Single-scale)")
        print("=" * 80)

        results = []

        for exp_name in self.experiments.keys():
            result = self.run_single_experiment(
                exp_name, model_params,
                train_loader, valid_loader, test_loader,
                num_epochs, device
            )
            results.append(result)

        # Analyze results
        self.analyze_core_results(results, dataset_name)

        return {'results': results, 'dataset': dataset_name}

    def analyze_core_results(self, results: List[Dict], dataset_name: str):
        """Analyze and present core ablation results."""

        successful_results = [r for r in results if r.get('success', False)]

        if not successful_results:
            print("No successful experiments to analyze!")
            return

        df = pd.DataFrame(successful_results)

        print(f"\n{'='*80}")
        print("CORE ABLATION STUDY RESULTS")
        print(f"{'='*80}")

        # Separate by ablation category
        signal_awareness_results = df[df['category'] == 'Signal-Awareness'].copy()
        multi_scale_results = df[df['category'] == 'Multi-Scale'].copy()

        # ABLATION 1 ANALYSIS: Signal-Awareness
        print("\nABLATION 1: NECESSITY OF SIGNAL-AWARENESS")
        print("-" * 50)

        if len(signal_awareness_results) >= 2:
            dywpe_result = signal_awareness_results[signal_awareness_results['pe_type'] == 'dywpe'].iloc[0]
            static_result = signal_awareness_results[signal_awareness_results['pe_type'] == 'static_wavelet'].iloc[0]

            signal_awareness_gain = dywpe_result['best_val_acc'] - static_result['best_val_acc']
            signal_awareness_gain_pct = (signal_awareness_gain / static_result['best_val_acc']) * 100

            print(f"DyWPE (Signal-Aware):     {dywpe_result['best_val_acc']:.4f}")
            print(f"Static Wavelet PE:        {static_result['best_val_acc']:.4f}")
            print(f"Signal-Awareness Gain:    {signal_awareness_gain:+.4f} ({signal_awareness_gain_pct:+.2f}%)")

            # Test set comparison if available
            if dywpe_result.get('test_acc') is not None and static_result.get('test_acc') is not None:
                test_gain = dywpe_result['test_acc'] - static_result['test_acc']
                test_gain_pct = (test_gain / static_result['test_acc']) * 100
                print(f"Test Set - DyWPE:         {dywpe_result['test_acc']:.4f}")
                print(f"Test Set - Static:        {static_result['test_acc']:.4f}")
                print(f"Test Signal-Aware Gain:   {test_gain:+.4f} ({test_gain_pct:+.2f}%)")

            # Interpretation
            if signal_awareness_gain > 0.01:  # >1% improvement
                print("\n🎯 CONCLUSION: Signal-awareness provides SIGNIFICANT benefit")
                print("   The dynamic modulation with actual signal coefficients is crucial.")
            elif signal_awareness_gain > 0.005:  # >0.5% improvement
                print("\n✅ CONCLUSION: Signal-awareness provides MODERATE benefit")
                print("   The signal-aware approach outperforms the static version.")
            else:
                print("\n❌ CONCLUSION: Signal-awareness shows LIMITED benefit")
                print("   The multi-scale framework may be the primary contributor.")

        # ABLATION 2 ANALYSIS: Multi-Scale Representation
        print("\n\nABLATION 2: IMPORTANCE OF MULTI-SCALE REPRESENTATION")
        print("-" * 60)

        # Get DyWPE result for comparison
        if len(signal_awareness_results) > 0:
            dywpe_baseline = signal_awareness_results[signal_awareness_results['pe_type'] == 'dywpe'].iloc[0]

            if len(multi_scale_results) > 0:
                single_scale_result = multi_scale_results[multi_scale_results['pe_type'] == 'single_scale'].iloc[0]

                multi_scale_gain = dywpe_baseline['best_val_acc'] - single_scale_result['best_val_acc']
                multi_scale_gain_pct = (multi_scale_gain / single_scale_result['best_val_acc']) * 100

                print(f"DyWPE (Multi-Scale):      {dywpe_baseline['best_val_acc']:.4f}")
                print(f"Single-Scale DyWPE:       {single_scale_result['best_val_acc']:.4f}")
                print(f"Multi-Scale Gain:         {multi_scale_gain:+.4f} ({multi_scale_gain_pct:+.2f}%)")

                # Test set comparison
                if (dywpe_baseline.get('test_acc') is not None and
                    single_scale_result.get('test_acc') is not None):
                    test_gain = dywpe_baseline['test_acc'] - single_scale_result['test_acc']
                    test_gain_pct = (test_gain / single_scale_result['test_acc']) * 100
                    print(f"Test Set - Multi-Scale:   {dywpe_baseline['test_acc']:.4f}")
                    print(f"Test Set - Single-Scale:  {single_scale_result['test_acc']:.4f}")
                    print(f"Test Multi-Scale Gain:    {test_gain:+.4f} ({test_gain_pct:+.2f}%)")

                # Gated Conv comparison if available
                gated_conv_results = multi_scale_results[multi_scale_results['pe_type'] == 'gated_conv']
                if len(gated_conv_results) > 0:
                    gated_conv_result = gated_conv_results.iloc[0]
                    conv_gain = dywpe_baseline['best_val_acc'] - gated_conv_result['best_val_acc']
                    conv_gain_pct = (conv_gain / gated_conv_result['best_val_acc']) * 100

                    print(f"Gated Conv PE:            {gated_conv_result['best_val_acc']:.4f}")
                    print(f"DyWPE vs Conv Gain:       {conv_gain:+.4f} ({conv_gain_pct:+.2f}%)")

                # Interpretation
                if multi_scale_gain > 0.01:  # >1% improvement
                    print("\n🎯 CONCLUSION: Multi-scale decomposition is ESSENTIAL")
                    print("   The hierarchical wavelet representation provides significant value.")
                elif multi_scale_gain > 0.005:  # >0.5% improvement
                    print("\n✅ CONCLUSION: Multi-scale decomposition is BENEFICIAL")
                    print("   The multi-level DWT outperforms single-scale approaches.")
                else:
                    print("\n❓ CONCLUSION: Multi-scale benefit is UNCLEAR")
                    print("   Single-scale approaches may be sufficient for this dataset.")

        # OVERALL SUMMARY TABLE
        print(f"\n{'='*80}")
        print("COMPLETE RESULTS SUMMARY")
        print(f"{'='*80}")

        # Sort by validation accuracy
        df_sorted = df.sort_values('best_val_acc', ascending=False)

        print(f"{'Method':<20} {'Category':<15} {'Val Acc':<8} {'Test Acc':<8} {'Time(s)':<8}")
        print("-" * 70)

        for _, row in df_sorted.iterrows():
            test_acc_str = f"{row.get('test_acc', 0):.4f}" if row.get('test_acc') is not None else "N/A"
            print(f"{row['name']:<20} {row['category']:<15} {row['best_val_acc']:<8.4f} "
                  f"{test_acc_str:<8} {row['training_time']:<8.1f}")

        # KEY INSIGHTS
        print(f"\n{'='*80}")
        print("KEY INSIGHTS FOR PAPER")
        print(f"{'='*80}")

        best_method = df_sorted.iloc[0]
        print(f"🏆 Best Overall: {best_method['name']} ({best_method['best_val_acc']:.4f})")

        if len(signal_awareness_results) >= 2:
            dywpe_vs_static = signal_awareness_results.iloc[0]['best_val_acc'] - signal_awareness_results.iloc[1]['best_val_acc']
            if abs(dywpe_vs_static) < 0.005:
                print("⚠️  CRITICAL: Signal-awareness shows minimal improvement!")
                print("   Consider investigating why static approach performs similarly.")
            else:
                print(f"✅ Signal-awareness validated: {abs(dywpe_vs_static):.4f} improvement")

        if len(multi_scale_results) >= 1 and len(signal_awareness_results) >= 1:
            dywpe_acc = signal_awareness_results[signal_awareness_results['pe_type'] == 'dywpe'].iloc[0]['best_val_acc']
            single_scale_acc = multi_scale_results[multi_scale_results['pe_type'] == 'single_scale'].iloc[0]['best_val_acc']
            multi_vs_single = dywpe_acc - single_scale_acc

            if abs(multi_vs_single) < 0.005:
                print("⚠️  CRITICAL: Multi-scale shows minimal improvement!")
                print("   Consider if single-scale is sufficient for your datasets.")
            else:
                print(f"✅ Multi-scale validated: {abs(multi_vs_single):.4f} improvement")

        # Save results
        csv_path = self.results_dir / f"{dataset_name}_core_ablation_results.csv"
        df_sorted.to_csv(csv_path, index=False)

        # Save detailed analysis
        analysis_path = self.results_dir / f"{dataset_name}_ablation_analysis.json"
        analysis_data = {
            'dataset': dataset_name,
            'signal_awareness_analysis': {
                'dywpe_acc': signal_awareness_results[signal_awareness_results['pe_type'] == 'dywpe'].iloc[0]['best_val_acc'] if len(signal_awareness_results) > 0 else None,
                'static_acc': signal_awareness_results[signal_awareness_results['pe_type'] == 'static_wavelet'].iloc[0]['best_val_acc'] if len(signal_awareness_results) > 1 else None,
                'gain': signal_awareness_gain if 'signal_awareness_gain' in locals() else None,
                'gain_percent': signal_awareness_gain_pct if 'signal_awareness_gain_pct' in locals() else None
            },
            'multi_scale_analysis': {
                'multi_scale_acc': dywpe_baseline['best_val_acc'] if 'dywpe_baseline' in locals() else None,
                'single_scale_acc': single_scale_result['best_val_acc'] if 'single_scale_result' in locals() else None,
                'gain': multi_scale_gain if 'multi_scale_gain' in locals() else None,
                'gain_percent': multi_scale_gain_pct if 'multi_scale_gain_pct' in locals() else None
            },
            'best_method': best_method['name'],
            'best_accuracy': best_method['best_val_acc']
        }

        with open(analysis_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)

        print(f"\nResults saved:")
        print(f"  CSV: {csv_path}")
        print(f"  Analysis: {analysis_path}")


# Main function to run core ablation studies
def run_core_ablation_studies(dataset_name: str, model_params: Dict,
                             train_loader, valid_loader, test_loader,
                             num_epochs: int = 50) -> Dict:
    """
    Run the two core ablation studies for DyWPE.

    This tests the fundamental claims:
    1. Signal-awareness is crucial for performance
    2. Multi-scale representation is necessary

    Usage:
    results = run_core_ablation_studies(
        'LSST',
        model_params,
        train_loader_fixed,
        valid_loader_fixed,
        test_loader_fixed,
        num_epochs=50
    )
    """

    study = CoreAblationStudy(results_dir=f"core_ablation_{dataset_name}")

    results = study.run_complete_ablation(
        dataset_name=dataset_name,
        model_params=model_params,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    return results


# Example usage
if __name__ == "__main__":
    """
    Example of how to run the core ablation studies.
    """

    # Your model parameters
    model_params = {
        'input_timesteps': 36,
        'in_channels': 6,
        'patch_size': 8,
        'embedding_dim': 32,
        'num_transformer_layers': 4,
        'num_heads': 4,
        'dim_feedforward': 128,
        'dropout': 0.2,
        'num_classes': 14
    }

    # Run core ablation studies
    results = run_core_ablation_studies(
        dataset_name='LSST',
        model_params=model_params,
        train_loader=None, 
        valid_loader=None, 
        test_loader=None,   
        num_epochs=50
    )

    print("Core ablation studies completed!")