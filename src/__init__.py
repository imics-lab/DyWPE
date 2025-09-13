"""
DyWPE: Dynamic Wavelet Positional Encoding

A signal-aware positional encoding framework for time series transformers.
"""

# Core imports
from .core.dywpe import DyWPE
from .core.positional_encodings import tAPE, LearnedPositionalEncoding, FixedPositionalEncoding, ConvSPE, TemporalPositionalEncoding, RelativePositionalEncoding, RotaryPositionalEncoding

# Model imports
from .models.transformer import TimeSeriesTransformer, create_model_with_dywpe
from .models.embeddings import TimeSeriesPatchEmbeddingLayer

# Ablation study imports
from .ablation.benchmark import PositionalEncodingBenchmark
from .ablation.signal_awareness import StaticWaveletPE, run_signal_awareness_ablation
from .ablation.multiscale import SingleScaleDyWPE, GatedConvolutionalPE, run_multiscale_ablation

__all__ = [
    # Core components
    "scr",
    
    # Models
    "TimeSeriesTransformer",
    "create_model_with_dywpe", 
    "TimeSeriesPatchEmbeddingLayer",
    
    # Benchmark
    "PositionalEncodingBenchmark",
    
    # Ablation studies
    "StaticWaveletPE",
    "SingleScaleDyWPE",
    "GatedConvolutionalPE",
    "run_signal_awareness_ablation",
    "run_multiscale_ablation",
]