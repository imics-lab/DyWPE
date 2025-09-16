## Usage the ablation studies and benchmark

```python
from src.ablation.benchmark import PositionalEncodingBenchmark

benchmark = PositionalEncodingBenchmark()

model_params = {
    'input_timesteps': input_timesteps,
    'in_channels': in_channels,
    'patch_size': patch_size,
    'embedding_dim': 32,
    'num_transformer_layers': 4,
    'num_heads': 4,
    'dim_feedforward': 128,
    'dropout': 0.2,
    'num_classes': num_classes
}

# Run benchmark
models = benchmark.run_full_benchmark(
    model_params,
    train_loader,
    valid_loader,
    test_loader,
    encodings=['dywpe'],
    n_epochs=num_epochs
)

benchmark.print_summary()
benchmark.plot_comparison('results.png')

```


## Ablation Studies

This repository includes the two critical ablation studies that validate DyWPE's core contributions:

### 1. Signal-Awareness Study

**Question**: Is signal-awareness (`P = f(x, θ)`) better than static approaches (`P = f(θ)`)?

```python
from src.ablation.signal_awarness import run_signal_awareness_ablation

results = run_signal_awareness_ablation(
    dataset_name='dataset_name',
    model_params=model_params,
    pe_params=pe_params,
    train_loader=train_loader,
    valid_loader=valid_loader,
    test_loader=test_loader,
    num_epochs=num_epochs
)
```

**Compares**:
- **DyWPE**: Full signal-aware approach
- **Static Wavelet PE**: Same multi-scale framework, no signal dependency

### 2. Multi-Scale Study

**Question**: Is hierarchical wavelet decomposition essential?

```python
from src.ablation.multiscale import run_multiscale_ablation

results = run_multiscale_ablation(
    dataset_name='dataset_name',
    model_params=model_params,
    pe_params=pe_params,
    train_loader=train_loader,
    valid_loader=valid_loader,
    test_loader=test_loader,
    num_epochs=num_epochs
)
```

**Compares**:
- **DyWPE (Multi-Scale)**: Full hierarchical decomposition
- **Single-Scale DyWPE**: J=1 decomposition only
- **Gated Conv PE**: Alternative single-scale approach

### Run All Ablations

```bash
# Run both core ablation studies
python experiments/run_ablation_studies.py --study both --dataset data_name --epochs num_epochs

# Run signal-awareness study only
python experiments/run_ablation_studies.py --study signal_awareness --dataset data_name --epochs num_epochs

# Run multi-scale study only  
python experiments/run_ablation_studies.py --study multiscale --dataset data_name --epochs num_epochs
```

### Complete Example

See `scripts/complete_example.py` for a comprehensive tutorial that demonstrates:

- Basic DyWPE usage
- Running ablation studies
- Interpreting results

```bash
python scripts/complete_example.py
```



