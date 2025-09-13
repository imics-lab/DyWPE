"""
Positional Encoding Benchmark Framework
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Optional

from ..models.transformer import TimeSeriesTransformer



class PositionalEncodingBenchmark:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = {}

    def benchmark_encoding(self, encoding_type, model_params, train_loader, valid_loader,
                          test_loader=None, n_epochs=10, verbose=True):
        """Benchmark a single positional encoding method"""

        if verbose:
            print(f"\n{'='*50}")
            print(f"Benchmarking: {encoding_type.upper()}")
            print(f"{'='*50}")

        # Create model with specified positional encoding
        model = TimeSeriesTransformer(pos_encoding=encoding_type, **model_params).to(self.device)

        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Track metrics
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        epoch_times = []

        best_val_acc = 0.0
        best_model_state = None

        for epoch in range(n_epochs):
            epoch_start = time.time()

            # Training
            model.train()
            train_loss, train_correct, total_train = 0, 0, 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_correct += (outputs.argmax(1) == labels).sum().item()
                total_train += labels.size(0)

            # Validation
            model.eval()
            val_loss, val_correct, total_val = 0, 0, 0

            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    val_correct += (outputs.argmax(1) == labels).sum().item()
                    total_val += labels.size(0)

            # Calculate metrics
            epoch_time = time.time() - epoch_start
            train_acc = train_correct / total_train
            val_acc = val_correct / total_val
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(valid_loader)

            # Store metrics
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            epoch_times.append(epoch_time)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()

            if verbose and (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch+1:2d}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Time: {epoch_time:.2f}s")

        # Test set evaluation with best model
        test_acc = None
        test_loss = None
        if test_loader is not None and best_model_state is not None:
            # Load best model
            model.load_state_dict(best_model_state)
            model.eval()

            test_loss_sum, test_correct, total_test = 0, 0, 0

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    test_loss_sum += loss.item()
                    test_correct += (outputs.argmax(1) == labels).sum().item()
                    total_test += labels.size(0)

            test_acc = test_correct / total_test
            test_loss = test_loss_sum / len(test_loader)

            if verbose:
                print(f"Test Acc: {test_acc:.4f}")

        # Store results
        self.results[encoding_type] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'epoch_times': epoch_times,
            'best_val_acc': best_val_acc,
            'avg_epoch_time': np.mean(epoch_times),
            'final_train_acc': train_accs[-1],
            'final_val_acc': val_accs[-1],
            'test_acc': test_acc,
            'test_loss': test_loss
        }

        if verbose:
            print(f"Best Val Acc: {best_val_acc:.4f}")
            if test_acc is not None:
                print(f"Final Test Acc: {test_acc:.4f}")
            print(f"Avg Epoch Time: {np.mean(epoch_times):.2f}s")

        return model

    def run_full_benchmark(self, model_params, train_loader, valid_loader,
                          test_loader=None, encodings=None, n_epochs=10):
        """Run benchmark on multiple positional encoding methods"""

        if encodings is None:
            encodings = ['fixed', 'learned', 'rope', 'relative', 'tAPE',
                        'dywpe', 'eRPE', 'SPE', 'TUPE']

        print(f"Starting benchmark on {len(encodings)} positional encodings...")
        print(f"Epochs per encoding: {n_epochs}")
        if test_loader is not None:
            print("Test set evaluation: ENABLED")
        else:
            print("Test set evaluation: DISABLED (no test_loader provided)")

        models = {}
        for encoding in encodings:
            try:
                models[encoding] = self.benchmark_encoding(
                    encoding, model_params, train_loader, valid_loader, test_loader, n_epochs
                )
            except Exception as e:
                print(f"Error with {encoding}: {e}")
                continue

        return models

    def plot_comparison(self, save_path=None):
        """Plot comparison of different positional encodings"""
        if not self.results:
            print("No results to plot. Run benchmark first.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Validation Accuracy over epochs
        ax1 = axes[0, 0]
        for encoding, results in self.results.items():
            ax1.plot(results['val_accs'], label=encoding, marker='o', markersize=3)
        ax1.set_title('Validation Accuracy over Epochs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)

        # Plot 2: Training Loss over epochs
        ax2 = axes[0, 1]
        for encoding, results in self.results.items():
            ax2.plot(results['train_losses'], label=encoding, marker='s', markersize=3)
        ax2.set_title('Training Loss over Epochs')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        # Plot 3: Best validation accuracy comparison
        ax3 = axes[1, 0]
        encodings = list(self.results.keys())
        best_accs = [self.results[enc]['best_val_acc'] for enc in encodings]
        bars = ax3.bar(encodings, best_accs)
        ax3.set_title('Best Validation Accuracy by Encoding')
        ax3.set_ylabel('Accuracy')
        ax3.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, acc in zip(bars, best_accs):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{acc:.3f}', ha='center', va='bottom')

        # Plot 4: Average epoch time comparison
        ax4 = axes[1, 1]
        avg_times = [self.results[enc]['avg_epoch_time'] for enc in encodings]
        bars = ax4.bar(encodings, avg_times)
        ax4.set_title('Average Training Time per Epoch')
        ax4.set_ylabel('Time (seconds)')
        ax4.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, time_val in zip(bars, avg_times):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{time_val:.2f}s', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def print_summary(self):
        """Print summary of benchmark results"""
        if not self.results:
            print("No results to summarize.")
            return

        print(f"\n{'='*100}")
        print("POSITIONAL ENCODING BENCHMARK SUMMARY")
        print(f"{'='*100}")

        # Check if test results are available
        has_test_results = any(self.results[enc].get('test_acc') is not None for enc in self.results)

        # Create summary table headers
        if has_test_results:
            headers = ['Encoding', 'Best Val Acc', 'Final Val Acc', 'Test Acc', 'Avg Time/Epoch']
            print(f"{headers[0]:<15} {headers[1]:<12} {headers[2]:<12} {headers[3]:<10} {headers[4]:<15}")
            print("-" * 75)
        else:
            headers = ['Encoding', 'Best Val Acc', 'Final Val Acc', 'Avg Time/Epoch']
            print(f"{headers[0]:<12} {headers[1]:<12} {headers[2]:<12} {headers[3]:<15}")
            print("-" * 60)

        # Sort by best validation accuracy
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['best_val_acc'],
            reverse=True
        )

        for encoding, results in sorted_results:
            if has_test_results:
                test_acc = results.get('test_acc')
                test_acc_str = f"{test_acc:.4f}" if test_acc is not None else "N/A"
                print(f"{encoding:<15} {results['best_val_acc']:<12.4f} "
                      f"{results['final_val_acc']:<12.4f} {test_acc_str:<10} "
                      f"{results['avg_epoch_time']:<15.2f}")
            else:
                print(f"{encoding:<12} {results['best_val_acc']:<12.4f} "
                      f"{results['final_val_acc']:<12.4f} {results['avg_epoch_time']:<15.2f}")

        # Find best performer
        best_encoding = sorted_results[0][0]
        best_val_acc = sorted_results[0][1]['best_val_acc']
        best_test_acc = sorted_results[0][1].get('test_acc')

        print(f"\nBest performing encoding: {best_encoding.upper()} "
              f"with {best_val_acc:.4f} validation accuracy")

        if best_test_acc is not None:
            print(f"Test accuracy of best model: {best_test_acc:.4f}")

        # Performance analysis
        if has_test_results:
            print(f"\n{'='*50}")
            print("TEST SET PERFORMANCE ANALYSIS")
            print(f"{'='*50}")

            test_results = [(enc, res.get('test_acc', 0)) for enc, res in sorted_results if res.get('test_acc') is not None]
            test_results.sort(key=lambda x: x[1], reverse=True)

            print("Ranking by Test Accuracy:")
            for i, (enc, test_acc) in enumerate(test_results, 1):
                val_acc = self.results[enc]['best_val_acc']
                generalization_gap = val_acc - test_acc
                print(f"{i:2d}. {enc:<15} Test: {test_acc:.4f} | Val: {val_acc:.4f} | Gap: {generalization_gap:+.4f}")

            # Find best generalization
            best_gen = min(test_results, key=lambda x: abs(self.results[x[0]]['best_val_acc'] - x[1]))
            gen_gap = self.results[best_gen[0]]['best_val_acc'] - best_gen[1]
            print(f"\nBest generalization: {best_gen[0]} (gap: {gen_gap:+.4f})")