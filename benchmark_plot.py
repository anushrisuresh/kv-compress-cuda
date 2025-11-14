#!/usr/bin/env python3
"""
Benchmark script for dense vs streaming attention.
Runs attention_compare with different sequence lengths and plots results.
"""

import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def run_benchmark(T, window_size=64, sink_size=16):
    """Run attention_compare and parse output."""
    cmd = ['./attention_compare', str(T), str(window_size), str(sink_size)]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent
        )
        
        # Parse output for timing information
        output = result.stdout
        
        # Extract dense time: "Dense attention average time   : X.XX ms"
        dense_match = re.search(r'Dense attention average time\s*:\s*([\d.]+)\s*ms', output)
        # Extract streaming time: "Streaming attention avg time   : X.XX ms"
        stream_match = re.search(r'Streaming attention avg time\s*:\s*([\d.]+)\s*ms', output)
        # Extract speedup: "Speedup (dense / streaming)    : X.XXx"
        speedup_match = re.search(r'Speedup \(dense / streaming\)\s*:\s*([\d.]+)x', output)
        
        if dense_match and stream_match:
            dense_time = float(dense_match.group(1))
            stream_time = float(stream_match.group(1))
            speedup = float(speedup_match.group(1)) if speedup_match else dense_time / stream_time
            
            return {
                'T': T,
                'dense_time': dense_time,
                'stream_time': stream_time,
                'speedup': speedup
            }
        else:
            print(f"Warning: Could not parse output for T={T}")
            print("Output:", output)
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"Error running benchmark for T={T}:")
        print(e.stderr)
        return None
    except FileNotFoundError:
        print("Error: attention_compare executable not found!")
        print("Please compile it first: nvcc -O3 -std=c++17 -arch=sm_80 --expt-relaxed-constexpr attention.cu -o attention_compare")
        return None

def benchmark_range(T_values, window_size=64, sink_size=16):
    """Run benchmarks for a range of sequence lengths."""
    results = []
    
    print("Running benchmarks...")
    print("=" * 60)
    
    for T in T_values:
        print(f"Testing T={T}...", end=" ", flush=True)
        result = run_benchmark(T, window_size, sink_size)
        if result:
            results.append(result)
            print(f"✓ Dense: {result['dense_time']:.3f}ms, "
                  f"Streaming: {result['stream_time']:.3f}ms, "
                  f"Speedup: {result['speedup']:.2f}x")
        else:
            print("✗ Failed")
    
    print("=" * 60)
    return results

def plot_results(results, window_size=64, sink_size=16, save_path='attention_benchmark.png'):
    """Create a pretty plot of the benchmark results."""
    if not results:
        print("No results to plot!")
        return
    
    # Extract data
    T_values = [r['T'] for r in results]
    dense_times = [r['dense_time'] for r in results]
    stream_times = [r['stream_time'] for r in results]
    speedups = [r['speedup'] for r in results]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Runtime comparison
    ax1.plot(T_values, dense_times, 'o-', label='Dense Attention', 
             linewidth=2.5, markersize=8, color='#1f77b4')
    ax1.plot(T_values, stream_times, 's-', label=f'Streaming (window={window_size}, sink={sink_size})', 
             linewidth=2.5, markersize=8, color='#ff7f0e')
    
    ax1.set_xlabel('Sequence Length (T)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Runtime (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Attention Runtime vs Sequence Length', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    
    # Format x-axis to show actual T values
    ax1.set_xticks(T_values)
    ax1.set_xticklabels([str(t) for t in T_values], rotation=45, ha='right')
    
    # Plot 2: Speedup
    ax2.plot(T_values, speedups, 'o-', label='Speedup (Dense / Streaming)', 
             linewidth=2.5, markersize=8, color='#2ca02c')
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    ax2.set_xlabel('Sequence Length (T)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Speedup (×)', fontsize=12, fontweight='bold')
    ax2.set_title('Streaming Speedup vs Dense Attention', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xscale('log', base=2)
    
    # Format x-axis
    ax2.set_xticks(T_values)
    ax2.set_xticklabels([str(t) for t in T_values], rotation=45, ha='right')
    
    # Add text annotation with max speedup
    max_speedup_idx = np.argmax(speedups)
    max_speedup = speedups[max_speedup_idx]
    max_T = T_values[max_speedup_idx]
    ax2.annotate(f'Max: {max_speedup:.2f}× at T={max_T}',
                xy=(max_T, max_speedup),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    
    # Also print a summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'T':<8} {'Dense (ms)':<12} {'Streaming (ms)':<15} {'Speedup':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['T']:<8} {r['dense_time']:<12.3f} {r['stream_time']:<15.3f} {r['speedup']:<10.2f}x")
    print("=" * 60)

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Benchmark dense vs streaming attention and plot results'
    )
    parser.add_argument('--T-min', type=int, default=128, 
                       help='Minimum sequence length (default: 128)')
    parser.add_argument('--T-max', type=int, default=4096,
                       help='Maximum sequence length (default: 4096)')
    parser.add_argument('--T-step', type=int, default=2,
                       help='Sequence length step (multiplier, default: 2)')
    parser.add_argument('--window-size', type=int, default=64,
                       help='Sliding window size (default: 64)')
    parser.add_argument('--sink-size', type=int, default=16,
                       help='Sink token count (default: 16)')
    parser.add_argument('--output', type=str, default='attention_benchmark.png',
                       help='Output plot filename (default: attention_benchmark.png)')
    
    args = parser.parse_args()
    
    # Generate T values (powers of 2)
    T_values = []
    T = args.T_min
    while T <= args.T_max:
        T_values.append(T)
        T *= args.T_step
    
    print(f"Benchmarking sequence lengths: {T_values}")
    print(f"Window size: {args.window_size}, Sink size: {args.sink_size}")
    print()
    
    # Run benchmarks
    results = benchmark_range(T_values, args.window_size, args.sink_size)
    
    if results:
        # Create plot
        plot_results(results, args.window_size, args.sink_size, args.output)
    else:
        print("No successful benchmarks. Exiting.")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())

