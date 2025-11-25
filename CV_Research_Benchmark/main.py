import matplotlib
matplotlib.use('Agg') # Prevent GUI errors on servers
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from src.config import Config
from src.utils import set_seed
from src.models import (
    VGG11_Gated, VGG11_Baseline, VGG13_Gated, VGG13_Baseline,
    ResNet18_Gated, ResNet18_Baseline, ResNet34_Gated, ResNet34_Baseline,
    GatedConv2d
)
from src.train import train_engine

# Results container for the final table
BENCHMARK_RESULTS = []

def save_plot(base_hist, gated_hist, name):
    if not os.path.exists('results'): os.makedirs('results')
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(base_hist['acc'], label='Baseline')
    plt.plot(gated_hist['acc'], label='Gated')
    plt.title(f'{name}: Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(gated_hist['sparsity'], color='green', label='Density')
    plt.title(f'{name}: Density')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'results/{name}.png')
    plt.close()

def run_pair(model_base, model_gated, model_name, dataset):
    # 1. Baseline
    base_hist, _ = train_engine(model_base, dataset, f"{model_name}-Base")
    final_base_acc = base_hist['acc'][-1]
    
    # 2. Gated
    gated_hist, _ = train_engine(model_gated, dataset, f"{model_name}-Gated")
    final_gated_acc = gated_hist['acc'][-1]
    final_density = gated_hist['sparsity'][-1]
    
    # 3. Save Visualization
    save_plot(base_hist, gated_hist, f"{model_name}_{dataset}")
    
    # 4. Log Stats
    BENCHMARK_RESULTS.append({
        "Model": model_name,
        "Dataset": dataset,
        "Base Acc": final_base_acc,
        "Gated Acc": final_gated_acc,
        "Reduction": (1 - final_density) * 100
    })

def print_table():
    print("\n" + "="*75)
    print(f"{'LATENCY-AWARE GATING BENCHMARK RESULTS':^75}")
    print("="*75)
    print(f"{'Model':<15} | {'Dataset':<10} | {'Base Acc':<10} | {'Gated Acc':<10} | {'Pruned (%)':<10}")
    print("-" * 75)
    
    for row in BENCHMARK_RESULTS:
        print(f"{row['Model']:<15} | {row['Dataset']:<10} | {row['Base Acc']:<9.2f}% | {row['Gated Acc']:<9.2f}% | {row['Reduction']:<9.1f}%")
    print("-" * 75)
    print("Visualization plots saved in 'results/' folder.")
    print("="*75 + "\n")

def main():
    set_seed(42)
    
    # --- DEFINING THE BENCHMARK SUITE ---
    # Add/Remove lines here to customize the suite
    tasks = [
        # (ModelName, Dataset, NumClasses, BaseClass, GatedClass)
        ('VGG11',    'CIFAR100', 100, VGG11_Baseline,    VGG11_Gated),
        ('ResNet18', 'CIFAR10',  10,  ResNet18_Baseline, ResNet18_Gated),
        ('ResNet34', 'SVHN',     10,  ResNet34_Baseline, ResNet34_Gated),
    ]
    
    print(f"Starting Benchmark Suite with {len(tasks)} tasks...")
    
    for name, data, nc, BaseClass, GatedClass in tasks:
        run_pair(BaseClass(nc), GatedClass(nc), name, data)
        
    print_table()

if __name__ == "__main__":
    main()