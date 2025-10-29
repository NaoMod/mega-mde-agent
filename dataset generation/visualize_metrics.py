#!/usr/bin/env python3
"""
Dataset Metrics Visualization

This script creates visually appealing graphs from diversity metrics CSV files.
It visualizes the comparison between seed and generated datasets for both
single-tool and multi-tool scenarios.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import matplotlib.ticker as mtick

# Set the style
plt.style.use('ggplot')
sns.set_palette('viridis')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Output directory
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    """Load the CSV data files"""
    single_tool_path = OUTPUT_DIR / "single_openRewrite_tool_comparison.csv"
    multi_tool_path = OUTPUT_DIR / "multi_openRewrite_tool_comparison.csv"
    
    single_df = pd.read_csv(single_tool_path)
    multi_df = pd.read_csv(multi_tool_path)
    
    return single_df, multi_df

# Radar chart creation function removed as requested
    
def create_bar_comparison(single_df, multi_df):
    """Create bar charts comparing key metrics between datasets"""
    metrics = ["Distance", "Dispersion", "Isocontour Radius"]
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Seed vs. Generated Dataset Comparison", fontsize=16, fontweight='bold')
    
    for i, metric in enumerate(metrics):
        # Extract data
        single_seed = single_df[single_df["Metric"] == metric]["Seed Dataset"].values[0]
        single_gen = single_df[single_df["Metric"] == metric]["Generated Dataset"].values[0]
        multi_seed = multi_df[multi_df["Metric"] == metric]["Seed Dataset"].values[0]
        multi_gen = multi_df[multi_df["Metric"] == metric]["Generated Dataset"].values[0]
        
        # Data for plotting
        datasets = ['Single-Tool\nSeed', 'Single-Tool\nGenerated', 'Multi-Tool\nSeed', 'Multi-Tool\nGenerated']
        values = [single_seed, single_gen, multi_seed, multi_gen]
        colors = ['blue', 'red', 'blue', 'red']
        
        # Create bars with individual alpha values and hatches
        bars = []
        hatches = ['', '', '///', '///']  # Define hatches
        for j, (dataset, value, color, hatch) in enumerate(zip(datasets, values, colors, hatches)):
            alpha = 0.4 if 'Multi-Tool' in dataset else 0.7
            bar = axs[i].bar(dataset, value, color=color, alpha=alpha)
            bars.append(bar[0])  # Store bar object
        
        # Add hatching to multi-tool bars
        for j, bar in enumerate(bars):
            bar.set_hatch(hatches[j])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            axs[i].text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Customize
        axs[i].set_title(metric, fontweight='bold')
        axs[i].set_ylim(0, max(values) * 1.15)  # Add 15% padding
        axs[i].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add legend
    handles = [
        plt.Rectangle((0,0),1,1, color='blue', alpha=0.7, label='Seed Dataset - Single Tool'),
        plt.Rectangle((0,0),1,1, color='red', alpha=0.7, label='Generated Dataset - Single Tool'),
        plt.Rectangle((0,0),1,1, color='blue', alpha=0.4, hatch='///', label='Seed Dataset - Multi Tool'),
        plt.Rectangle((0,0),1,1, color='red', alpha=0.4, hatch='///', label='Generated Dataset - Multi Tool')
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.05), 
              ncol=2, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the legend
    plt.savefig(OUTPUT_DIR / "openRewrite_bar_comparison.png", dpi=300, bbox_inches='tight')

def create_lexical_diversity_chart(single_df, multi_df):
    """Create a comparison of lexical diversity metrics"""
    metrics = ["Vocabulary Size", "Unique 3-grams"]
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle("Lexical Diversity Metrics", fontsize=16, fontweight='bold')
    
    # Process each metric
    for i, metric in enumerate(metrics):
        # Extract data
        single_seed = single_df[single_df["Metric"] == metric]["Seed Dataset"].values[0]
        single_gen = single_df[single_df["Metric"] == metric]["Generated Dataset"].values[0]
        multi_seed = multi_df[multi_df["Metric"] == metric]["Seed Dataset"].values[0]
        multi_gen = multi_df[multi_df["Metric"] == metric]["Generated Dataset"].values[0]
        
        # Calculate ratios
        single_ratio = single_gen / single_seed
        multi_ratio = multi_gen / multi_seed
        
        # Data for plotting
        categories = ['Single-Tool', 'Multi-Tool']
        seed_values = [single_seed, multi_seed]
        gen_values = [single_gen, multi_gen]
        
        # Set positions and width
        pos = np.arange(len(categories))
        bar_width = 0.35
        
        # Create bars
        axs[i].bar(pos - bar_width/2, seed_values, bar_width, label='Seed', color='blue', alpha=0.7)
        axs[i].bar(pos + bar_width/2, gen_values, bar_width, label='Generated', color='red', alpha=0.7)
        
        # Add value labels
        for j, val in enumerate(seed_values):
            axs[i].text(pos[j] - bar_width/2, val + max(gen_values)*0.02, 
                      f'{int(val)}', ha='center', va='bottom', fontsize=9)
        
        for j, val in enumerate(gen_values):
            axs[i].text(pos[j] + bar_width/2, val + max(gen_values)*0.02, 
                      f'{int(val)}', ha='center', va='bottom', fontsize=9)
            
            # Add ratio on top
            ratio = gen_values[j] / seed_values[j]
            axs[i].text(pos[j], max(gen_values)*0.1, 
                      f'×{ratio:.1f}', ha='center', va='bottom', 
                      fontsize=12, fontweight='bold', color='darkgreen')
        
        # Customize
        axs[i].set_title(metric, fontweight='bold')
        axs[i].set_xticks(pos)
        axs[i].set_xticklabels(categories)
        axs[i].set_ylim(0, max(gen_values) * 1.2)  # Add 20% padding
        axs[i].grid(axis='y', linestyle='--', alpha=0.7)
        axs[i].legend()
        
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "openRewrite_lexical_diversity.png", dpi=300, bbox_inches='tight')

# Affinity gauge function removed as requested

def create_summary_dashboard():
    """Create a combined dashboard with all visualizations"""
    # Load data
    single_df, multi_df = load_data()
    
    # Create individual visualizations
    create_bar_comparison(single_df, multi_df)
    create_lexical_diversity_chart(single_df, multi_df)
    
    print("Visualizations created successfully!")
    print(f"Output saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    create_summary_dashboard()