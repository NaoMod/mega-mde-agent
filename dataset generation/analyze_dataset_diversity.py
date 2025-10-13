#!/usr/bin/env python3
"""
Dataset Diversity Comparison Script

This script compares a small original dataset of seeds with a larger generated dataset
using diversity metrics from the paper "Diversity-oriented Data Augmentation with Large Language Models".

It calculates:
- Distance (average pairwise Euclidean distance)
- Dispersion (1 - average cosine similarity)
- Isocontour Radius (geometric mean of per-dimension standard deviations)
- Vocabulary Size (unique words)
- Unique 3-grams (distinct 3-word sequences)
- Affinity (similarity between mean embeddings of datasets)

The script processes seed instructions and generated instructions, embeds them using OpenAI,
and provides both numeric comparisons and visualizations.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import umap
from collections import Counter
import re
from openai import OpenAI
from typing import List, Dict, Tuple, Any, Union
from pathlib import Path

# Set up OpenAI client
# Try to load from .env file first, then fall back to environment variable
from dotenv import load_dotenv
from pathlib import Path

# Load .env file from project root directory
env_path = Path(__file__).resolve().parents[1] / '.env'
load_dotenv(dotenv_path=env_path)

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    # If still not found, raise error
    raise ValueError("OPENAI_API_KEY not found. Please add it to your .env file or environment variables.")

# Initialize the OpenAI client with the new API format
client = OpenAI(api_key=api_key)

EMBEDDING_MODEL = "text-embedding-3-small"
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_seed_instructions(seed_file_path: str) -> List[str]:
    """
    Extracts instruction text from a Python file containing seed definitions.
    
    Args:
        seed_file_path: Path to the Python file with seed instructions
        
    Returns:
        List of instruction text strings
    """
    with open(seed_file_path, 'r') as f:
        content = f.read()
    
    # Extract instructions using regex
    pattern = r'instruction="(.*?)",?\n'
    matches = re.findall(pattern, content, re.DOTALL)
    
    return matches

def load_generated_instructions(json_file_path: str) -> List[str]:
    """
    Loads generated instructions from a JSON file.
    
    Args:
        json_file_path: Path to the JSON file with generated instructions
        
    Returns:
        List of instruction text strings
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Extract instruction text from each item
    instructions = [item.get("instruction", "") for item in data if item.get("instruction")]
    
    return instructions

def get_embeddings(texts: List[str]) -> np.ndarray:
    """
    Get embeddings for a list of texts using OpenAI's API.
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        Numpy array of embeddings (shape: len(texts) x embedding_dim)
    """
    # Process in batches to avoid API limits
    batch_size = 100
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
    
    return np.array(all_embeddings)

def compute_distance(embeddings: np.ndarray) -> float:
    """
    Calculate average pairwise Euclidean distance between embeddings.
        
    """
    pairwise_distances = pdist(embeddings, metric='euclidean')
    return np.mean(pairwise_distances)

def compute_dispersion(embeddings: np.ndarray) -> float:
    """
    Calculate dispersion (1 - average cosine similarity).

    """
    # Normalize embeddings for cosine similarity
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
    similarities = cosine_similarity(normalized_embeddings)
    
    # Exclude self-similarities (diagonal elements)
    n = similarities.shape[0]
    mask = ~np.eye(n, dtype=bool)
    avg_similarity = similarities[mask].mean()
    
    return 1.0 - avg_similarity

def compute_isocontour_radius(embeddings: np.ndarray) -> float:
    """
    Calculate the isocontour radius (geometric mean of per-dimension standard deviations).
    
    Args:
        embeddings: Embedding vectors (shape: n_samples x embedding_dim)
        
    Returns:
        Isocontour radius
    """
    std_devs = np.std(embeddings, axis=0)
    # Filter out zeros to avoid issues with geometric mean
    std_devs = std_devs[std_devs > 0]
    
    if len(std_devs) == 0:
        return 0.0
        
    return np.exp(np.mean(np.log(std_devs)))



def tokenize_text(text: str) -> List[str]:
    """
    Simple tokenization function to split text into words.
    
    Args:
        text: Input text string
        
    Returns:
        List of tokens
    """
    # Convert to lowercase, remove punctuation, and split
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()

def compute_vocabulary_size(texts: List[str]) -> int:
    """
    Calculate the vocabulary size (number of unique words).
    
    Args:
        texts: List of text strings
        
    Returns:
        Number of unique words
    """
    all_tokens = []
    for text in texts:
        all_tokens.extend(tokenize_text(text))
    
    return len(set(all_tokens))

def compute_unique_ngrams(texts: List[str], n: int = 3) -> int:
    """
    Calculate the number of unique n-grams.
    
    Args:
        texts: List of text strings
        n: Size of n-grams (default: 3)
        
    Returns:
        Number of unique n-grams
    """
    all_ngrams = []
    for text in texts:
        tokens = tokenize_text(text)
        if len(tokens) >= n:
            ngrams = [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
            all_ngrams.extend(ngrams)
    
    return len(set(all_ngrams))

def compute_affinity(embeddings1: np.ndarray, embeddings2: np.ndarray) -> float:
    """
    Calculate affinity score (cosine similarity between mean embeddings).
    
    Args:
        embeddings1: First set of embeddings
        embeddings2: Second set of embeddings
        
    Returns:
        Affinity score
    """
    mean_embedding1 = np.mean(embeddings1, axis=0)
    mean_embedding2 = np.mean(embeddings2, axis=0)
    
    # Normalize for cosine similarity
    mean_embedding1 = mean_embedding1 / np.linalg.norm(mean_embedding1)
    mean_embedding2 = mean_embedding2 / np.linalg.norm(mean_embedding2)
    
    return np.dot(mean_embedding1, mean_embedding2)

def visualize_embeddings(embeddings1: np.ndarray, embeddings2: np.ndarray, labels: Tuple[str, str]):
    """
    Visualize embeddings using UMAP for dimension reduction.
    
    Args:
        embeddings1: Embeddings of first dataset
        embeddings2: Embeddings of second dataset
        labels: Labels for the two datasets
    """
    # Combine embeddings for joint reduction
    combined_embeddings = np.vstack([embeddings1, embeddings2])
    
    # Create labels for plotting
    combined_labels = [labels[0]] * len(embeddings1) + [labels[1]] * len(embeddings2)
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        'dataset': combined_labels
    })
    
    # Set up the figure with just UMAP visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # UMAP for dimension reduction
    try:
        # Configure UMAP for better visualization
        reducer = umap.UMAP(
            random_state=42,
            min_dist=0.1,  
            n_neighbors=15,  # Balance between local and global structure
            metric='cosine'  # Better for text embeddings
        )
        umap_result = reducer.fit_transform(combined_embeddings)
        df['umap_x'] = umap_result[:, 0]
        df['umap_y'] = umap_result[:, 1]
        
        # Plot UMAP results with enhanced styling
        scatter = sns.scatterplot(
            data=df, 
            x='umap_x', 
            y='umap_y', 
            hue='dataset',
            palette=['blue', 'red'],  
            alpha=0.8,  # Slightly more opaque
            s=70,  # Larger points
            edgecolor='white',  # White edges for better visibility
            linewidth=0.5,
            ax=ax
        )
        
        # Add title and customize legend
        ax.set_title('Semantic Distribution of Datasets (UMAP)', fontsize=16)
        ax.set_xlabel('UMAP Dimension 1', fontsize=12)
        ax.set_ylabel('UMAP Dimension 2', fontsize=12)
        
        # Remove axes to focus on the relationships
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Move legend to a better position
        plt.legend(title='Dataset', loc='upper right', frameon=True)
        
    except Exception as e:
        ax.text(0.5, 0.5, f"UMAP visualization failed:\n{str(e)}", 
                ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "embedding_visualization.png", dpi=300)
    plt.close()

def normalize_metrics(metrics_dict: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Normalize all metrics to [0,1] range using min-max scaling.
    
    Args:
        metrics_dict: Dictionary with metrics for each dataset
        
    Returns:
        Dictionary with normalized metrics
    """
    normalized = {}
    
    # Extract all metrics
    metric_names = list(next(iter(metrics_dict.values())).keys())
    
    # Initialize normalized dictionary with same structure
    for dataset, metrics in metrics_dict.items():
        normalized[dataset] = {}
    
    # Normalize each metric separately
    for metric in metric_names:
        values = [metrics[metric] for metrics in metrics_dict.values()]
        min_val = min(values)
        max_val = max(values)
        
        # Avoid division by zero
        if max_val == min_val:
            norm_values = [1.0 for _ in values]
        else:
            norm_values = [(v - min_val) / (max_val - min_val) for v in values]
        
        # Add normalized values back to dictionary
        for i, dataset in enumerate(metrics_dict.keys()):
            normalized[dataset][metric] = norm_values[i]
    
    return normalized

def main():
    """
    Main function to run the dataset diversity analysis
    """
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Define file paths
    seed_file_path = SCRIPT_DIR / "single_tool_seeds.py"
    generated_file_path = SCRIPT_DIR / "outputs" / "simple_100_dataset.json"
    
    # Load datasets
    seed_instructions = load_seed_instructions(seed_file_path)
    generated_instructions = load_generated_instructions(generated_file_path)
    
    # Get embeddings
    seed_embeddings = get_embeddings(seed_instructions)
    generated_embeddings = get_embeddings(generated_instructions)
    
    # Initialize metrics dictionary
    metrics = {
        "Seed Dataset": {},
        "Generated Dataset": {}
    }
    
    # Compute metrics for seed dataset
    metrics["Seed Dataset"]["Distance"] = compute_distance(seed_embeddings)
    metrics["Seed Dataset"]["Dispersion"] = compute_dispersion(seed_embeddings)
    metrics["Seed Dataset"]["Isocontour Radius"] = compute_isocontour_radius(seed_embeddings)
    metrics["Seed Dataset"]["Vocabulary Size"] = compute_vocabulary_size(seed_instructions)
    metrics["Seed Dataset"]["Unique 3-grams"] = compute_unique_ngrams(seed_instructions)
    
    # Compute metrics for generated dataset
    metrics["Generated Dataset"]["Distance"] = compute_distance(generated_embeddings)
    metrics["Generated Dataset"]["Dispersion"] = compute_dispersion(generated_embeddings)
    metrics["Generated Dataset"]["Isocontour Radius"] = compute_isocontour_radius(generated_embeddings)
    metrics["Generated Dataset"]["Vocabulary Size"] = compute_vocabulary_size(generated_instructions)
    metrics["Generated Dataset"]["Unique 3-grams"] = compute_unique_ngrams(generated_instructions)
    
    # Compute affinity
    affinity = compute_affinity(seed_embeddings, generated_embeddings)
    
    # Normalize metrics
    normalized_metrics = normalize_metrics(metrics)
    
    # Create result dataframe
    results = []
    for metric in metrics["Seed Dataset"].keys():
        seed_val = metrics["Seed Dataset"][metric]
        gen_val = metrics["Generated Dataset"][metric]
        
        results.append({
            "Metric": metric,
            "Seed Dataset": seed_val,
            "Generated Dataset": gen_val
        })
    
    # Add affinity score
    results.append({
        "Metric": "Affinity",
        "Seed Dataset": "N/A",
        "Generated Dataset": affinity
    })
    
    # Convert to DataFrame for easier display and export
    df_results = pd.DataFrame(results)
    
    # Save results
    csv_path = OUTPUT_DIR / "dataset_comparison.csv"
    df_results.to_csv(csv_path, index=False)
    
    # Visualize embeddings
    visualize_embeddings(
        seed_embeddings, 
        generated_embeddings,
        labels=("Seed Dataset", "Generated Dataset")
    )
    
    # Analyze diversity - use raw metrics instead of normalized ones
    seed_diversity = np.mean([
        metrics["Seed Dataset"]["Distance"] / max(metrics["Seed Dataset"]["Distance"], metrics["Generated Dataset"]["Distance"]),
        metrics["Seed Dataset"]["Dispersion"] / max(metrics["Seed Dataset"]["Dispersion"], metrics["Generated Dataset"]["Dispersion"]),
        metrics["Seed Dataset"]["Isocontour Radius"] / max(metrics["Seed Dataset"]["Isocontour Radius"], metrics["Generated Dataset"]["Isocontour Radius"])
    ])
    
    gen_diversity = np.mean([
        metrics["Generated Dataset"]["Distance"] / max(metrics["Seed Dataset"]["Distance"], metrics["Generated Dataset"]["Distance"]),
        metrics["Generated Dataset"]["Dispersion"] / max(metrics["Seed Dataset"]["Dispersion"], metrics["Generated Dataset"]["Dispersion"]),
        metrics["Generated Dataset"]["Isocontour Radius"] / max(metrics["Seed Dataset"]["Isocontour Radius"], metrics["Generated Dataset"]["Isocontour Radius"])
    ])
    
    # Calculate metrics
    diversity_ratio = gen_diversity / max(0.0001, seed_diversity)  # Prevent division by zero
    affinity_pct = affinity * 100
    
    # Create a dictionary with analysis results that can be used by other components
    analysis_results = {
        "diversity_ratio": diversity_ratio,
        "affinity_score": affinity,
        "affinity_percentage": affinity_pct,
        "metrics": metrics,
        "normalized_metrics": normalized_metrics,
        "results_dataframe": df_results,
        "is_diverse": gen_diversity > seed_diversity,
        "is_representative": affinity > 0.7,
        "overall_quality": "high" if gen_diversity >= seed_diversity and affinity > 0.7 else 
                          "medium" if (gen_diversity >= seed_diversity or affinity > 0.7) else 
                          "low"
    }
    
    return analysis_results

if __name__ == "__main__":
    analysis_results = main()
    
    # Save results to JSON with proper serialization
    import json
    
    def serialize_for_json(obj):
        """Helper function to serialize numpy and pandas objects for JSON"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float64) or isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, dict):
            return {k: serialize_for_json(v) for k, v in obj.items()}
        else:
            return obj
            
    serializable_results = serialize_for_json(analysis_results)
    
    with open(OUTPUT_DIR / "analysis_results.json", "w") as f:
        json.dump(serializable_results, f, indent=2)