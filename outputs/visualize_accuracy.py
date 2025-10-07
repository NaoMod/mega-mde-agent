#!/usr/bin/env python3
"""
Visualization script for agent accuracy progression
Creates a bar chart showing agent versions ordered by accuracy
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_accuracy_data():
    """Load accuracy data from evaluation summary"""
    summary_file = Path(__file__).parent / "accuracy_evaluation_summary.json"
    
    with open(summary_file, 'r') as f:
        data = json.load(f)
    
    return data

def create_accuracy_plot():
    """Create a bar chart showing agent versions vs accuracy"""
    data = load_accuracy_data()
    
    # Map internal names to display labels as requested
    version_labels = {
        'v0': 'V0: Minimal (no tools, no role)', # Without tools passed nor role-based prompting
        'v1': 'V1: Role-based prompt (no tools)', # Without tools passed + Adding a Role-based prompt
        'without_rag': 'V3: All tools injected (no RAG)',# we pass all tools but no RAG
        'v2': 'V4: RAG + relevant tools + guidelines',# With RAG + some guidelines ( pass the relevant tools)
        'baseline': 'V5: RAG + all tools + all guidelines',# With RAG + All guidelines ( pass the relevant tools)
        'v5': 'V5: RAG + all tools + all guidelines'
    }
    
    # Extract agent names and accuracies with proper labels
    agents = []
    accuracies = []
    
    for agent_name, results in data.items():
        display_name = version_labels.get(agent_name, agent_name.title())
        agents.append(display_name)
        accuracies.append(results['accuracy'])
    
    # Create tuples and sort by accuracy
    agent_accuracy_pairs = list(zip(agents, accuracies))
    agent_accuracy_pairs.sort(key=lambda x: x[1])  # Sort by accuracy
    
    # Unpack sorted data
    sorted_agents, sorted_accuracies = zip(*agent_accuracy_pairs)
    
    # Convert accuracies to percentages
    sorted_accuracies_pct = [acc * 100 for acc in sorted_accuracies]
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    bars = plt.bar(sorted_agents, sorted_accuracies_pct, 
                   color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffa726'])
    
    # Customize the plot
    plt.title('Prompt Evolution Experiment: Systematic Addition of Context and Guidelines\n' +
              'V0 (Minimal) → V1 (+Role) → V3 (+All Tools) → V4 (+RAG+Guidelines) → V5 (+All Guidelines)', 
              fontsize=12, fontweight='bold', pad=20)
    plt.xlabel('Agent Version', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 100)
    
    # Add value labels on bars
    for bar, accuracy in zip(bars, sorted_accuracies_pct):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{accuracy:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    output_file = Path(__file__).parent / "agent_accuracy_progression.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Close the plot to free memory
    plt.close()
    
    return sorted_agents, sorted_accuracies_pct

def print_accuracy_table():
    """Print a formatted table of results"""
    data = load_accuracy_data()
    
    # Map internal names to display labels (same as chart)
    version_labels = {
        'v0': 'V0: Minimal (no tools, no role)',
        'v1': 'V1: Role-based prompt (no tools)', 
        'without_rag': 'V3: All tools injected (no RAG)',
        'v2': 'V4: RAG + relevant tools + guidelines',
        'baseline': 'V5: RAG + all tools + all guidelines',
        'v5': 'V5: RAG + all tools + all guidelines'
    }
    
    # Create tuples with proper labels and sort by accuracy
    agent_accuracy_pairs = []
    for name, results in data.items():
        display_name = version_labels.get(name, name.title())
        agent_accuracy_pairs.append((display_name, results['accuracy']))
    
    agent_accuracy_pairs.sort(key=lambda x: x[1])
    
    print("\n=== Agent Accuracy Progression (Ordered by Performance) ===")
    print(f"{'Rank':<6} {'Agent Version':<15} {'Accuracy':<10} {'Improvement'}")
    print("-" * 55)
    
    prev_accuracy = 0
    for i, (agent, accuracy) in enumerate(agent_accuracy_pairs, 1):
        improvement = accuracy - prev_accuracy if i > 1 else 0
        print(f"{i:<6} {agent:<15} {accuracy*100:>6.1f}%    {improvement*100:>+6.1f}%")
        prev_accuracy = accuracy
    
    print("-" * 55)
    total_improvement = agent_accuracy_pairs[-1][1] - agent_accuracy_pairs[0][1]
    print(f"Total improvement from worst to best: {total_improvement*100:+.1f}%")

if __name__ == "__main__":
    print("Creating agent accuracy visualization...")
    
    try:
        # Print table
        print_accuracy_table()
        
        # Create plot
        agents, accuracies = create_accuracy_plot()
        
        print(f"\nVisualization complete!")
        print(f"Agents in order: {' → '.join(agents)}")
        print(f"Accuracies: {' → '.join([f'{acc:.1f}%' for acc in accuracies])}")
        
    except Exception as e:
        print(f"Error creating visualization: {e}")