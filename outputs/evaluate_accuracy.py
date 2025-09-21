#!/usr/bin/env python3
"""
Script to evaluate the accuracy of agent execution results.

Scoring criteria:
- Score 1: If success is true AND the expected tool is found among the executed tools
- Score 0.5: For multi-instruction cases where each expected tool gets 0.5 points
- Score 0: If the expected tool is not found or execution failed

The script maps API names to tool names:
- "ToolName.get_tool" -> "list_transformation_ToolName_tool"
- "ToolName.apply_tool" -> "apply_ToolName_tool"
"""

import json

from pathlib import Path

def map_api_to_tool_name(api_name):

    if not api_name or '.' not in api_name:
        return api_name
    
    tool_base, action = api_name.split('.', 1)
    
    if action == "get_tool":
        return f"list_transformation_{tool_base}_tool"
    elif action == "apply_tool":
        return f"apply_{tool_base}_tool"
    elif action == "apply":
        return f"apply_{tool_base}_transformation_tool"
    else:
        # Handle other patterns if needed
        return api_name

def evaluate_instruction(instruction_data):

    expected_apis = instruction_data.get("expected_apis", [])
    execution_results = instruction_data.get("execution_results", [])
    
    if not expected_apis:
        return 0.0
    
    # Map expected APIs to tool names
    expected_tools = [map_api_to_tool_name(api) for api in expected_apis]
    
    # Get successfully executed tools
    successful_tools = []
    for result in execution_results:
        if result.get("success", False):
            successful_tools.append(result.get("tool_name", ""))
    
    # Calculate score
    if len(expected_tools) == 1:
        # Single tool expected - score 1 if found, 0 if not
        expected_tool = expected_tools[0]
        return 1.0 if expected_tool in successful_tools else 0.0
    else:
        # Multiple tools expected - 0.5 points for each found tool
        score = 0.0
        points_per_tool = 0.5 if len(expected_tools) == 2 else (1.0 / len(expected_tools))
        
        for expected_tool in expected_tools:
            if expected_tool in successful_tools:
                score += points_per_tool
        
        return min(score, 1.0)  # Cap at 1.0

def evaluate_file(file_path):
    """
    Evaluate all instructions in a JSON file and return the average accuracy.
    
    Args:
        file_path: Path to the JSON file containing execution results
        
    Returns:
        tuple: (average_accuracy, detailed_results)
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        return 0.0, []
    
    if not isinstance(data, list):
        return 0.0, []
    
    detailed_results = []
    total_score = 0.0
    
    for i, instruction_data in enumerate(data):
        score = evaluate_instruction(instruction_data)
        total_score += score
        
        detailed_results.append({
            "instruction_index": i,
            "instruction": instruction_data.get("instruction", ""),
            "expected_apis": instruction_data.get("expected_apis", []),
            "score": score,
            "success_tools": [
                result.get("tool_name", "") 
                for result in instruction_data.get("execution_results", [])
                if result.get("success", False)
            ]
        })
    
    average_accuracy = total_score / len(data) if data else 0.0
    return average_accuracy, detailed_results

def main():
    """Main function to evaluate both result files."""
    
    # Define the paths to the result files
    outputs_dir = Path(__file__).parent
    baseline_file = outputs_dir / "agent_baseline_execution_results.json"
    new_file = outputs_dir / "agent_execution_results_20250921_181958.json"
    

    # Evaluate baseline results
    if baseline_file.exists():
        baseline_accuracy, baseline_details = evaluate_file(baseline_file)

    else:
        baseline_accuracy, baseline_details = 0.0, []
    
    # Evaluate new results
    if new_file.exists():
        new_accuracy, new_details = evaluate_file(new_file)
    else:
        new_accuracy, new_details = 0.0, []
    
    # Compare results
    if baseline_details and new_details:
        improvement = new_accuracy - baseline_accuracy

    # Show detailed results for both baseline and new
        
    # Save detailed results to a file
    if new_details or baseline_details:
        results_summary = {
            "baseline": {
                "file": baseline_file.name if baseline_file.exists() else None,
                "accuracy": baseline_accuracy,
                "total_instructions": len(baseline_details)
            },
            "new": {
                "file": new_file.name if new_file.exists() else None,
                "accuracy": new_accuracy,
                "total_instructions": len(new_details)
            }
        }
        
        summary_file = outputs_dir / "accuracy_evaluation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"Detailed results saved to: {summary_file}")

if __name__ == "__main__":
    main()