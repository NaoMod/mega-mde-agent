import json
import csv
import os

# TODO: Planning vs Execution Gap: How often does the agent plan correctly but fail in execution?

def load_execution_results(file_path):
    """Load execution results and organize by instruction index."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def map_api_to_tool_name(api_name):
    """Map API names to tool names."""
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
        return api_name

def evaluate_instruction(instruction_data):
    """Evaluate if an instruction succeeded based on expected vs executed tools."""
    expected_apis = instruction_data.get("expected_apis", [])
    execution_results = instruction_data.get("execution_results", [])
    
    if not expected_apis:
        return False, []
    
    # Map expected APIs to tool names
    expected_tools = [map_api_to_tool_name(api) for api in expected_apis]
    
    # Get successfully executed tools
    successful_tools = []
    for result in execution_results:
        if result.get("success", False):
            successful_tools.append(result.get("tool_name", ""))
    
    # Check if all expected tools were successfully executed
    success = all(tool in successful_tools for tool in expected_tools)
    
    return success, expected_tools

def main():
    current_dir = '/Users/zakariahachm/Downloads/llm-agents-mde/outputs'
    
    # Load all three versions
    baseline_data = load_execution_results(os.path.join(current_dir, 'agent_baseline_execution_results.json'))
    without_2tools_data = load_execution_results(os.path.join(current_dir, 'agent_execution_results_without_2tools.json'))
    without_pe_data = load_execution_results(os.path.join(current_dir, 'agent_execution_results_withoutPE_techniques.json'))
    
    # Ensure all datasets have same number of instructions
    num_instructions = min(len(baseline_data), len(without_2tools_data), len(without_pe_data))
    
    print(f"Analyzing {num_instructions} instructions across 3 versions")
    print("-" * 60)
    
    # Create CSV report
    csv_file = os.path.join(current_dir, 'report_generation.csv')
    
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Instruction_ID', 'Instruction', 'Baseline_Status', 'Without_2tools_Status', 'Without_PE_Status', 'Failed_Version', 'Missing_Tools'])
        
        mixed_results_count = 0
        
        for i in range(num_instructions):
            instruction_text = baseline_data[i].get('instruction', f'Instruction {i+1}')
            
            # Evaluate each version
            baseline_success, baseline_expected = evaluate_instruction(baseline_data[i])
            without_2tools_success, without_2tools_expected = evaluate_instruction(without_2tools_data[i])
            without_pe_success, without_pe_expected = evaluate_instruction(without_pe_data[i])
            
            # Check if we have mixed results (some succeed, some fail)
            results = [baseline_success, without_2tools_success, without_pe_success]
            if not all(results) and any(results):  # Mixed results
                mixed_results_count += 1
                
                # Identify failed versions and their missing tools
                failed_versions = []
                missing_tools = []
                
                if not baseline_success:
                    failed_versions.append('Baseline')
                    missing_tools.extend(baseline_expected)
                
                if not without_2tools_success:
                    failed_versions.append('Without_2tools')
                    missing_tools.extend(without_2tools_expected)
                
                if not without_pe_success:
                    failed_versions.append('Without_PE')
                    missing_tools.extend(without_pe_expected)
                
                writer.writerow([
                    i + 1,
                    instruction_text[:100] + "..." if len(instruction_text) > 100 else instruction_text,
                    "SUCCESS" if baseline_success else "FAILED",
                    "SUCCESS" if without_2tools_success else "FAILED",
                    "SUCCESS" if without_pe_success else "FAILED",
                    " | ".join(failed_versions),
                    " | ".join(set(missing_tools))
                ])
                
if __name__ == "__main__":
    main()