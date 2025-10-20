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
    

    # Load only baseline and reduced are ons
    reduced_tools_data = load_execution_results(os.path.join(current_dir, 'agent_execution_results_MCPAgent_reduced_tools_20251020_122115.json'))
    num_instructions = len(reduced_tools_data)


    print(f"Analyzing {num_instructions} instructions across 2 versions")
    print("-" * 60)

    # Create CSV report
    csv_file = os.path.join(current_dir, 'report_generation.csv')

    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Instruction_ID', 'Instruction', 'Missing_Tools', 'Score'])

        transformations_with_issues = {}

        # Use the same scoring as the accuracy script
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
                return api_name

        def score_instruction(instruction_data):
            expected_apis = instruction_data.get("expected_apis", [])
            execution_results = instruction_data.get("execution_results", [])
            if not expected_apis:
                return 0.0
            expected_tools = [map_api_to_tool_name(api) for api in expected_apis]
            successful_tools = [result.get("tool_name", "") for result in execution_results if result.get("success", False)]
            if len(expected_tools) == 1:
                return 1.0 if expected_tools[0] in successful_tools else 0.0
            else:
                score = 0.0
                points_per_tool = 0.5 if len(expected_tools) == 2 else (1.0 / len(expected_tools))
                for expected_tool in expected_tools:
                    if expected_tool in successful_tools:
                        score += points_per_tool
                return min(score, 1.0)

    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Instruction_ID', 'Instruction', 'Missing_Tools', 'Score'])

        unique_missing_tools = set()
        for i in range(num_instructions):
            reduced_tools_item = reduced_tools_data[i]
            instruction_text = reduced_tools_item.get('instruction', f'Instruction {i+1}')
            score = score_instruction(reduced_tools_item)
            if score < 1.0:
                expected_apis = reduced_tools_item.get("expected_apis", [])
                expected_tools = [map_api_to_tool_name(api) for api in expected_apis]
                execution_results = reduced_tools_item.get("execution_results", [])
                successful_tools = [result.get("tool_name", "") for result in execution_results if result.get("success", False)]
                missing_tools = [tool for tool in expected_tools if tool not in successful_tools]

                writer.writerow([
                    i + 1,
                    instruction_text[:100] + "..." if len(instruction_text) > 100 else instruction_text,
                    " | ".join(set(missing_tools)),
                    f"{score:.2f}"
                ])

                # Collect unique missing tools
                unique_missing_tools.update(missing_tools)

                # Extract transformations with issues
                for tool in missing_tools:
                    if "transformation" in tool:
                        if "list_transformation_" in tool:
                            transfo_name = tool.replace("list_transformation_", "").replace("_tool", "")
                            transformations_with_issues[transfo_name] = transformations_with_issues.get(transfo_name, 0) + 1
                        elif "apply_" in tool and "_transformation_tool" in tool:
                            transfo_name = tool.replace("apply_", "").replace("_transformation_tool", "")
                            transformations_with_issues[transfo_name] = transformations_with_issues.get(transfo_name, 0) + 1

        # Filter and sort transformations that appear more than 3 times
        frequent_issues = {k: v for k, v in transformations_with_issues.items() if v > 3}
        sorted_frequent = sorted(frequent_issues.items(), key=lambda x: x[1], reverse=True)

        # Add transformations with issues as last row in CSV
        if sorted_frequent:
            transformations_summary = " | ".join([f"{transfo}({count})" for transfo, count in sorted_frequent])
            writer.writerow(['SUMMARY', 'Transformations with issues (>3 times)', transformations_summary, ''])

        # Add unique missing tools as the final row
        if unique_missing_tools:
            writer.writerow(['ALL_MISSING_TOOLS', 'Unique missing tools across all failures', " | ".join(sorted(unique_missing_tools)), ''])
                
if __name__ == "__main__":
    main()