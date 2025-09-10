import sys
from pathlib import Path
WORKDIR = Path(__file__).resolve().parents[1]
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))
from pipeline import write_final_dataset
from src.core.megamodel import MegamodelRegistry
from scripts.run_agent import populate_registry
from src.agents.agent import MCPAgent
from pipeline import (
    _serialize_historical_executions,
    _infer_capabilities_from_registry,
    discover_patterns,
    _build_type_graph,
    sample_apis,
    generate_single_tool_instructions,
    generate_multi_tool_instructions
)


def main() -> None:
    # 1. Get a populated registry
    registry = MegamodelRegistry()
    populate_registry(registry)
    
    # Then run the agent to generate some execution history
    print("\nExecuting agent to generate history...")
    agent = MCPAgent(registry)  # Use same registry instance

    # Ask the agent to transform Class to Relational 
    user_goal = "transform this Class model /Users/zakariahachm/Downloads/llm-agents-mde/src/examples/class.xmi to a Relational model"
    result = agent.run(user_goal)  # This will create session/traces in our registry
    
    print("\nAgent execution completed. Now testing pipeline components...")

    # # Test 2: Type graph building
    # print("\nTesting _build_type_graph:")
    # # Get tools and capabilities first
    # atl_tools = registry.tools_by_server.get("atl_server", [])
    # emf_tools = registry.tools_by_server.get("emf_server", [])
    # tools = [
    #     {"name": getattr(t, "name", ""), "description": getattr(t, "description", "")}
    #     for t in [*atl_tools, *emf_tools]
    # ]
    # capabilities = _infer_capabilities_from_registry(registry, tools)
    
    # type_graph = _build_type_graph(capabilities)
    
    # print("\nTool Chains (Follow Edges):")
    # for tool_name, followers in type_graph["follow_edges"].items():
    #     if followers:  # Only show tools that can be followed by others
    #         print(f"\n{tool_name} can be followed by:")
    #         for follower in followers:
    #             print(f"  {follower}")
    
    # print("\nTool Dependencies (Precede Edges):")
    # for tool_name, predecessors in type_graph["precede_edges"].items():
    #     if predecessors:  # Only show tools that have predecessors
    #         print(f"\n{tool_name} can be preceded by:")
    #         for pred in predecessors:
    #             print(f"  {pred}")

    # # Test 3: Historical executions serialization
    # print("\nTesting _serialize_historical_executions:")
    executions = _serialize_historical_executions(registry)
    
    # if not executions:
    #     print("No execution history found.")
    # else:
    #     print("\nFound execution history:")
    #     for execution in executions:
    #         print(f"\nSession: {execution['session_id']} (Status: {execution['status']})")
    #         for trace in execution['traces']:
    #             print(f"\n  Trace: {trace['trace_id']}")
    #             for inv in trace['invocations']:
    #                 success = "success" if inv['success'] else "failed"
    #                 print(f"    {success} {inv['tool_name']} ({inv['server_name']}) at {inv['timestamp']}")

    # # Test 4: Pattern Discovery
    # print("\nTesting discover_patterns:")
    # Get tools and capabilities first
    atl_tools = registry.tools_by_server.get("atl_server", [])
    emf_tools = registry.tools_by_server.get("emf_server", [])
    tools = [
        {"name": getattr(t, "name", ""), "description": getattr(t, "description", "")}
        for t in [*atl_tools, *emf_tools]
    ]
    capabilities = _infer_capabilities_from_registry(registry, tools)
    
    # Create components dictionary
    components = {
        "executions": executions,
        "capabilities": capabilities,
        "tools": tools
    }
    
    # First get patterns for insights
    patterns_result = discover_patterns(components)
    insights = {
        "common_workflows": patterns_result["common_patterns"],
        "type_graph": patterns_result["type_graph"]
    }
    
    # Test 5: Sample APIs
    print("\nTesting sample_apis:")
    sampled_apis = sample_apis(components, insights)
    
    print("\nTop Ranked Tools (by usage frequency and connectivity):")
    for i, api in enumerate(sampled_apis, 1):
        name = api.get("name", "")
        # Calculate scores for this tool
        freq_score = sum(1 for pattern in patterns_result["common_patterns"] 
                        for tool in pattern["tools"] if tool == name)
        connectivity = len(patterns_result["type_graph"]["follow_edges"].get(name, [])) + \
                      len(patterns_result["type_graph"]["precede_edges"].get(name, []))
        
        print(f"\n{i}. {name}")
        print(f"   Usage Frequency: {freq_score}")
        print(f"   Connectivity Score: {connectivity}")
    
    # # Test 6: Generate Instructions for Top Tools
    # print("\nTesting generate_single_tool_instructions:")
    
    # instructions = generate_single_tool_instructions(
    #     selected_apis=sampled_apis,
    #     per_api=1,  # one instruction per tool
    #     llm_max_calls=10  # allow up to 10 calls for our 10 tools
    # )
    
    # print("\nGenerated Instructions Dataset:")
    # for i, item in enumerate(instructions, 1):
    #     print(f"\nInstruction {i}:")
    #     print(f"Pattern: {item['pattern']}")
    #     print(f"Instruction: {item['instruction']}")
    #     print(f"API: {item['relevant_apis'][0]['api_name']}")
        
    # # Save the single-tool dataset
    # output_path = Path(__file__).parent / "outputs" / "single_tool_instructions.json"
    # write_final_dataset(instructions, output_path)
    # print(f"\nSingle-tool dataset saved to: {output_path}")

    # # Test 7: Generate Multi-Tool Instructions
    # print("\nTesting generate_multi_tool_instructions:")
    
    multi_instructions = generate_multi_tool_instructions(
        selected_apis=sampled_apis,
        chain_len=2,  # two tools in sequence
        per_item=1,   # one instruction per combination
        llm_max_calls=5,  # allow up to 5 calls
        capabilities=capabilities,  # pass capabilities for type compatibility
        enforce_type_compat=True,   # ensure tools can be chained
        insights=insights  # pass the insights with graph and historical patterns
    )
    
    print("\nGenerated Multi-Tool Instructions Dataset:")
    for i, item in enumerate(multi_instructions, 1):
        print(f"\nInstruction {i}:")
        print(f"Pattern: {item['pattern']}")
        print(f"Instruction: {item['instruction']}")
        apis = item['relevant_apis']
        print(f"APIs: {' -> '.join(api['api_name'] for api in apis)}")
        
    # Save the multi-tool dataset
    multi_output_path = Path(__file__).parent / "outputs" / "multi_tool_instructions.json"
    write_final_dataset(multi_instructions, multi_output_path)
    print(f"\nMulti-tool dataset saved to: {multi_output_path}")

if __name__ == "__main__":
    main()
