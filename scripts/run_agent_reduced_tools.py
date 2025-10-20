import sys
import os
import asyncio
import json
import argparse
import random
from pathlib import Path
import datetime
from typing import List
from dotenv import load_dotenv
from scripts.run_agent_versions import populate_registry
from src.agents.workflow import WorkflowPlan
# Load environment variables from .env file
load_dotenv(Path(__file__).parent / '.env')
# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import project modules
from src.core.megamodel import MegamodelRegistry
from src.mcp_ext.client import MCPClient
from src.agents.execution import MCPInvocation
from src.agents.agent import MCPAgent


# Define 10 tools to remove (you can modify this list)
TOOLS_TO_REMOVE = [
    "list_transformation_KM32EMF_tool",
    "apply_KM32EMF_transformation_tool",
    "list_transformation_MySQL2KM3_tool",
    "apply_MySQL2KM3_transformation_tool",*
    "list_transformation_Families2Persons_tool",
    "apply_Families2Persons_transformation_tool",
    "list_transformation_XML2Ant_tool",*
    "apply_XML2Ant_transformation_tool",
    "list_transformation_Make2Ant_tool",
    "apply_Make2Ant_transformation_tool"
]



def load_dataset(file_name: str) -> List[dict]:
    """Load a dataset file from dataset generation/outputs."""
    base_outputs = Path(__file__).parent.parent / "dataset generation" / "outputs"
    fpath = base_outputs / file_name
    try:
        with open(fpath, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                print(f"Loaded {len(data)} items from {fpath}")
                return data
            else:
                print(f"Warning: dataset at {fpath} is not a list; returning empty list")
                return []
    except Exception as e:
        print(f"Error loading dataset file {fpath}: {e}")
        return []


def sample_dataset(single_dataset: List[dict], multi_dataset: List[dict], 
                   single_count: int = 100, multi_count: int = 100, 
                   seed: int = 42) -> List[dict]:
    """
    Randomly sample items from single and multi datasets.
    
    Args:
        single_dataset: List of single-tool instructions
        multi_dataset: List of multi-tool instructions
    
    Returns:
        Combined list of sampled items
    """
    random.seed(seed)
    
    # Sample from single dataset
    if len(single_dataset) >= single_count:
        sampled_single = random.sample(single_dataset, single_count)
    else:
        print(f"Warning: Only {len(single_dataset)} single-tool items available, using all")
        sampled_single = single_dataset
    
    # Sample from multi dataset
    if len(multi_dataset) >= multi_count:
        sampled_multi = random.sample(multi_dataset, multi_count)
    else:
        print(f"Warning: Only {len(multi_dataset)} multi-tool items available, using all")
        sampled_multi = multi_dataset
    
    return sampled_single + sampled_multi


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run agent evaluation with reduced tools on sampled dataset")
    parser.add_argument("--agent", type=str, default="agent2", 
                        help="Agent version to evaluate (e.g., agent2, agent3)")
    parser.add_argument("--single-count", type=int, default=100,
                        help="Number of single-tool instructions to sample (default: 100)")
    parser.add_argument("--multi-count", type=int, default=100,
                        help="Number of multi-tool instructions to sample (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    args = parser.parse_args()
    


    # Create a single event loop for the entire script
    async def main():
        # 1. Set up registry and populate tools/transformations
        registry = MegamodelRegistry()
        await populate_registry(registry)
        
        # 2. Get ATL server script path for connections
        atl_server = registry.get_mcp_server("atl_server")
        if not atl_server or not atl_server.metadata.get("script_path"):
            print("ERROR: ATL server not properly configured. Check script path in registry.")
            return
            
        script_path = atl_server.metadata.get("script_path")
        
        print(f"\n=== Evaluating MCPAgent with reduced tools ===\n")
        agent_name = "MCPAgent"
        # Prepare per-agent results
        all_execution_results: List[dict] = []
        agent = None
        try:
            # Instantiate agent and connect client to ATL server
            agent = MCPAgent(registry)
            client = MCPClient()
            await client.connect_to_server(script_path)
            # Store the client in the executor
            agent.executor.mcp_clients["atl_server"] = client
            # Remove the specified tools from the registry after population
            atl_tools = registry.tools_by_server.get("atl_server", [])
            original_atl_count = len(atl_tools)
            atl_tools = [tool for tool in atl_tools if getattr(tool, 'name', '') not in TOOLS_TO_REMOVE]
            removed_count = original_atl_count - len(atl_tools)
            registry.tools_by_server["atl_server"] = atl_tools
            print(f"Original ATL tools: {original_atl_count}")
            print(f"Removed tools: {removed_count}")
            print(f"Remaining ATL tools: {len(atl_tools)}")
            print(f"Tools removed: {TOOLS_TO_REMOVE}")
            print("=" * 50 + "\n")

            # Load both datasets
            simple_dataset = load_dataset("simple_500_dataset.json")
            multi_dataset = load_dataset("multi_500_dataset.json")
            
            # Sample the datasets
            combined_dataset = sample_dataset(
                simple_dataset, 
                multi_dataset, 
                args.single_count, 
                args.multi_count,
                args.seed
            )
            
            if not combined_dataset:
                print("ERROR: No entries found in sampled dataset, nothing to run.")
                return
                
            print(f"\n-- Running {len(combined_dataset)} sampled instructions --")
            for i, item in enumerate(combined_dataset):
                instruction = item.get("instruction", "")
                pattern = item.get("pattern", "")
                apis = item.get("relevant_apis", [])
                api_names = [api.get("api_name", "") for api in apis]
                
                print(f"\n[{i+1}/{len(combined_dataset)}] Running instruction: {instruction}")
                print(f"  Expected APIs: {api_names}")
                
                # Generate the plan without timeout
                try:
                    # Call plan generation directly without timeout
                    plan = await asyncio.to_thread(agent.plan_workflow, instruction)
                    
                    # Force ATL server for transformation tools
                    for step in plan.steps:
                        name = getattr(step, 'tool_name', '') or ''
                        if name.startswith(('apply_', 'list_transformation_')):
                            step.server_name = 'atl_server'
                    # proceed to execute the generated plan for this instruction
                except (asyncio.TimeoutError, Exception) as e:
                    print(f"\n\nWARNING: Planning timed out or failed for instruction {i+1}: {e}")
                    plan = WorkflowPlan(instruction)
                    plan.status = "planning_failed"
                
                # Execute the plan
                try:
                    plan.start_execution()
                    session = agent.executor.registry.create_session()
                    session.start()
                    trace = session.create_new_trace()
                    
                    results = []
                    # Execute steps without timeout
                    for step in plan.steps:
                        step.status = "ready"
                        result = await agent.executor.execute_step_async(step)
                        results.append(result)
                        invocation = MCPInvocation(
                            tool_name=step.tool_name,
                            server_name=step.server_name,
                            arguments=step.parameters,
                            result=result.get("result", {}),
                            success=result.get("success", False)
                        )
                        trace.add_invocation(invocation)
                    
                    # Mark plan as completed after execution
                    plan.status = "completed"
                    session.end()
                    
                    # Create a serializable result for this instruction
                    execution_result = {
                        "instruction": instruction,
                        "pattern": pattern,
                        "expected_apis": api_names,
                        "plan_steps": [
                            {
                                "tool_name": step.tool_name,
                                "server_name": step.server_name,
                                "parameters": step.parameters
                            } for step in plan.steps
                        ],
                        "execution_results": [],
                        "status": plan.status
                    }
                    
                    # Process the execution results
                    timeout_occurred = plan.status == "timeout"
                    
                    # Converts result objects to JSON-serializable format
                    for idx, (step, result) in enumerate(zip(plan.steps, results)):
                        # Skip timeout error result which was added artificially
                        if timeout_occurred and idx == len(plan.steps):
                            continue
                            
                        success = result.get('success', False)
                        result_data = result.get('result', {})
                        serialized_result = {
                            "tool_name": step.tool_name,
                            "success": success,
                            "error": result.get('error', '') if not success else ''
                        }
                        if success:
                            if hasattr(result_data, 'to_dict'):
                                serialized_result["result"] = result_data.to_dict()
                            elif hasattr(result_data, '__dict__'):
                                try:
                                    result_dict = result_data.__dict__
                                    if 'text' in result_dict:
                                        serialized_result["result"] = {"text": result_dict['text']}
                                    else:
                                        clean_dict = {}
                                        for k, v in result_dict.items():
                                            if isinstance(v, (str, int, float, bool, type(None))):
                                                clean_dict[k] = v
                                            else:
                                                clean_dict[k] = str(v)
                                        serialized_result["result"] = clean_dict
                                except Exception:
                                    serialized_result["result"] = str(result_data)
                            else:
                                serialized_result["result"] = str(result_data)
                        execution_result["execution_results"].append(serialized_result)
                    
                    # Add timeout error if it occurred
                    if timeout_occurred and len(results) > len(plan.steps):
                        timeout_result = results[-1]  # Last result is the timeout error
                        execution_result["execution_results"].append({
                            "tool_name": "execution_timeout",
                            "success": False,
                            "error": timeout_result.get('error', 'Execution timed out')
                        })
                    
                    all_execution_results.append(execution_result)
                except KeyboardInterrupt:
                    # Propagate to outer handler; no partial saves
                    raise
                except Exception as e:
                    print(f"Error during plan execution: {e}")
                    import traceback
                    traceback.print_exc()
                
                print("--- End Execution ---")
            
            # Save all results
            if all_execution_results:
                outputs_dir = Path(__file__).parent.parent / "outputs"
                outputs_dir.mkdir(exist_ok=True)
                try:
                    # Save results with descriptive filename
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_filename = f"agent_execution_results_{agent_name}_reduced_tools_{timestamp}.json"
                    output_path = outputs_dir / output_filename
                    with open(output_path, 'w') as f:
                        # Convert any sets to lists before serializing
                        json_str = json.dumps(all_execution_results, indent=2, default=lambda o: list(o) if isinstance(o, set) else str(o))
                        f.write(json_str)
                    print(f"\n{'='*60}")
                    print(f"Execution results saved to: {output_path}")
                    print(f"Agent: {agent_name}")
                    print(f"Tools removed: {len(TOOLS_TO_REMOVE)}")
                    print(f"Instructions evaluated: {len(all_execution_results)}")
                    print(f"{'='*60}\n")
                except Exception as e:
                    print(f"Error saving results for {agent_name}: {e}")
                    import traceback
                    traceback.print_exc()
        except BaseException as e:
            # BaseException catches everything including KeyboardInterrupt, SystemExit, etc.
            print(f"Error setting up or running {agent_name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup per-agent clients
            try:
                for server_name, cli in getattr(agent, 'executor', object()).mcp_clients.items():
                    try:
                        if hasattr(cli, 'exit_stack'):
                            try:
                                await cli.exit_stack.aclose()
                            except BaseException:
                                # Swallow cancellation/errors on shutdown to avoid aborting the whole run
                                pass
                    except BaseException:
                        pass
            except BaseException:
                pass
            # Small delay to allow subprocess cleanup before next agent
            try:
                await asyncio.sleep(0.2)
            except BaseException:
                # Ignore cancellations occurring at this checkpoint
                pass
    
    # Run the async main function with graceful handling of Ctrl+C
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
