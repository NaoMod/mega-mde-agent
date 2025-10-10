import sys
import os
import asyncio
import json
import subprocess
from pathlib import Path
import datetime
import importlib.util
from typing import Optional, List
# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import project modules
from mcp_servers.atl_server.atl_mcp_server import fetch_transformations
from src.core.megamodel import MegamodelRegistry
from src.core.am3 import ReferenceModel, TransformationModel
from src.mcp_ext.integrator import MCPServerIntegrator
from src.mcp_ext.client import MCPClient
from src.agents.execution import MCPInvocation

async def populate_registry(registry):
    integrator = MCPServerIntegrator(registry)
    
    # Get server script paths
    atl_server_script = os.path.join(os.path.dirname(__file__), '..', 'mcp_servers', 'atl_server', 'atl_mcp_server.py')
    emf_server_script = os.path.join(os.path.dirname(__file__), '..', 'mcp_servers', 'emf_server', 'stateless_emf_server.py')
    
    # Setup servers with script paths in metadata
    atl_server = integrator.setup_atl_server()
    emf_server = integrator.setup_emf_server()
    
    # Add script paths to metadata
    atl_server.metadata["script_path"] = atl_server_script
    emf_server.metadata["script_path"] = emf_server_script
    

    # Get ATL tools
    atl_client = MCPClient()
    tools = []
    try:
        await atl_client.connect_to_server(atl_server_script)
        session = await atl_client.get_session()
        response = await session.list_tools()
        tools = response.tools
    finally:
        # Close streams in the same task to avoid anyio cancel-scope warnings
        await atl_client.cleanup()
    
    atl_tools = tools

    # Discover EMF tools using MCP protocol
    emf_client = MCPClient()
    tools = []
    try:
        await emf_client.connect_to_server(emf_server_script)
        session = await emf_client.get_session()
        response = await session.list_tools()
        tools = response.tools
    finally:
        # Close streams in the same task to avoid anyio cancel-scope warnings
        await emf_client.cleanup()
    
    emf_tools = tools
    
    # Register tools with the megamodel registry
    registry.tools_by_server["atl_server"] = atl_tools
    registry.tools_by_server["emf_server"] = emf_tools

    # Call ATL server to get enabled transformations
    enabled_transformations = fetch_transformations()

    # Register transformation tools for ATL server
    def get_or_register_metamodel(uri, name):
        mm = registry.get_entity(uri)
        if not mm:
            mm = ReferenceModel(uri=uri, name=name)
            registry.register_entity(mm)
        return mm

    # Fetch samples once from ATL server
    try:
        samples_raw = subprocess.run([
            'curl', '-s', '-X', 'GET', 'http://localhost:8080/transformations/samples'
        ], capture_output=True, text=True, check=True)
        samples_data = json.loads(samples_raw.stdout)
        # Map name -> sampleSources
        samples_by_name = {entry.get('name'): entry.get('sampleSources', []) for entry in (samples_data or [])}
    except Exception:
        samples_by_name = {}
    for transfo_data in enabled_transformations:
        transfo_name = transfo_data.get('name')
        # Input metamodels
        input_mms = transfo_data.get('input_metamodels', [])
        source_ref = None
        if input_mms:
            mm = input_mms[0]
            source_ref = get_or_register_metamodel(mm.get('path'), mm.get('name', mm.get('path')))
        # Output metamodels
        output_mms = transfo_data.get('output_metamodels', [])
        target_ref = None
        if output_mms:
            mm = output_mms[0]
            target_ref = get_or_register_metamodel(mm.get('path'), mm.get('name', mm.get('path')))
        # Register transformation with references
        transfo_entity = TransformationModel(
            uri=transfo_data.get('atlFile', transfo_data.get('name', 'unknown')),
            name=transfo_data.get('name', 'unknown'),
            source_metamodel=source_ref,
            target_metamodel=target_ref,
            sample_sources=samples_by_name.get(transfo_name, [])
        )
        registry.register_entity(transfo_entity)


def load_agent_class_from_file(file_path: Path):
    """Dynamically load MCPAgent class from a given Python file path.
    Returns the MCPAgent class or None if not found.
    """
    spec = importlib.util.spec_from_file_location("agent_module", str(file_path))
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return getattr(module, "MCPAgent", None)


def load_combined_dataset() -> List[dict]:
    """Load and combine simple_500_dataset.json and multi_500_dataset.json into one list."""
    base_outputs = Path(__file__).parent.parent / "dataset generation" / "outputs"
    files = [
        base_outputs / "simple_100_dataset.json",
        base_outputs / "multi_100_dataset.json",
    ]
    combined: List[dict] = []
    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    combined.extend(data)
                else:
                    print(f"Warning: dataset at {fpath} is not a list; skipping")
        except Exception as e:
            print(f"Error loading dataset file {fpath}: {e}")
    print(f"Loaded combined dataset with {len(combined)} instructions from {len(files)} files")
    return combined

def get_last_instruction_index(agent_name):
    """Get the last successfully executed instruction index for an agent from saved checkpoint."""
    checkpoint_path = Path(__file__).parent.parent / "outputs" / f"{agent_name}_checkpoint.json"
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
                return checkpoint_data.get("last_instruction_index", 0)
        except Exception as e:
            print(f"Error reading checkpoint: {e}")
    return 0

def save_checkpoint(agent_name, last_instruction_index):
    """Save the current progress to a checkpoint file."""
    checkpoint_path = Path(__file__).parent.parent / "outputs" / f"{agent_name}_checkpoint.json"
    try:
        with open(checkpoint_path, 'w') as f:
            json.dump({"last_instruction_index": last_instruction_index}, f)
        print(f"Checkpoint saved for {agent_name} at instruction {last_instruction_index}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

if __name__ == "__main__":
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
        
        # 3. Load the combined dataset once
        dataset = load_combined_dataset()
        if not dataset:
            print("No dataset entries found. Exiting.")
            return
        
        # 4. Discover agent version files
        agents_dir = Path(__file__).parent.parent / "evaluation" / "agent_versions"
        # Modified to only evaluate agent1 and agent2
        agent_files = sorted([p for p in agents_dir.glob("agent[1-2].py") if p.is_file()])
        if not agent_files:
            print(f"No agent version files found in {agents_dir}")
            return
        print(f"Only evaluating agents 1 and 2: {[p.stem for p in agent_files]}")
        
        # 5. Evaluate each agent over the same dataset
        for agent_path in agent_files:
            agent_name = agent_path.stem  # e.g., agent1, agent2, ...

            print(f"\n==== Evaluating {agent_name} on combined dataset ====")
            MCPAgentCls = load_agent_class_from_file(agent_path)
            if MCPAgentCls is None:
                print(f"Skipping {agent_name}: MCPAgent class not found")
                continue
            
            # Prepare per-agent results
            all_execution_results: List[dict] = []
            agent = None
            try:
                # Instantiate agent and connect client to ATL server
                agent = MCPAgentCls(registry)
                client = MCPClient()
                await client.connect_to_server(script_path)
                # Store the client in the executor
                agent.executor.mcp_clients["atl_server"] = client
                
                # Get the last instruction index for this agent
                start_index = get_last_instruction_index(agent_name)
                print(f"Starting from instruction {start_index+1} for {agent_name}")
                
                # Run all instructions starting from the checkpoint
                for i, item in enumerate(dataset):
                    # Skip already processed instructions
                    if i < start_index:
                        continue
                        
                    instruction = item.get("instruction", "")
                    pattern = item.get("pattern", "")
                    apis = item.get("relevant_apis", [])
                    api_names = [api.get("api_name", "") for api in apis]
                    
                    print(f"\n[{i+1}/{len(dataset)}] Running instruction: {instruction}")
                    print(f"  Expected APIs: {api_names}")
                    
                    # Special handling for instruction around 130 which has been problematic
                    if i == 129 or i == 130:
                        print(f"NOTICE: This is instruction {i+1}, which has been problematic. Strict 60-second timeout enforced.")
                    
                    # Generate the plan
                    plan = agent.plan_workflow(instruction)
                    
                    # Force ATL server for transformation tools
                    for step in plan.steps:
                        name = getattr(step, 'tool_name', '') or ''
                        if name.startswith(('apply_', 'list_transformation_')):
                            step.server_name = 'atl_server'
                    
                    # Execute the plan with timeout
                    try:
                        plan.start_execution()
                        session = agent.executor.registry.create_session()
                        session.start()
                        trace = session.create_new_trace()
                        
                        results = []
                        # Set a strict 60-second timeout for all instructions
                        INSTRUCTION_TIMEOUT = 60  # 60 seconds timeout
                        
                        # Create a task for executing all steps with timeout
                        async def execute_steps_with_timeout():
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
                        
                        # Create a task to ensure we can cancel it properly
                        task = asyncio.create_task(execute_steps_with_timeout())
                        
                        try:
                            # Run the execution with strict timeout
                            await asyncio.wait_for(task, INSTRUCTION_TIMEOUT)
                            plan.status = "completed"
                        except asyncio.TimeoutError:
                            # Cancel the task when timeout occurs
                            task.cancel()
                            try:
                                await task  # Wait for cancellation to complete
                            except asyncio.CancelledError:
                                pass  # Task was cancelled successfully
                                
                            print(f"\n\nWARNING: Instruction timed out after {INSTRUCTION_TIMEOUT} seconds. Moving to next instruction.")
                            plan.status = "timeout"
                            results.append({"success": False, "error": f"Execution timed out after {INSTRUCTION_TIMEOUT} seconds"})
                        finally:
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
                            "status": plan.status  # Include execution status (completed or timeout)
                        }
                        
                        # Process the execution results
                        timeout_occurred = plan.status == "timeout"
                        
                        # For normal results
                        for i, (step, result) in enumerate(zip(plan.steps, results)):
                            # Skip timeout error result which was added artificially
                            if timeout_occurred and i == len(plan.steps):
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
                        # Save checkpoint after each successful instruction
                        save_checkpoint(agent_name, i)
                    except KeyboardInterrupt:
                        print(f"\n\nKeyboard interrupt detected during instruction {i+1}.")
                        # Save results so far before exiting
                        if all_execution_results:
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            output_filename = f"agent_execution_results_{agent_name}_{timestamp}_partial.json"
                            output_path = Path(__file__).parent.parent / "outputs" / output_filename
                            output_path.parent.mkdir(exist_ok=True)
                            with open(output_path, 'w') as f:
                                json.dump(all_execution_results, f, indent=2)
                            print(f"\nPartial execution results for {agent_name} saved to: {output_path}")
                            # Always save checkpoint on interrupt
                            save_checkpoint(agent_name, i)
                            print(f"Checkpoint saved. You can restart to continue from instruction {i+1}.")
                        raise  # Re-raise to exit
                    except Exception as e:
                        print(f"Error during plan execution: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    print("--- End Execution ---")
                
                # Save execution results for this agent
                if all_execution_results:
                    try:
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_filename = f"agent_execution_results_{agent_name}_{timestamp}.json"
                        output_path = Path(__file__).parent.parent / "outputs" / output_filename
                        output_path.parent.mkdir(exist_ok=True)
                        with open(output_path, 'w') as f:
                            json.dump(all_execution_results, f, indent=2)
                        print(f"\nExecution results for {agent_name} saved to: {output_path}")
                    except Exception as e:
                        print(f"Error saving results for {agent_name}: {e}")
                        import traceback
                        traceback.print_exc()
            except BaseException as e:
                # Catch setup/connect or other fatal errors for this agent and continue
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
        print("\nProgram terminated by user. Partial results have been saved.")
        print("You can restart to continue from the last saved checkpoint.")
