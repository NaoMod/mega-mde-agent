import sys
import os
import asyncio
import json
import subprocess
from pathlib import Path
# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import project modules
from mcp_servers.atl_server.atl_mcp_server import fetch_transformations
from src.core.megamodel import MegamodelRegistry
from src.core.am3 import ReferenceModel, TransformationModel
from src.mcp_ext.integrator import MCPServerIntegrator
from src.agents.agent import MCPAgent
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

if __name__ == "__main__":
    # Create a single event loop for the entire script
    async def main():
        # Initialize storage for execution results
        all_execution_results = []
        
        # Define excluded transformations (MySQL2KM3 and KM32EMF)
        excluded_transformations = ["MySQL2KM3", "KM32EMF"]
        excluded_tools = []
        for transfo in excluded_transformations:
            excluded_tools.append(f"apply_{transfo}")
            excluded_tools.append(f"list_transformation_{transfo}")
        
        # 1. Set up registry and agent
        registry = MegamodelRegistry()
        await populate_registry(registry)
        agent = MCPAgent(registry)
        
        
        # 3. Connect to ATL server - must be inside the same async function
        atl_server = registry.get_mcp_server("atl_server")
        if not atl_server or not atl_server.metadata.get("script_path"):
            print("ERROR: ATL server not properly configured. Check script path in registry.")
            return
            
        script_path = atl_server.metadata.get("script_path")
        try:
            # Create and connect client
            client = MCPClient()
            await client.connect_to_server(script_path)
            
            
            # Store the client in the executor
            agent.executor.mcp_clients["atl_server"] = client           
            
            # 4. Load the dataset
            dataset_path = Path(__file__).parent.parent / "dataset generation" / "outputs" / "regression_testing_dataset.json"
            try:
                with open(dataset_path, 'r') as f:
                    dataset = json.load(f)
                    print(f"Loaded dataset with {len(dataset)} instructions")
            except Exception as e:
                print(f"Error loading dataset: {e}")
                dataset = []
                
            # 5. Run the instructions
            for i, item in enumerate(dataset):  # Run all instructions
                instruction = item.get("instruction", "")
                pattern = item.get("pattern", "")
                apis = item.get("relevant_apis", [])
                api_names = [api.get("api_name", "") for api in apis]
                
                print(f"\n[{i+1}/{len(dataset)}] Running instruction: {instruction}")
                print(f"  Expected APIs: {api_names}")
                
                # Generate the plan
                plan = agent.plan_workflow(instruction)
                
                # Force ATL server for transformation tools
                for step in plan.steps:
                    name = getattr(step, 'tool_name', '') or ''
                    if name.startswith(('apply_', 'list_transformation_')):
                        step.server_name = 'atl_server'
                
                # Execute the plan
                try:
                    # We'll execute each step manually for better control
                    plan.start_execution()
                    session = agent.executor.registry.create_session()
                    session.start()
                    trace = session.create_new_trace()
                    
                    results = []
                    for step in plan.steps:
                        # Update step readiness
                        step.status = "ready"
                        
                        # Execute the step
                        result = await agent.executor.execute_step_async(step)
                        results.append(result)
                        
                        # Add to trace
                        invocation = MCPInvocation(
                            tool_name=step.tool_name,
                            server_name=step.server_name,
                            arguments=step.parameters,
                            result=result.get("result", {}),
                            success=result["success"]
                        )
                        trace.add_invocation(invocation)
                        
                        # Print result
                        success = result.get('success', False)
                        
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
                        "execution_results": []
                    }
                    
                    # Process the execution results
                    for step, result in zip(plan.steps, results):
                        success = result.get('success', False)
                        result_data = result.get('result', {})
                        
                        # Create a serializable version of the result
                        serialized_result = {
                            "tool_name": step.tool_name,
                            "success": success,
                            "error": result.get('error', '') if not success else ''
                        }
                        
                        # Handle result data based on type
                        if success:
                            if hasattr(result_data, 'to_dict'):
                                serialized_result["result"] = result_data.to_dict()
                            elif hasattr(result_data, '__dict__'):
                                try:
                                    result_dict = result_data.__dict__
                                    if 'text' in result_dict:
                                        serialized_result["result"] = {"text": result_dict['text']}
                                    else:
                                        # Make a serializable dict
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
                    
                    # Add to all results
                    all_execution_results.append(execution_result)
                    
                except Exception as e:
                    print(f"Error during plan execution: {e}")
                    import traceback
                    traceback.print_exc()
                    
                print("--- End Execution ---")
                
            # 6. Cleanup - do this directly without using the restored function
            for server_name, client in agent.executor.mcp_clients.items():
                try:
                    if hasattr(client, 'exit_stack'):
                        await client.exit_stack.aclose()
                except Exception:
                    pass
            
            # Save execution results to a JSON file
            if all_execution_results:
                try:
                    output_path = Path(__file__).parent.parent / "outputs" / "agent_execution_results.json"
                    # Ensure the output directory exists
                    output_path.parent.mkdir(exist_ok=True)
                    
                    with open(output_path, 'w') as f:
                        json.dump(all_execution_results, f, indent=2)
                except Exception:
                    import traceback
                    traceback.print_exc()
            
        except Exception:
            import traceback
            traceback.print_exc()
    
    # Run the async main function
    asyncio.run(main())
