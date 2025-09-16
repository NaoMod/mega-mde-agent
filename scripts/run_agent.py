import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mcp_servers.atl_server.atl_mcp_server import fetch_transformations
import sys
import os
import asyncio
# Add src to path 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from src.core.megamodel import MegamodelRegistry
from src.core.am3 import ReferenceModel, TransformationModel
from src.mcp_ext.integrator import MCPServerIntegrator
from src.agents.agent import MCPAgent
from src.mcp_ext.client import MCPClient
import json
import subprocess

def populate_registry(registry):
    print("Populating MegamodelRegistry with ATL/EMF servers, tools, and transformations...")
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
    
    # print(f"ATL Server registered: {atl_server.name}")
    # print(f"EMF Server registered: {emf_server.name}")
    # print(f"ATL Server tools: {atl_server.tools}")
    # print(f"EMF Server tools: {emf_server.tools}")


    atl_server_script = os.path.join(os.path.dirname(__file__), '..', 'mcp_servers', 'atl_server', 'atl_mcp_server.py')
    atl_client = MCPClient()
    async def get_atl_tools():
        await atl_client.connect_to_server(atl_server_script)
        tools = []
        try:
            session = await atl_client.get_session()
            response = await session.list_tools()
            tools = response.tools
        finally:
            # Close streams in the same task to avoid anyio cancel-scope warnings
            await atl_client.cleanup()
        return tools
    atl_tools = asyncio.run(get_atl_tools())


    # Discover EMF tools using MCP protocol
    emf_server_script = os.path.join(os.path.dirname(__file__), '..', 'mcp_servers', 'emf_server', 'stateless_emf_server.py')
    emf_client = MCPClient()
    async def get_emf_tools():
        await emf_client.connect_to_server(emf_server_script)
        tools = []
        try:
            session = await emf_client.get_session()
            response = await session.list_tools()
            tools = response.tools
        finally:
            # Close streams in the same task to avoid anyio cancel-scope warnings
            await emf_client.cleanup()
        return tools
    emf_tools = asyncio.run(get_emf_tools())
    
    # Register tools with the megamodel registry
    registry.tools_by_server["atl_server"] = atl_tools
    registry.tools_by_server["emf_server"] = emf_tools
    print(f"Added {len(atl_tools)} ATL tools and {len(emf_tools)} EMF tools to registry")

    # Call ATL server to get enabled transformations
    # enabled_transformations should be fetched using MCP protocol, not atl_client.call_tool
    enabled_transformations = fetch_transformations()
    print(f"Enabled ATL transformations: {enabled_transformations}")

    # Register transformation tools for ATL server
    print("Registering transformation tools for ATL server...")
    def get_or_register_metamodel(uri, name):
        mm = registry.get_entity(uri)
        if not mm:
            mm = ReferenceModel(uri=uri, name=name)
            registry.register_entity(mm)
        return mm

    print("Registering metamodels and transformations...")

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
    registry = MegamodelRegistry()
    populate_registry(registry)

    agent = MCPAgent(registry)

    # Print all ATL transformations and their IN/OUT metamodels
    print("\n--- ATL Transformations and Metamodel Links ---")
    for entity in registry.entities.values():
        if hasattr(entity, "source_metamodel") and hasattr(entity, "target_metamodel"):
            in_mm = entity.source_metamodel
            out_mm = entity.target_metamodel
            print(f"Transformation: {getattr(entity, 'name', None)} | IN: {getattr(in_mm, 'name', None)} ({getattr(in_mm, 'uri', None)}) | OUT: {getattr(out_mm, 'name', None)} ({getattr(out_mm, 'uri', None)})")
    print("--- End ATL Transformations ---\n")

    # Ask the agent to list details for a specific transformation, without forcing a tool
    user_goal = "transform this Class model /Users/zakariahachm/Downloads/llm-agents-mde/src/examples/class.xmi to a Relational model"
    #user_goal= "can you add a class object to this model /Users/zakariahachm/Downloads/llm-agents-mde/src/examples/class.xmi"
    print(f"\nAgent user goal: {user_goal}")

    print("\n--- Agent Planning Debug ---")
    plan = agent.plan_workflow(user_goal)
    for i, step in enumerate(plan.steps):
        print(f"Step {i+1}: tool={step.tool_name}, server={step.server_name}, params={step.parameters}, desc={step.description}")
    print("--- End Agent Planning Debug ---\n")

    # Do not inject any manual parameters; let the agent choose the correct tool and inputs

    print("\n--- Agent Execution Debug ---")
    result = agent.executor.execute_workflow(plan)
    print(result)
    print("--- End Agent Execution Debug ---\n")
