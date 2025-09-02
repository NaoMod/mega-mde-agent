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
from src.mcp.integrator import MCPServerIntegrator
from src.agents.agent import MCPAgent
from src.mcp.infrastructure import MCPTool
from src.mcp.client import MCPClient

def populate_registry(registry):
    print("Populating MegamodelRegistry with ATL/EMF servers, tools, and transformations...")
    integrator = MCPServerIntegrator(registry)
    atl_server = integrator.setup_atl_server()
    emf_server = integrator.setup_emf_server()
    print(f"ATL Server registered: {atl_server.name}")
    print(f"EMF Server registered: {emf_server.name}")
    print(f"ATL Server tools: {atl_server.tools}")
    print(f"EMF Server tools: {emf_server.tools}")


    atl_server_script = os.path.join(os.path.dirname(__file__), '..', 'mcp_servers', 'atl_server', 'atl_mcp_server.py')
    atl_client = MCPClient()
    async def get_atl_tools():
        await atl_client.connect_to_server(atl_server_script)
        session = await atl_client.get_session()
        response = await session.list_tools()
        return response.tools
    atl_tools = asyncio.run(get_atl_tools())


    # Discover EMF tools using MCP protocol
    emf_server_script = os.path.join(os.path.dirname(__file__), '..', 'mcp_servers', 'emf_server', 'stateless_emf_server.py')
    emf_client = MCPClient()
    async def get_emf_tools():
        await emf_client.connect_to_server(emf_server_script)
        session = await emf_client.get_session()
        response = await session.list_tools()
        return response.tools
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
            target_metamodel=target_ref
        )
        registry.register_entity(transfo_entity)
        print(f"Registered transformation: {transfo_entity.uri} ({transfo_entity.name}) | IN: {getattr(source_ref, 'name', None)} | OUT: {getattr(target_ref, 'name', None)}")

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
    user_goal = "transform this model /Users/zakariahachm/Downloads/llm-agents-mde/src/examples/class.xmi to a Relational model;  then list me the transformation that transform an Ant file to a maven model"
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
