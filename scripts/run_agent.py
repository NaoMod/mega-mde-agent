import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mcp_servers.atl_server.atl_mcp_server import fetch_transformations
import sys
import os
import requests

# Add src to path FIRST
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.megamodel import MegamodelRegistry
from src.core.am3 import ReferenceModel, TransformationModel
from src.mcp.integrator import MCPServerIntegrator

from src.agents.agent import MCPAgent
from src.mcp.infrastructure import MCPTool

def populate_registry(registry):
    print("Populating MegamodelRegistry with ATL/EMF servers, tools, and transformations...")
    integrator = MCPServerIntegrator(registry)
    atl_server = integrator.setup_atl_server()
    emf_server = integrator.setup_emf_server()
    print(f"ATL Server registered: {atl_server.name}")
    print(f"EMF Server registered: {emf_server.name}")
    print(f"ATL Server tools: {atl_server.tools}")
    print(f"EMF Server tools: {emf_server.tools}")

    # Discover ATL tools using MCP protocol
    from src.mcp.client import MCPClient
    atl_server_script = os.path.join(os.path.dirname(__file__), '..', 'mcp_servers', 'atl_server', 'atl_mcp_server.py')
    atl_client = MCPClient()
    import asyncio
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

    # Call ATL server to get enabled transformations
    # enabled_transformations should be fetched using MCP protocol, not atl_client.call_tool
    enabled_transformations = fetch_transformations()
    print(f"Enabled ATL transformations: {enabled_transformations}")

    # Register transformation tools for ATL server
    print("Registering transformation tools for ATL server...")
    for transfo_data in enabled_transformations:
        transfo_name = transfo_data.get('name')
        # Apply transformation tool
        apply_tool = MCPTool(
            name=f"apply_{transfo_name}_transformation_tool",
            description=f"Apply transformation {transfo_name}",
            server_name=atl_server.name
        )
        atl_tools.append(apply_tool)
        print(f"Registered transformation tool: {apply_tool}")
        # Info tool
        info_tool = MCPTool(
            name=f"list_transformation_{transfo_name}_tool",
            description=f"Get info for transformation {transfo_name}",
            server_name=atl_server.name
        )
        atl_tools.append(info_tool)
        print(f"Registered info tool: {info_tool}")
    atl_server.tools = atl_tools
    registry.tools_by_server[atl_server.name] = atl_tools

    # Call ATL server to get enabled transformations
    # enabled_transformations should be fetched using MCP protocol, not atl_client.call_tool
    enabled_transformations = fetch_transformations()
    print(f"Enabled ATL transformations: {enabled_transformations}")

    # Extract and register all transformations and metamodels
    metamodel_uris = set()
    print("Registering metamodels and transformations...")
    for transfo_data in enabled_transformations:
        # Register input metamodels
        input_mms = transfo_data.get('input_metamodels', [])
        for mm in input_mms:
            uri = mm.get('path')
            name = mm.get('name', uri)
            if uri and uri not in metamodel_uris:
                registry.register_entity(ReferenceModel(uri=uri, name=name))
                metamodel_uris.add(uri)
                print(f"Registered input metamodel: {uri} ({name})")
        # Register output metamodels
        output_mms = transfo_data.get('output_metamodels', [])
        for mm in output_mms:
            uri = mm.get('path')
            name = mm.get('name', uri)
            if uri and uri not in metamodel_uris:
                registry.register_entity(ReferenceModel(uri=uri, name=name))
                metamodel_uris.add(uri)
                print(f"Registered output metamodel: {uri} ({name})")
        # Register transformation
        source_mm = input_mms[0]['path'] if input_mms else None
        target_mm = output_mms[0]['path'] if output_mms else None
        source_ref = registry.get_entity(source_mm) if source_mm else None
        target_ref = registry.get_entity(target_mm) if target_mm else None
        transfo_entity = TransformationModel(
            uri=transfo_data.get('atlFile', transfo_data.get('name', 'unknown')),
            name=transfo_data.get('name', 'unknown'),
            source_metamodel=source_ref,
            target_metamodel=target_ref
        )
        registry.register_entity(transfo_entity)
        print(f"Registered transformation: {transfo_entity.uri} ({transfo_entity.name})")

if __name__ == "__main__":
    registry = MegamodelRegistry()
    populate_registry(registry)

    agent = MCPAgent(registry)

    # Ask the agent to list details for a specific transformation, without forcing a tool
    user_goal = "transform this model /Users/zakariahachm/Downloads/llm-agents-mde/src/examples/class.xmi to a Relational model "
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
