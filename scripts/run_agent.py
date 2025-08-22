import sys
import os
import requests

# Add src to path FIRST
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.megamodel import MegamodelRegistry
from core.am3 import ReferenceModel, TransformationModel
from mcp.integrator import MCPServerIntegrator
from mcp.client import ATLServerClient
from agents.agent import MCPAgent
from mcp.infrastructure import MCPTool

def populate_registry(registry):
    print("Populating MegamodelRegistry with ATL/EMF servers, tools, and transformations...")
    integrator = MCPServerIntegrator(registry)
    atl_server = integrator.setup_atl_server()
    emf_server = integrator.setup_emf_server()
    print(f"ATL Server registered: {atl_server.name}")
    print(f"EMF Server registered: {emf_server.name}")
    print(f"ATL Server tools: {atl_server.tools}")
    print(f"EMF Server tools: {emf_server.tools}")

    # Fetch and register tools for ATL server (port 8081)
    atl_tools = []
    atl_client = ATLServerClient(base_url=f"http://{atl_server.host}:{atl_server.port}")
    try:
        atl_tools_response = requests.get(f"http://{atl_server.host}:8081/tools")
        print(f"Raw ATL /tools response: {atl_tools_response.text}")
        atl_tools_json = atl_tools_response.json()
        if "tools" in atl_tools_json:
            for tool in atl_tools_json["tools"]:
                tool_obj = MCPTool(
                    name=tool.get("name"),
                    description=tool.get("description", ""),
                    endpoint=tool.get("name"),
                    server_name=atl_server.name
                )
                atl_tools.append(tool_obj)
        atl_server.tools = atl_tools
        registry.tools_by_server[atl_server.name] = atl_tools
        print(f"ATL server tool objects registered: {atl_tools}")
    except Exception as e:
        print(f"Could not fetch ATL server tools: {e}")

    # Fetch and register tools for EMF server (port 8082)
    emf_tools = []
    try:
        emf_tools_response = requests.get(f"http://{emf_server.host}:8082/tools")
        print(f"Raw EMF /tools response: {emf_tools_response.text}")
        emf_tools_json = emf_tools_response.json()
        if "tools" in emf_tools_json:
            for tool in emf_tools_json["tools"]:
                tool_obj = MCPTool(
                    name=tool.get("name"),
                    description=tool.get("description", ""),
                    endpoint=tool.get("name"),
                    server_name=emf_server.name
                )
                emf_tools.append(tool_obj)
        emf_server.tools = emf_tools
        registry.tools_by_server[emf_server.name] = emf_tools
        print(f"EMF server tool objects registered: {emf_tools}")
    except Exception as e:
        print(f"Could not fetch EMF server tools: {e}")

    # Call ATL server to get enabled transformations
    enabled_transformations = atl_client.call_tool("/transformations/enabled", method="GET")
    print(f"Enabled ATL transformations: {enabled_transformations}")

    # Register transformation tools for ATL server
    print("Registering transformation tools for ATL server...")
    for transfo_data in enabled_transformations:
        transfo_name = transfo_data.get('name')
        # Apply transformation tool
        apply_endpoint = f"/transformation/{transfo_name}/apply"
        apply_tool = MCPTool(
            name=f"apply_{transfo_name}_transformation_tool",
            description=f"Apply transformation {transfo_name}",
            endpoint=apply_endpoint,
            server_name=atl_server.name
        )
        atl_tools.append(apply_tool)
        print(f"Registered transformation tool: {apply_tool}")
        # Info tool
        info_endpoint = f"/transformation/{transfo_name}"
        info_tool = MCPTool(
            name=f"list_transformation_{transfo_name}_tool",
            description=f"Get info for transformation {transfo_name}",
            endpoint=info_endpoint,
            server_name=atl_server.name
        )
        atl_tools.append(info_tool)
        print(f"Registered info tool: {info_tool}")
    atl_server.tools = atl_tools
    registry.tools_by_server[atl_server.name] = atl_tools

    # Call ATL server to get enabled transformations
    enabled_transformations = atl_client.call_tool("/transformations/enabled", method="GET")
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

    user_goal = "Transform this class model '/Users/zakariahachm/Downloads/llm-agents-mde/src/examples/class.xmi' into a relational model"
    print(f"\nAgent user goal: {user_goal}")

    print("\n--- Agent Planning Debug ---")
    plan = agent.plan_workflow(user_goal)
    for i, step in enumerate(plan.steps):
        print(f"Step {i+1}: tool={step.tool_name}, server={step.server_name}, params={step.parameters}, desc={step.description}")
    print("--- End Agent Planning Debug ---\n")

    print("\n--- Agent Execution Debug ---")
    result = agent.executor.execute_workflow(plan)
    print(result)
    print("--- End Agent Execution Debug ---\n")
