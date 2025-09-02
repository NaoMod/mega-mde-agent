import sys
import os
import requests

# Add src to path FIRST
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.megamodel import MegamodelRegistry
from core.am3 import ReferenceModel, TransformationModel
from mcp.integrator import MCPServerIntegrator
from mcp.client import ATLServerClient, EMFServerClient
from agents.agent import MCPAgent
from mcp.infrastructure import MCPServer, MCPTool

def populate_registry(registry):
    print("Populating MegamodelRegistry with ATL/EMF servers, tools, and transformations...")
    integrator = MCPServerIntegrator(registry)
    atl_server = integrator.setup_atl_server()
    emf_server = integrator.setup_emf_server()
    print(f"ATL Server registered: {atl_server.name}")
    print(f"EMF Server registered: {emf_server.name}")
    print(f"ATL Server tools: {atl_server.tools}")
    print(f"EMF Server tools: {emf_server.tools}")

    # Fetch and register tools for ATL server
    atl_tools = []
    atl_client = ATLServerClient(base_url=f"http://{atl_server.host}:{atl_server.port}", tools_port=atl_server.tools_port)
    try:
        atl_tools_response = atl_client.get_tools()
        print(f"Raw ATL /tools response: {atl_tools_response}")
        if "tools" in atl_tools_response:
            for tool in atl_tools_response["tools"]:
                tool_name = tool.get("name")
                desc = tool.get("description", "")
                # If tool is an apply transformation tool, set endpoint to REST endpoint
                if tool_name.startswith("apply_") and tool_name.endswith("_transformation_tool"):
                    transfo_name = tool_name[len("apply_"):-len("_transformation_tool")]
                    endpoint = f"/transformation/{transfo_name}/apply"
                elif tool_name.startswith("list_transformation_") and tool_name.endswith("_tool"):
                    transfo_name = tool_name[len("list_transformation_"):-len("_tool")]
                    endpoint = f"/transformation/{transfo_name}"
                else:
                    endpoint = tool_name
                tool_obj = MCPTool(
                    name=tool_name,
                    description=desc,
                    endpoint=endpoint,
                    server_name=atl_server.name
                )
                atl_tools.append(tool_obj)
        atl_server.tools = atl_tools
        registry.tools_by_server[atl_server.name] = atl_tools
        print(f"ATL server tool objects registered: {atl_tools}")
    except Exception as e:
        print(f"Could not fetch ATL server tools: {e}")

    # Fetch and register tools for EMF server
    emf_tools = []
    emf_client = EMFServerClient(base_url=f"http://{emf_server.host}:{emf_server.port}", tools_port=emf_server.tools_port)
    try:
            emf_stateless_base = "http://localhost:8082"
            resp = requests.get(f"{emf_stateless_base}/tools", timeout=5)
            resp.raise_for_status()
            tools_payload = resp.json().get("tools", [])
            emf_stateless_tools = []
            emf_stateless_server = MCPServer(name="emf_stateless_server", base_url=emf_stateless_base)
            for t in tools_payload:
                tool_name = t.get("name")
                desc = t.get("description", "")
                if not tool_name:
                    continue
                emf_stateless_tools.append(MCPTool(
                    name=tool_name,
                    description=desc,
                    endpoint=tool_name,          # endpoint is the MCP tool name (planning use)
                    server_name=emf_stateless_server.name
                ))
            emf_stateless_server.tools = emf_stateless_tools
            registry.tools_by_server[emf_stateless_server.name] = emf_stateless_tools
            print(f"Registered {len(emf_stateless_tools)} tools from EMF stateless server ({emf_stateless_base})")
    except Exception as e:
            print(f"Warning: failed to discover EMF stateless tools: {e}")

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
