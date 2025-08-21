import sys
import os

# Add src to path FIRST
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.megamodel import MegamodelRegistry
from core.am3 import ReferenceModel, TransformationModel
from mcp.integrator import MCPServerIntegrator
from mcp.client import ATLServerClient
import requests

def main():
    print("Initializing MegamodelRegistry and populating with ATL server enabled transformations and metamodels...")
    registry = MegamodelRegistry()

    # Setup MCP servers
    integrator = MCPServerIntegrator(registry)
    atl_server = integrator.setup_atl_server()
    emf_server = integrator.setup_emf_server()
    print(f"ATL Server registered: {atl_server.name}")
    print(f"EMF Server registered: {emf_server.name}")


    # Fetch and register tools for ATL server (port 8081)
    atl_tools = []
    atl_client = ATLServerClient(base_url=f"http://{atl_server.host}:{atl_server.port}")
    try:
        atl_tools_response = requests.get(f"http://{atl_server.host}:8081/tools")
        atl_tools_json = atl_tools_response.json()
        if "tools" in atl_tools_json:
            for tool in atl_tools_json["tools"]:
                atl_tools.append(tool["name"])
        atl_server.tools = atl_tools
    except Exception as e:
        print(f"Could not fetch ATL server tools: {e}")

    # Fetch and register tools for EMF server (port 8082)
    emf_tools = []
    try:
        emf_tools_response = requests.get(f"http://{emf_server.host}:8082/tools")
        emf_tools_json = emf_tools_response.json()
        if "tools" in emf_tools_json:
            for tool in emf_tools_json["tools"]:
                emf_tools.append(tool["name"])
        emf_server.tools = emf_tools
    except Exception as e:
        print(f"Could not fetch EMF server tools: {e}")

    # Call ATL server to get enabled transformations
    enabled_transformations = atl_client.call_tool("/transformations/enabled", method="GET")
    #print(f"Fetched {len(enabled_transformations)} enabled transformations from ATL server.")

    # Extract and register all transformations and metamodels
    metamodel_uris = set()
    for transfo_data in enabled_transformations:
        # Register input metamodels
        input_mms = transfo_data.get('input_metamodels', [])
        for mm in input_mms:
            uri = mm.get('path')
            name = mm.get('name', uri)
            if uri and uri not in metamodel_uris:
                registry.register_entity(ReferenceModel(uri=uri, name=name))
                metamodel_uris.add(uri)
                #print(f"Registered input metamodel: {uri}")
        # Register output metamodels
        output_mms = transfo_data.get('output_metamodels', [])
        for mm in output_mms:
            uri = mm.get('path')
            name = mm.get('name', uri)
            if uri and uri not in metamodel_uris:
                registry.register_entity(ReferenceModel(uri=uri, name=name))
                metamodel_uris.add(uri)
                #print(f"Registered output metamodel: {uri}")
        # Register transformation
        # Try to link source/target metamodels if possible
        source_mm = input_mms[0]['path'] if input_mms else None
        target_mm = output_mms[0]['path'] if output_mms else None
        source_ref = registry.get_entity(source_mm) if source_mm else None
        target_ref = registry.get_entity(target_mm) if target_mm else None
        registry.register_entity(TransformationModel(
            uri=transfo_data.get('atlFile', transfo_data.get('name', 'unknown')),
            name=transfo_data.get('name', 'unknown'),
            source_metamodel=source_ref,
            target_metamodel=target_ref
        ))

if __name__ == "__main__":
    main()
