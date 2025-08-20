import logging
import re
import sys
import os
import json
import subprocess
from mcp.server.fastmcp import FastMCP
from typing import Dict, Any, List

# Constants
ATL_SERVER_BASE = "http://localhost:8080"

# Set up logging to stderr as required by MCP
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger('atl_mcp_server')

# Initialize the MCP server
mcp = FastMCP("atl")

def fetch_transformations() -> list:
    """Fetch enabled transformations from the ATL server."""
    result = subprocess.run(['curl', '-X', 'GET', 'localhost:8080/transformations/enabled'], 
                          capture_output=True, text=True, check=True)
    return json.loads(result.stdout)

def get_transformation_details(transformation_name: str) -> Dict[str, Any]:
    """Get details of a specific transformation."""
    result = subprocess.run(['curl', '-X', 'GET', 'localhost:8080/transformations/enabled'], 
                          capture_output=True, text=True, check=True)
    transformations = json.loads(result.stdout)
    return next((t for t in transformations if t["name"] == transformation_name), None)

def create_transformation_description(transformation_name: str) -> str:
    """Create a description for the apply transformation tool."""
    transformation = get_transformation_details(transformation_name)
    if not transformation:
        return f"Transformation {transformation_name} not found"
    
    input_path = transformation["input_metamodels"][0]["path"]
    input_model = input_path.split("/")[-1].replace(".ecore", "")
    
    output_path = transformation["output_metamodels"][0]["path"]
    output_model = output_path.split("/")[-1].replace(".ecore", "")
    
    return f"Input metamodel: {input_model}, Output metamodel: {output_model}. This tool takes: {input_model} model as input and produces {output_model} model."

def generate_get_tool_description(transformation_name: str) -> str:
    """Create a description for the get transformation tool."""
    transformation = get_transformation_details(transformation_name)
    if not transformation:
        return f"Transformation {transformation_name} not found"
    
    input_path = transformation["input_metamodels"][0]["path"]
    input_model = input_path.split("/")[-1].replace(".ecore", "")
    
    output_path = transformation["output_metamodels"][0]["path"]
    output_model = output_path.split("/")[-1].replace(".ecore", "")
    
    return f"This tool displays the details of the transformation {transformation_name} that transforms: {input_model} model into {output_model} model."

def _extract_from_content(content: str) -> str:
    """Extract metamodel name from XMI file content."""
    simple_xmlns_pattern = r'xmlns="([^"]*)"'
    match = re.search(simple_xmlns_pattern, content)
    if match:
        return match.group(1)

    complex_xmlns_pattern = r'xmlns:(\w+)="([^"]*)"'
    matches = re.findall(complex_xmlns_pattern, content)
    
    filtered_matches = [
        (prefix, url) for prefix, url in matches 
        if not prefix in ['xmi', 'xsi']
    ]
    
    if filtered_matches:
        metamodel_prefix = filtered_matches[0][0]
        root_pattern = rf'<{metamodel_prefix}:\w+'
        if re.search(root_pattern, content):
            return metamodel_prefix.upper()

    return None

@mcp.tool(name="extract_input_metamodel_name", description="Extracts the metamodel name from an XMI file. The input should be exactly a file path to an XMI file. Returns the metamodel name (like 'Class', 'Grafcet', 'ECORE', or 'KM3').")
async def get_input_metamodel(file_path: str) -> str:
    """Extract the metamodel name from an XMI file."""
    try:
        file_path = str(file_path).strip()
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        metamodel_name = _extract_from_content(content)
        return f"Input metamodel name: {metamodel_name}" if metamodel_name else "Could not extract metamodel name from the file."
    except FileNotFoundError:
        return f"Error: File not found at path: {file_path}"
    except Exception as e:
        return f"An error occurred while processing the file: {str(e)}"

def get_transformation_names() -> List[str]:
    """Get list of enabled transformation names from the ATL server."""
    result = subprocess.run(['curl', '-X', 'GET', 'localhost:8080/transformations/enabled'], 
                          capture_output=True, text=True, check=True)
    transformations = json.loads(result.stdout)
    return [t["name"] for t in transformations]

# Create dynamic tools for each transformation
transformations = fetch_transformations()
for t in transformations:
    name = t["name"]
    
    def create_apply_transformation(trans_name: str):
        @mcp.tool(name=f"apply_{trans_name}_transformation_tool", description=create_transformation_description(trans_name))
        async def apply_transformation(file_path: str) -> str:
            """Apply an ATL transformation to a model file."""
            try:
                # Handle dictionary input by taking the first value
                if isinstance(file_path, dict):
                    file_path = next(iter(file_path.values()), '')
                
                file_path = str(file_path).strip()
                if not os.path.exists(file_path):
                    return f"Error: File not found at {file_path}"
                
                transformation_name = trans_name
                command = [
                    'curl', 
                    f'localhost:8080/transformation/{transformation_name}/apply', 
                    '-F', f'IN=@{file_path}'
                ]
                result = subprocess.run(command, capture_output=True, text=True, check=True)
                return f"Transformation {transformation_name} applied:\n{result.stdout}"
            except subprocess.CalledProcessError as e:
                return f"An error occurred while applying transformation: {e.stderr}"
            except Exception as e:
                return f"An error occurred: {str(e)}"
        return apply_transformation

    def create_get_transformation(trans_name: str):
        @mcp.tool(name=f"list_transformation_{trans_name}_tool", description=generate_get_tool_description(trans_name))
        async def get_transformation_info() -> str:
            """Get details about a specific ATL transformation."""
            try:
                transformation_name = trans_name
                command = ['curl', '-X', 'GET', f'localhost:8080/transformation/{transformation_name}']
                result = subprocess.run(command, capture_output=True, text=True, check=True)
                return f"The transformation '{transformation_name}':\n{result.stdout} successfully fetched."
            except subprocess.CalledProcessError as e:
                return f"An error occurred while fetching the transformation: {e.stderr}"
        return get_transformation_info

    # Register the tools with the correct closure
    create_apply_transformation(name)
    create_get_transformation(name)

if __name__ == "__main__":
    try:
        mcp.run(transport='stdio')
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        sys.exit(1) 