from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import requests


class ServerStatus(Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"

class HttpMethod(Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"

@dataclass
class MCPCapability:
    """Capability exposed by MCP server"""
    input_types: List[str]
    output_types: List[str] 
    can_execute: bool = True
    description: str = ""

@dataclass
class MCPEndpoint:
    """MCP API endpoint"""
    path: str
    method: HttpMethod
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def build_url(self, base_url: str, **path_params) -> str:
        """Build complete URL with path parameters"""
        url = f"{base_url}{self.path}"
        for key, value in path_params.items():
            url = url.replace(f"{{{key}}}", str(value))
        return url

@dataclass
class MCPTool:
    """Tool available through MCP server"""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    endpoint: Optional[MCPEndpoint] = None
    server_name: str = ""
    
    def get_parameter_info(self) -> Dict[str, Any]:
        """Get information about tool parameters"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "server": self.server_name
        }

@dataclass
class MCPResource:
    """Resource managed by MCP server"""
    uri: str
    resource_type: str = "model"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPServer:
    """MCP Server representation"""
    host: str
    port: int
    name: str = ""
    status: ServerStatus = ServerStatus.DISCONNECTED
    capabilities: List[MCPCapability] = field(default_factory=list)
    tools: List[MCPTool] = field(default_factory=list)
    resources: List[MCPResource] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.name:
            self.name = f"server_{self.host}_{self.port}"
    

    
    def connect(self) -> bool:
        """Test connection to the MCP server"""
        try:
            #check in the documetation how to ping the server
            response = requests.get(f"{self.endpoint_url}/health", timeout=5)
            if response.status_code == 200:
                self.status = ServerStatus.CONNECTED
                return True
            else:
                self.status = ServerStatus.ERROR
                return False
        except Exception:
            self.status = ServerStatus.ERROR
            return False
    
    def add_tool(self, tool: MCPTool) -> None:
        """Add a tool to this server"""
        tool.server_name = self.name
        self.tools.append(tool)
    
    def get_tool(self, tool_name: str) -> Optional[MCPTool]:
        """Get a tool by name"""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None
    
    def list_tool_names(self) -> List[str]:
        """Get list of all tool names"""
        return [tool.name for tool in self.tools]
    
    def add_capability(self, capability: MCPCapability) -> None:
        """Add a capability to this server"""
        self.capabilities.append(capability)
    
    def supports_input_type(self, input_type: str) -> bool:
        """Check if server supports a specific input type"""
        for capability in self.capabilities:
            if input_type in capability.input_types:
                return True
        return False
    
    def supports_output_type(self, output_type: str) -> bool:
        """Check if server supports a specific output type"""
        for capability in self.capabilities:
            if output_type in capability.output_types:
                return True
        return False
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get comprehensive server information"""
        return {
            "name": self.name,
            "host": self.host,
            "port": self.port,
            "status": self.status.value,
            "endpoint_url": self.endpoint_url,
            "tools_count": len(self.tools),
            "tools": [tool.name for tool in self.tools],
            "capabilities_count": len(self.capabilities),
            "capabilities": [
                {
                    "input_types": cap.input_types,
                    "output_types": cap.output_types,
                    "can_execute": cap.can_execute
                } for cap in self.capabilities
            ],
            "resources_count": len(self.resources),
            "metadata": self.metadata
        }