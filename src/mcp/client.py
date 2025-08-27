

import os
import requests

class ATLServerClient:
    """Client for ATL server using HTTP requests"""
    def __init__(self, base_url: str = "http://localhost:8080", tools_port: int = 8081):
        self.base_url = base_url
        self.tools_port = tools_port
        
    def get_tools(self):
        """Get list of tools from the tools port"""
        tools_url = f"{self.base_url.rsplit(':', 1)[0]}:{self.tools_port}/tools"
        # DEBUG: outgoing request
        print(f"[HTTP] GET {tools_url}")
        response = requests.get(tools_url)
        print(f"[HTTP] -> {response.status_code} {response.reason}")
        response.raise_for_status()
        return response.json()

    def call_tool(self, endpoint: str, method: str = "GET", params=None, data=None, files=None):
        url = f"{self.base_url}{endpoint}"
        # DEBUG: outgoing request summary
        file_keys = list((files or {}).keys()) if isinstance(files, dict) else []
        print(f"[HTTP] {method.upper()} {url} params={bool(params)} data={bool(data)} files={file_keys}")
        if method == "GET":
            response = requests.get(url, params=params)
        elif method == "POST":
            response = requests.post(url, params=params, data=data, files=files)
        elif method == "PUT":
            response = requests.put(url, params=params, data=data, files=files)
        elif method == "DELETE":
            response = requests.delete(url, params=params)
        else:
            raise ValueError(f"Unsupported method: {method}")
        print(f"[HTTP] -> {response.status_code} {response.reason}")
        response.raise_for_status()
        return response.json() if response.headers.get('Content-Type', '').startswith('application/json') else response.text

class EMFServerClient:
    """Client for EMF server using HTTP requests"""
    def __init__(self, base_url: str = "http://localhost:8080", tools_port: int = 8082):
        self.base_url = base_url
        self.tools_port = tools_port
        
    def get_tools(self):
        """Get list of tools from the tools port"""
        tools_url = f"{self.base_url.rsplit(':', 1)[0]}:{self.tools_port}/tools"
        # DEBUG: outgoing request
        print(f"[HTTP] GET {tools_url}")
        response = requests.get(tools_url)
        print(f"[HTTP] -> {response.status_code} {response.reason}")
        response.raise_for_status()
        return response.json()

    def call_model_tool(self, endpoint: str, method: str = "GET", data=None, files=None):
        url = f"{self.base_url}{endpoint}"
        # DEBUG: outgoing request summary
        file_keys = list((files or {}).keys()) if isinstance(files, dict) else []
        print(f"[HTTP] {method.upper()} {url} data={bool(data)} files={file_keys}")
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, data=data, files=files)
        elif method == "PUT":
            response = requests.put(url, data=data, files=files)
        elif method == "DELETE":
            response = requests.delete(url)
        else:
            raise ValueError(f"Unsupported method: {method}")
        print(f"[HTTP] -> {response.status_code} {response.reason}")
        response.raise_for_status()
        return response.json() if response.headers.get('Content-Type', '').startswith('application/json') else response.text
    