


import os
import requests

class ATLServerClient:
    """Client for ATL server using HTTP requests"""
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url

    def call_tool(self, endpoint: str, method: str = "GET", params=None, data=None, files=None):
        url = f"{self.base_url}{endpoint}"
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
        response.raise_for_status()
        return response.json() if response.headers.get('Content-Type', '').startswith('application/json') else response.text

class EMFServerClient:
    """Client for EMF server using HTTP requests"""
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url

    def call_model_tool(self, endpoint: str, method: str = "GET", data=None, files=None):
        url = f"{self.base_url}{endpoint}"
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
        response.raise_for_status()
        return response.json() if response.headers.get('Content-Type', '').startswith('application/json') else response.text
    