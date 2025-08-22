"""
Workflow Executor - Simple workflow execution
"""
import os
import re
from typing import Dict, Any, List
import time

from core.megamodel import MegamodelRegistry
from mcp.client import ATLServerClient, EMFServerClient
from agents.execution import AgentSession, ExecutionTrace, MCPInvocation
from agents.planning import WorkflowPlan, PlanStep, AgentGoal, StepStatus

class WorkflowExecutor:
    """Simple workflow executor"""
    
    def __init__(self, registry: MegamodelRegistry):
        self.registry = registry
        self.atl_client = ATLServerClient()
        self.emf_client = EMFServerClient()
    
    def execute_step(self, step: PlanStep) -> Dict[str, Any]:
        """Execute a single workflow step using generic HTTP client"""
        step.start_execution()
        start_time = time.time()
        try:
            # Resolve the tool from the registry first, regardless of declared server
            all_tools = self.registry.discover_tools()
            tool_obj = next((t for t in all_tools if getattr(t, "name", "") == step.tool_name), None)
            print(f"[DEBUG] Resolving tool '{step.tool_name}' (declared server={step.server_name}) -> registry match={bool(tool_obj)}")

            # Handle tools that aren't HTTP endpoints
            if step.tool_name == "extract_input_metamodel_name":
                file_path = step.parameters.get("file_path", "") if isinstance(step.parameters, dict) else ""
                result = {"result": f"Requested metamodel extraction for: {file_path}"}
                duration = time.time() - start_time
                step.mark_completed(result)
                return {"step_id": step.step_id, "success": True, "result": result, "duration": duration}

            method = step.parameters.get("method", "POST") if isinstance(step.parameters, dict) else "POST"
            params = step.parameters.get("params") if isinstance(step.parameters, dict) else None
            data = step.parameters.get("data") if isinstance(step.parameters, dict) else None
            files = step.parameters.get("files") if isinstance(step.parameters, dict) else None

            endpoint = None
            server_for_tool = None
            if tool_obj is not None:
                server_for_tool = getattr(tool_obj, "server_name", None) or step.server_name
                ep = getattr(tool_obj, "endpoint", None)
                # endpoint can be a string or MCPEndpoint
                if ep is None:
                    endpoint = None
                elif isinstance(ep, str):
                    endpoint = ep
                else:
                    # Assume MCPEndpoint-like object with 'path'
                    endpoint = getattr(ep, "path", None)
                # If endpoint is missing or equals the tool name, derive from naming convention
                if not endpoint or endpoint == getattr(tool_obj, "name", ""):
                    name = getattr(tool_obj, "name", "")
                    m_apply = re.match(r"^apply_(.+)_transformation_tool$", name)
                    m_info = re.match(r"^list_transformation_(.+)_tool$", name)
                    if m_apply:
                        trans = m_apply.group(1)
                        endpoint = f"/transformation/{trans}/apply"
                        print(f"[DEBUG] Derived endpoint for apply tool '{name}': {endpoint}")
                    elif m_info:
                        trans = m_info.group(1)
                        endpoint = f"/transformation/{trans}"
                        print(f"[DEBUG] Derived endpoint for info tool '{name}': {endpoint}")
                print(f"[DEBUG] Registry endpoint for '{step.tool_name}': {endpoint} (server={server_for_tool})")
            else:
                # As a last resort, try to interpret tool_name as a raw endpoint or derive from naming
                name = step.tool_name or ""
                if isinstance(name, str) and name.startswith('/'):
                    endpoint = name
                    server_for_tool = step.server_name
                else:
                    m_apply = re.match(r"^apply_(.+)_transformation_tool$", name)
                    m_info = re.match(r"^list_transformation_(.+)_tool$", name)
                    if m_apply:
                        trans = m_apply.group(1)
                        endpoint = f"/transformation/{trans}/apply"
                        server_for_tool = step.server_name or "atl_server"
                        print(f"[DEBUG] Derived endpoint from tool name '{name}': {endpoint}")
                    elif m_info:
                        trans = m_info.group(1)
                        endpoint = f"/transformation/{trans}"
                        server_for_tool = step.server_name or "atl_server"
                        print(f"[DEBUG] Derived endpoint from tool name '{name}': {endpoint}")
                print(f"[DEBUG] Fallback endpoint for '{step.tool_name}': {endpoint}")

            if not endpoint:
                result = {"error": f"Could not resolve endpoint for tool: {step.tool_name}"}
            else:
                # Normalize endpoint to start with '/'
                if isinstance(endpoint, str) and not endpoint.startswith('/'):
                    endpoint = f"/{endpoint}"
                # Route to the correct client based on server_for_tool
                if server_for_tool == "atl_server":
                    file_path = step.parameters.get("file_path") if isinstance(step.parameters, dict) else None
                    if file_path and os.path.exists(file_path):
                        with open(file_path, "rb") as f:
                            files = {"IN": f}
                            result = self.atl_client.call_tool(endpoint, method=method, files=files)
                    else:
                        result = self.atl_client.call_tool(endpoint, method=method, params=params, data=data, files=files)
                elif server_for_tool == "emf_server":
                    result = self.emf_client.call_model_tool(endpoint)
                else:
                    # If server is unknown but endpoint looks like an ATL route, send to ATL
                    if isinstance(endpoint, str) and endpoint.startswith("/transformation/"):
                        result = self.atl_client.call_tool(endpoint, method=method, params=params, data=data, files=files)
                    else:
                        result = {"error": f"Unknown server for tool '{step.tool_name}': {server_for_tool}"}

            duration = time.time() - start_time
            if isinstance(result, dict) and "error" in result:
                step.mark_failed(result["error"])
                return {
                    "step_id": step.step_id,
                    "success": False,
                    "error": result["error"],
                    "duration": duration
                }
            else:
                step.mark_completed(result)
                return {
                    "step_id": step.step_id,
                    "success": True,
                    "result": result,
                    "duration": duration
                }
        except Exception as e:
            duration = time.time() - start_time
            step.mark_failed(str(e))
            return {
                "step_id": step.step_id,
                "success": False,
                "error": str(e),
                "duration": duration
            }
    
    def execute_workflow(self, plan: WorkflowPlan) -> Dict[str, Any]:
        """Execute a complete workflow"""
        session = self.registry.create_session()
        session.start()
        trace = session.create_new_trace()
        
        plan.start_execution()
        results = []
        
        try:
            while not plan.check_completion():
                ready_steps = plan.get_ready_steps()
                
                if not ready_steps:
                    break  # No more steps to execute
                
                for step in ready_steps:
                    result = self.execute_step(step)
                    results.append(result)
                    
                    # Add to trace
                    invocation = MCPInvocation(
                        tool_name=step.tool_name,
                        server_name=step.server_name,
                        arguments=step.parameters,
                        result=result.get("result", {}),
                        success=result["success"]
                    )
                    trace.add_invocation(invocation)
                
                plan._update_step_readiness()
            
            session.end()
            return {
                "session_id": session.session_id,
                "status": plan.status.value,
                "results": results,
                "trace_analysis": trace.analyze()
            }
        except Exception as e:
            session.end()
            return {
                "session_id": session.session_id,
                "status": "error",
                "error": str(e),
                "results": results,
                "trace_analysis": trace.analyze()
            }
    
    def create_simple_workflow(self) -> WorkflowPlan:
        """Create a simple generic workflow"""
        goal = AgentGoal(
            description="Generic model transformation workflow",
            success_criteria={"completed": True}
        )
        
        plan = WorkflowPlan(goal=goal)
        
        # Call the ATL server endpoint to get enabled transformations (not a tool name)
        step1 = PlanStep(
            tool_name="/transformations/enabled",
            server_name="atl_server",
            parameters={"method": "GET"},
            description="Get enabled ATL transformations"
        )
        plan.add_step(step1)
        
        return plan