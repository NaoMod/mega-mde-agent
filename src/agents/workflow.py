"""
Workflow Executor - Simple workflow execution
"""
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
        """Execute a single workflow step"""
        step.start_execution()
        start_time = time.time()
        
        try:
            # Route to appropriate client
            if step.server_name == "atl_server":
                result = self.atl_client.call_transformation_tool(step.tool_name, **step.parameters)
            elif step.server_name == "emf_server":
                result = self.emf_client.call_model_tool(step.tool_name, **step.parameters)
            else:
                result = {"error": f"Unknown server: {step.server_name}"}
            
            duration = time.time() - start_time
            
            if "error" in result:
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
            trace.end()
            
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
                "status": "failed",
                "error": str(e)
            }
    
    def create_simple_workflow(self) -> WorkflowPlan:
        """Create a simple generic workflow"""
        goal = AgentGoal(
            description="Generic model transformation workflow",
            success_criteria={"completed": True}
        )
        
        plan = WorkflowPlan(goal=goal)
        
        # Simple steps - will be populated based on actual use case
        step1 = PlanStep(
            tool_name="extract_input_metamodel_name",
            server_name="atl_server",
            parameters={"file_path": "input.xmi"},
            description="Extract metamodel name"
        )
        plan.add_step(step1)
        
        return plan