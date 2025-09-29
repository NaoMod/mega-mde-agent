# LLM Agents for Model-Driven Engineering: A Megamodel-Based Approach

## Abstract

![Generic Dataset Overview](images/Overview_generic_dataset.drawio%20(1).png)

### Context

Agentification of MDE tasks is increasing as practitioners seek to automate modeling workflows.
MDE environments contain diverse models, metamodels, transformations, and tools that require coordination.
Current approaches lack a central repository to store information about model relationships, transformation traces, and execution contexts.
This information is needed for agents to understand available resources and plan workflows effectively.

### Challenge
MDE environments need systematic repositories to expose modeling informations to agents.
Existing megamodels capture static relationships but do not track dynamic agent executions.

### Solution
We extend the AM3 megamodel to serve as a repository containing model relationships, transformation metadata, and agent execution traces.
The MegamodelRegistry maintains entities, MCP servers, sessions, and workflow plans in a unified structure.
LLM agents use this repository to retrieve relevant tools and models.

### Results
We evaluate our approach through three phases.
First, we generate datasets based on MCP servers registered in the megamodel.
Second, we collect historical traces from agent executions present in the megamodel on these datasets.
Third, we perform analysis on the execution traces to identify performance metrics.
This analysis enables iterative agent improvements based on real execution data.

## 1. Introduction

Model-Driven Engineering workflows require coordination between heterogeneous tools and transformations. Traditional approaches rely on manual orchestration by domain experts who understand both the modeling languages and tool capabilities. The rise of Large Language Models presents opportunities to automate these workflows through agents.

Model-Driven Engineering workflows require coordination between heterogeneous tools and transformations. Traditional approaches rely on manual orchestration by domain experts who understand both the modeling languages and tool capabilities. The rise of Large Language Models presents opportunities to automate these workflows through agents.

Current MDE environments lack systematic approaches for exposing model information to autonomous agents. A megamodel is a repository that captures relationships between models, metamodels, and transformations within an MDE ecosystem. Existing megamodels document static relationships between models and metamodels but do not capture dynamic execution traces from tool interactions. This limitation prevents agents from accessing comprehensive information about modeling workflows and execution history.


This paper presents an extended AM3 megamodel that serves as a repository for agent-driven MDE workflows. Our approach integrates the Model Context Protocol to enable standardized communication with modeling servers. We demonstrate the effectiveness of this approach through automated dataset generation, execution and trace analysis.

## 2. Architecture Overview

Figure 1 illustrates the core components of our megamodel-based approach. The architecture consists of four main layers that work together to support agent-driven MDE workflows.

### 2.1 MegamodelRegistry

The MegamodelRegistry forms the central repository of our approach. It extends the traditional AM3 megamodel to include agent execution traces and tool interaction data. The registry maintains several key components:

**Modeling artifacts**: The registry stores AM3 entities including models, metamodels, and transformations as core AM3 entities. Each entity maintains relationships with other entities through directed relationships that capture dependencies, conformance, and transformation links. Models conform to metamodels, transformations operate on specific model types, and reference models provide metamodel definitions. The registry tracks terminal models that represent instantiated models. ( add the views description) 

**Tooling artefacts**: This component maintains a catalog of Model Context Protocol (MCP) servers that provide transformation capabilities through standardized interfaces. Each MCP server exposes a set of tools that can perform specific modeling operations such as model validation, transformation execution, or code generation. The registry tracks which tools are available on each server and maintains resource mappings that specify the input and output requirements for each tool. This enables automatic tool selection based on the current modeling context and required operations.

**Agent artefacts**: The registry stores agent-specific components including agent goals, workflow plans, and plan steps. Agent goals define high-level objectives that drive the modeling process, while workflow plans decompose these goals into executable sequences of actions. Each plan step specifies the tools to invoke and the input artifacts required. The registry maintains the relationship between goals and plans to support plan reuse and adaptation across different modeling scenarios.

**Execution traces artefacts**: This component captures the runtime behavior of agent-driven modeling processes. Execution traces record the sequence of tool invocations, parameter values, and transformation outcomes for each modeling session. Trace steps document individual operations including timing information, success status, and error messages. Agent sessions group related traces and maintain session-specific context information. This execution history enables process analysis, debugging, and workflow optimization.

### 3. Use Case: Agent-Based Regression Testing

To demonstrate the practical applicability of our megamodel-based approach, we present a use case from an industrial MDE environment where a software development company employs agents for automated regression testing of their model transformation workflows.

---

### 3.1 Scenario Context

The company maintains a complex MDE toolchain that processes various modeling artifacts through multiple transformation steps. As their modeling workflows evolve, they need to ensure that changes to transformations, metamodels, or tooling do not break existing functionality.  

Traditional manual regression testing is time-consuming and error-prone, especially when dealing with hundreds of model variants and transformation combinations.

The company decided to implement **agent-driven regression testing** that can automatically validate their transformation workflows. Their goal is to detect regressions early in the development cycle while maintaining test coverage across their MDE ecosystem.

---

### 3.2 Three-Phase Implementation

The megamodel enables this regression testing scenario through our **three-phase approach**:

#### Phase 1 - Test Dataset Generation
The company uses the **MegamodelRegistry** to automatically generate regression test scenarios based on their registered MCP servers and workflow patterns.  

The dataset generation process analyzes this information to create  test scenarios that cover:
- Different combinations of input model types and transformation sequences  
- Single and multi tools instructions
- The corresponding API calls for each instruction.



We start by querying the megamodel repository to obtain the available MCP tools from MCP servers. Each tool is described by its name and description, which are used as the basis for instruction generation.

We categorize the instructions into two groups according to the number of tools they involve:

Single-tool instructions – Generated by combining one tool’s description and set of single-tool seeds with a single-tool prompt template. The LLM then produces pairs of $<$instruction, API call$>$.

Multi-tool instructions – Generated by sampling multiple tools (e.g., two) plus the mulri-tool seeds along with the company workflow to be tested. The LLM generates instructions requiring multiple API calls.

Both categories of seeds are written by experts to ensure ground truth correctness. This step guarantees that the generated instructions align with the intended tool behavior.

The validated subsets (single-tool and multi-tool) are then merged into the final dataset.
![Regression testing dataset generation](images/Overview_generic_dataset.drawio.png)

#### Phase 2 - Automated Execution and Trace Collection
The **MCPAgent** executes the generated test scenarios against the current version of their MDE tools.  
Each test execution produces detailed traces that capture:
- Tool selection decisions made by the agent for each scenario   
- Success/failure outcomes with detailed error information when applicable  
- Transformation results  

These traces are automatically stored in the **MegamodelRegistry**, creating a record of the current system behavior that can be compared against other execution versions.

#### Phase 3 - Regression Analysis
The company analyzes the collected execution traces to identify potential regressions by comparing current results against historical baselines stored in the megamodel.  

The analysis includes:
- Performance regression detection through timing comparisons  
- Functional regression identification by comparing transformation outputs  
- Tool selection consistency analysis to detect unexpected agent behavior changes  
- Error pattern analysis to identify new failure modes  

This analysis enables the company to quickly identify when changes to their MDE infrastructure introduce regressions, allowing them to address issues before they impact production workflows.


## 5. Conclusion

This project presents a megamodel-based approach for supporting LLM agents in Model-Driven Engineering workflows. Our extended AM3 megamodel serves as a repository that enables agents to understand available tools, plan transformation workflows, and learn from execution history.

The approach demonstrates that systematic information management enables autonomous agents to operate effectively in complex MDE environments. The integration of Model Context Protocol provides standardized interfaces that support tool interoperability and agent adaptability.

We plan to evaluate the system on larger industrial case studies to validate scalability and practical applicability.
