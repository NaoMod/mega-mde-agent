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
