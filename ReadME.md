# LLM Agents for Model-Driven Engineering: A Megamodel-Based Approach

## Abstract

![Generic Dataset Overview](images/Overview_generic_dataset.drawio%20(1).png)

### Context

Agentification of MDE tasks is increasing as practitioners seek to automate modeling workflows.
MDE environments contain diverse models, metamodels, transformations, and tools that require coordination.
Current approaches lack a central repository to store information about model relationships, transformation traces, and execution contexts.
This information is needed for agents to understand available resources and plan workflows effectively.

### Challenge
MDE environments need systematic repositories to expose model information, relationships, and execution traces to autonomous agents.
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
