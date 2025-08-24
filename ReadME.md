# Intelligent Model-Driven Engineering Agent

This project creates an intelligent agent that helps automate Model-Driven Engineering (MDE) tasks using Large Language Models. The agent can understand natural language instructions, plan the necessary steps, and execute model transformations automatically.

## What Does It Do?

The agent serves as a bridge between your instructions and various MDE tools. When you tell it what you want to accomplish with your models, it:

1. Understands your goal
2. Finds the relevant tools and models
3. Creates a step-by-step plan
4. Executes the plan using available servers
5. Keeps track of everything that happens

## How It Works

The system is built around three main components:

1. A central registry (megamodel) that keeps track of:
   - Available tools and what they can do
   - Different types of models
   - Transformation rules
   - Server connections

2. A planning system that:
   - Takes your instructions in plain English
   - Matches them with available tools
   - Creates a workflow plan

3. An execution engine that:
   - Runs the planned steps
   - Coordinates between different servers
   - Records what was done

## A Simple Example

Let's say you want to transform a Class model into a Relational database model. Here's what happens:

1. You tell the agent: "transform Class model to Relational model"

2. The agent analyzes your request and:
   - Looks for tools related to "class", "model", "transform", and "relational"
   - Finds the relevant model types (Class and Relational models)
   - Identifies the servers that can help

3. The agent creates a plan like:
   ```json
   [
     {
       "tool_name": "loadClassModel",
       "server_name": "emf_server",
       "parameters": {"model_path": "input/class.xmi"},
       "description": "Load the Class model"
     },
     {
       "tool_name": "transformToRelational",
       "server_name": "atl_server",
       "parameters": {"source": "class.xmi", "target": "relational.xmi"},
       "description": "Transform Class to Relational"
     }
   ]
   ```

4. The agent integrates with a separate WorkflowExecutor system:
Plans are converted into WorkflowPlan objects with PlanStep instances
Each step specifies: tool_name, server_name, parameters, and description
The executor handles the actual tool invocation and result collection

## Project Structure

The code is organized in a way that separates different concerns:

### Core (`src/core/`)
- `am3.py` - Defines what models and transformations are
- `megamodel.py` - Keeps track of all available tools and models

### Server Communication (`src/mcp/`)
- `infrastructure.py` - Defines how to talk to different servers
- `integrator.py` - Connects to ATL and EMF servers
- `client.py` - Handles sending commands to servers

### Agent Logic (`src/agents/`)
- `planning.py` - Creates step-by-step plans
- `execution.py` - Keeps track of what's being done
- `workflow.py` - Runs the plans

### Examples
- `src/examples/main.py` - Shows how to use the system

---

## Key Components

### Models and Transformations

Each model in the system has:
- A type (like Class or Relational)
- A format (like XMI)
- Rules about how it can be used

Transformations are tools that:
- Take one type of model as input
- Produce another type of model as output
- Follow specific rules about how to convert between them

### Servers

The system works with two types of servers:
- EMF servers that handle loading and saving models
- ATL servers that perform model transformations

### The Agent's Brain

The agent makes decisions by:
1. Understanding what you want to do
2. Looking through available tools
3. Matching your goal with the right tools
4. Creating a sequence of steps
5. Running those steps in order

## System Features

The agent can:
- Figure out which transformations will work for your models
- Create multi-step plans when needed
- Work with different types of servers
- Keep track of everything it does
- Learn from successful operations

## Visual Overview

Below is a diagram showing how the different parts of the system connect:

![System Architecture](images/megamodel_without_relationships.png)
