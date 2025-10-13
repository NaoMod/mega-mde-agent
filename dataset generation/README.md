# Dataset Generation Workflow

## Overview

This folder outlines a workflow for generating datasets by leveraging a megamodel repository. The process involves extracting key components, performing analyses, generating insights, sampling APIs, and producing validated instruction–API class pairs.

## Single and Multi-Tool Instructions

The dataset generation produces two types of instruction datasets:

### Single-Tool Instructions (`simple_generate.py`)

- Instructions requiring one tool operation (get or apply)
- Generation process:
  1. Loads available tools from the ATL MCP server
  2. Creates 5 instructions per tool, balanced between "get" and "apply" operations:
     - Tools are classified into "get" and "apply" categories
     - Equal representation of tool types ensures pattern balance
  3. Uses LLM to generate diverse natural language instructions:
     - Variety in phrasing through different seed instructions
     - Multiple formatting styles (natural language, numbered steps, bullet points)
  4. Ensures uniqueness by deduplicating based on instruction text
  5. Default target: 500 instructions

### Multi-Tool Instructions (`multi_simple_generate.py`)

- Instructions requiring a sequence of two tools
- Generation process:
  1. Creates balanced workflow combinations across 4 patterns:
     - get→get (configuration comparison)
     - get→apply (check config, then transform)
     - apply→get (transform, then check config)
     - apply→apply (two transformations in sequence)
  2. Ensures pattern diversity through stratified sampling:
     - Equal distribution of each pattern (120 workflows per pattern)
     - Random shuffling of workflows for generation variety
     - Deduplication of instructions to maximize uniqueness
  3. Default target: 400 instructions

### Dataset Analysis (`analyze_dataset.py`)

- Evaluates dataset diversity metrics, including:
  - Distinct-n scores (lexical diversity): Measures unique n-gram proportions
  - Vocabulary size and lexical density: Quantifies lexical variety
  - Average pairwise distances and dispersion: Captures semantic diversity
  - Cross-similarity with seed instructions: Ensures faithfulness to original seed concepts
  - Pattern distribution analysis: Confirms balanced representation of operation types

## Original Workflow Steps

1. **Initialize**: Start with a megamodel repository and query AI models/capabilities.
2. **Extract**: Pull MCPServers, MCPTools, and MCPCapabilities.
3. **Analyze**:
   - Build a capability graph.
   - Identify historical workflow patterns.
4. **Generate Insights**: Derive insights from capabilities and common patterns.
5. **Sample APIs**: Perform API sampling/selection.
6. **Generate Pairs**: Use an LLM to create instruction–API class pairs.
7. **Validate & Output**: Validate results and produce the final dataset.

## Files

- `pipeline.py`: Orchestrator with stubbed steps for workflow execution.
- `simple_generate.py`: Generates single-tool instructions
- `multi_simple_generate.py`: Generates multi-tool instructions
- `analyze_dataset.py`: Analyzes dataset diversity metrics
- `outputs/`: Directory containing generated datasets and artifacts