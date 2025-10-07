# Simplified Prompt Evolution Experiment - 5 Versions

## Overview
Focused experiment with 5 key versions to show the most impactful prompt components.

## The 5 Key Versions

### **Version 0: Baseline Minimal** ✅ COMPLETED
```
Generate a JSON list of steps for: {user_goal}
```
- **Result**: 0.0% accuracy (0/80)
- **Purpose**: Absolute baseline

### **Version 1: Add JSON Structure**
```
Your goal is: {user_goal}
Generate a workflow plan as a JSON list of steps. Each step must be a JSON object with keys: tool_name, server_name, parameters, description.
```
- **Added**: Clear JSON format and required keys
- **Expected**: ~10-30% (basic structure but no context)

### **Version 2: Add RAG Tools Context**
```
Your goal is: {user_goal}
Relevant tools: {tool_names}
Generate a workflow plan as a JSON list of steps. Each step must be a JSON object with keys: tool_name, server_name, parameters, description.
```
- **Added**: RAG-retrieved relevant tools
- **Expected**: ~40-70% (MAJOR jump - tools available)

### **Version 3: Add Critical Rule (File Path)**
```
Your goal is: {user_goal}
Relevant tools: {tool_names}
Generate a workflow plan as a JSON list of steps. Each step must be a JSON object with keys: tool_name, server_name, parameters, description.
Rules: If you choose an apply_*_transformation_tool, you MUST include parameters.file_path with the absolute path to the input .xmi file.
```
- **Added**: Most critical execution rule
- **Expected**: ~70-90% (MAJOR jump - transformations work)

### **Version 4: Full Optimized (Current Baseline)**
```
You are an MDE agent. Your goal is: {user_goal}
Relevant tools: {tool_names}
Relevant models: {model_names}
Available server names: {available_servers}
Generate a workflow plan as a JSON list of steps. Each step must be a JSON object with keys: tool_name, server_name, parameters, description.
Rules: (1) Use list_transformation_*_tool for info-only queries (parameters can be {}).
(2) If you choose an apply_*_transformation_tool, you MUST include parameters.file_path with the absolute path to the input .xmi file.
(3) Use only the file path that appears in the user goal; do not invent paths.
Output ONLY the JSON list, no extra text.
```
- **Added**: All remaining optimizations
- **Expected**: ~96% (current baseline performance)

## Expected Accuracy Progression
- **V0**: 0.0% → **V1**: ~20% → **V2**: ~60% → **V3**: ~85% → **V4**: ~96%

## Key Insights Expected
1. **V0→V1**: JSON structure helps but limited without context
2. **V1→V2**: RAG tools = BIGGEST jump (context is king)
3. **V2→V3**: File path rule = SECOND biggest jump (execution success)
4. **V3→V4**: Final optimizations = smaller polish improvements

This will clearly show that **context (RAG)** and **execution rules** are the most critical components!