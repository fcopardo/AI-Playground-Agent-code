from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

# Setup the model using langchain-ollama wrapper
# Using a low temperature ensures the model stays factual and strict
model = ChatOllama(model="qwen2.5-coder:7b", temperature=0)

# Define the State of our Graph
class AgentState(TypedDict):
    task: str
    code: str
    feedback: str
    iterations: int

# Node 1: The Coder
def coder_node(state: AgentState):
    print("\n[Node: Coder] Writing code...")
    prompt = f"Task: {state['task']}. Previous feedback: {state.get('feedback', 'None')}"

    # Ask model to write code
    response = model.invoke([
        ("system", "Write clean Python code to accomplish the task. Return ONLY code inside ```python blocks."),
        ("user", prompt)
    ])

    return {
        "code": response.content,
        "iterations": state.get("iterations", 0) + 1
    }

# Node 2: The Reviewer (Safety / Quality Gate)
def reviewer_node(state: AgentState):
    print("[Node: Reviewer] Reviewing code structure...")
    code = state["code"]

    # Basic algorithmic check + LLM feedback
    if "import os" not in code and "env" in state["task"].lower():
        return {"feedback": "Fail: You forgot to import the 'os' module for environment variables."}

    if "```python" not in code:
        return {"feedback": "Fail: Please enclose your code cleanly inside a ```python block."}

    return {"feedback": "Pass"}

# Conditional Routing Logic
def should_continue(state: AgentState):
    if state["feedback"] == "Pass" or state["iterations"] >= 3:
        print("[Router] Code passed inspection or limit reached. Ending process.")
        return END
    else:
        print(f"[Router] Code failed inspection. Re-routing back to Coder. (Attempt {state['iterations']}/3)")
        return "coder"

# Build the Graph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("coder", coder_node)
workflow.add_node("reviewer", reviewer_node)

# Set Entry Point
workflow.set_entry_point("coder")

# Connect Nodes with Edges
workflow.add_edge("coder", "reviewer")
workflow.add_conditional_edges("reviewer", should_continue)

# Compile the Graph
app = workflow.compile()

# Execute the Agent
if __name__ == "__main__":
    initial_state = {
        "task": "Create a function to read an env variable named API_KEY.",
        "code": "",
        "feedback": "",
        "iterations": 0
    }

    final_output = app.invoke(initial_state)
    print("\n=== FINAL AGENT RESULT ===")
    print(final_output["code"])
