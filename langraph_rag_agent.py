from typing import TypedDict
import chromadb
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

# 1. Initialize Local Vector Database (ChromaDB)
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="local_coding_docs")

# Seed the vector database with mock custom framework / internal instructions
collection.add(
    documents=[
        "Internal rule: When connecting to any database, you must use the custom wrapper 'SecureDBConnection()'.",
        "Security rule: Never hardcode api tokens. Always pull them from os.environ.get('APP_TOKEN')."
    ],
    ids=["rule_db", "rule_token"]
)

# 2. Initialize our Local 7B LLM
model = ChatOllama(model="qwen2.5-coder:7b", temperature=0)

# 3. Define the State Object
class AgentState(TypedDict):
    task: str
    rag_context: str  # This holds the retrieved internal documentation
    code: str
    feedback: str
    iterations: int

# --- Graph Nodes ---

def rag_retrieval_node(state: AgentState):
    """Queries the local vector DB using the user's task instruction."""
    print("\n🔍 [Node: RAG Retrieval] Searching local vector database...")

    # Query ChromaDB using the task string
    results = collection.query(query_texts=[state["task"]], n_results=1)

    # Extract documents if found
    context_str = ""
    if results and results['documents'] and results['documents'][0]:
        context_str = "\n".join(results['documents'][0])
        print(f"📌 Found relevant internal docs:\n   -> {context_str}")
    else:
        print("⚠️ No relevant local documentation found.")

    return {"rag_context": context_str}


def coder_node(state: AgentState):
    """Generates the code using both the task prompt AND the RAG context."""
    print("\n💻 [Node: Coder] Writing code...")

    system_prompt = (
        "You are a strict, local coding assistant. Write clean Python code. "
        "You must strictly respect the provided Local Context Rules if applicable. "
        "Return ONLY the code block enclosed in ```python and ```."
    )

    # Injecting retrieved RAG context dynamically into the LLM prompt
    user_prompt = f"""
    Context/Rules to follow:
    {state['rag_context'] if state['rag_context'] else 'No special internal rules found.'}

    Task: {state['task']}
    Previous Reviewer Feedback: {state.get('feedback', 'None')}
    """

    response = model.invoke([
        ("system", system_prompt),
        ("user", user_prompt)
    ])

    return {
        "code": response.content,
        "iterations": state.get("iterations", 0) + 1
    }


def reviewer_node(state: AgentState):
    """Validates if the code satisfies safety standards and structural formats."""
    print("🔬 [Node: Reviewer] Reviewing code syntax and rules...")
    code = state["code"]

    # Algorithmic fallback validation rule matching our RAG data
    if "SecureDBConnection" not in code and "database" in state["task"].lower():
        return {"feedback": "Fail: You missed the custom internal framework rule! You must use 'SecureDBConnection()' to connect."}

    if "```python" not in code:
        return {"feedback": "Fail: Ensure your complete output is wrapped inside a ```python block."}

    return {"feedback": "Pass"}


# --- Graph Routing Logic ---

def should_continue(state: AgentState):
    if state["feedback"] == "Pass" or state["iterations"] >= 3:
        print("\n🏁 [Router] Criteria met or iterations exhausted. Processing completed.")
        return END
    else:
        print(f"\n🔄 [Router] Review failed. Sending back to Coder (Attempt {state['iterations']}/3)")
        return "coder"


# --- Constructing the State Machine Graph ---

workflow = StateGraph(AgentState)

# Register the nodes
workflow.add_node("rag_extractor", rag_retrieval_node)
workflow.add_node("coder", coder_node)
workflow.add_node("reviewer", reviewer_node)

# Set the linear flow starting at RAG
workflow.set_entry_point("rag_extractor")
workflow.add_edge("rag_extractor", "coder")  # Flows into Coder next
workflow.add_edge("coder", "reviewer")       # Flows into Reviewer next

# Apply the loop routing rule on the Reviewer node
workflow.add_conditional_edges("reviewer", should_continue)

# Compile the execution pipeline
app = workflow.compile()

# --- Execute Agent ---
if __name__ == "__main__":
    # Test case designed to trigger the database retrieval rule seeded in Chroma
    initial_state = {
        "task": "Write a python function to initialize a database configuration.",
        "rag_context": "",
        "code": "",
        "feedback": "",
        "iterations": 0
    }

    final_output = app.invoke(initial_state)
    print("\n================== FINAL AGENT CODE ==================")
    print(final_output["code"])
