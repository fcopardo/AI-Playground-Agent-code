import ollama
import chromadb
import os

# 1. Setup Local Vector DB for RAG
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="local_docs")

# Seed some dummy documentation for RAG simulation
collection.add(
    documents=["To initialize our internal logging system, always use internal_logger.init(level='DEBUG').",
               "The database connection string must be fetched using os.getenv('DB_URI')."],
    ids=["doc1", "doc2"]
)

MODEL = "qwen2.5-coder:7b"

def agent_write_code(prompt, use_rag=False):
    context = ""

    # Optional RAG step
    if use_rag:
        print("🔍 Querying local RAG database...")
        results = collection.query(query_texts=[prompt], n_results=1)
        if results['documents'][0]:
            context = f"\nUse this local documentation context:\n{results['documents'][0][0]}\n"

    system_prompt = (
        "You are a strict, local coding assistant. Return ONLY valid Python code block. "
        "Do not include explanations, markdown outside the code block, or conversational filler."
    )

    user_content = f"{context}Task: {prompt}"

    print("🤖 Agent is thinking...")
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    )

    return response['message']['content']

# Example Run
if __name__ == "__main__":
    task = "Write a python script that connects to a database and sets up internal logging."

    # Running with RAG enabled
    generated_code = agent_write_code(task, use_rag=True)

    print("\n--- Generated Code ---")
    print(generated_code)

    # Safe output: Save to a specific folder instead of executing it blindly
    os.makedirs("./output", exist_ok=True)
    with open("./output/generated_script.py", "w") as f:
        f.write(generated_code)
    print("\n💾 Code safely saved to ./output/generated_script.py")
