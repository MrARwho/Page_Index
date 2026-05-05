import json
import litellm

# Point to your local llama.cpp server
litellm.api_base = "http://localhost:8080/v1"
litellm.api_key = "sk-no-key-required"
MODEL = "openai/gemma-4-31b" # Prefix tells LiteLLM to use OpenAI API format

def load_tree(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_structure_for_prompt(nodes, depth=0):
    """Creates a lightweight 'Table of Contents' with summaries for the LLM to read."""
    toc_text = ""
    for node in nodes:
        indent = "  " * depth
        summary = node.get('summary', 'No summary available.')
        toc_text += f"{indent}- [Node ID: {node.get('node_id')}] {node.get('title')}: {summary}\n"
        
        # Recursively get children
        if 'nodes' in node and node['nodes']:
            toc_text += extract_structure_for_prompt(node['nodes'], depth + 1)
    return toc_text

def extract_all_text_from_node(node):
    """Recursively extracts all text from a node and its children."""
    text = node.get('text', '') + "\n"
    if 'nodes' in node and node['nodes']:
        for child in node['nodes']:
            text += extract_all_text_from_node(child)
    return text

def get_node_text(nodes, target_id):
    """Finds a specific Node ID and extracts all its text (including children)."""
    for node in nodes:
        if node.get('node_id') == target_id:
            return extract_all_text_from_node(node)
        if 'nodes' in node and node['nodes']:
            result = get_node_text(node['nodes'], target_id)
            if result:
                return result
    return ""

def query_vectorless_rag(tree_data, question):
    print(f"\n--- Question: {question} ---")
    
    # STEP 1: Build the Structure Prompt
    print("\n[1/3] Generating Tree Search Prompt...")
    
    # Check if tree_data has a 'structure' key (as produced by page_index)
    nodes_data = tree_data.get('structure', tree_data) if isinstance(tree_data, dict) else tree_data
    toc_text = extract_structure_for_prompt(nodes_data)
    
    search_prompt = f"""You are an elite hardware architecture assistant. 
Review the following document structure and summaries. 
Identify the relevant 'Node ID's that contain the answer to this question: "{question}"

Document Structure:
{toc_text}

Respond ONLY with a JSON array of string Node IDs. For example: ["0012", "0014"]. Do not output anything else."""

    # STEP 2: LLM Reasons and Selects the Node(s)
    print("[2/3] LLM is reasoning over the document structure...")
    response = litellm.completion(
        model=MODEL,
        messages=[{"role": "user", "content": search_prompt}],
        temperature=0.0,
        extra_body={
        "reasoning_budget": 1000, # Limit reasoning to 500 tokens
        "reasoning_format": "none" # Do not return the <think> blocks in the response
    }
    )
    
    import re
    content = response.choices[0].message.content.strip()
    match = re.search(r'\[.*\]', content, re.DOTALL)
    target_node_ids = []
    if match:
        try:
            target_node_ids = json.loads(match.group(0))
        except Exception:
            pass
            
    print(f"      -> LLM selected Node IDs: {target_node_ids}")
    
    # STEP 3: Extract Text and Generate Final Answer
    print(f"[3/3] Extracting raw text from Nodes {target_node_ids} and generating answer...")
    
    raw_text = ""
    for target_id in target_node_ids:
        raw_text += f"\n--- Section {target_id} ---\n"
        raw_text += get_node_text(nodes_data, target_id)
    
    if not raw_text.strip():
        return "Error: Could not find text for the selected nodes. The LLM might have hallucinated the IDs."
        
    answer_prompt = f"""Based strictly on the following technical specification text, answer the user's question.

Text:
{raw_text}

Question: {question}
Answer:"""

    final_response = litellm.completion(
        model=MODEL,
        messages=[{"role": "user", "content": answer_prompt}],
        temperature=0.0,
        extra_body={
        "reasoning_budget": 1000, # Limit reasoning to 500 tokens
        "reasoning_format": "none" # Do not return the <think> blocks in the response
    }
    )
    
    return final_response.choices[0].message.content

if __name__ == "__main__":
    # Point this to the JSON file you just successfully generated!
    tree = load_tree("./results/UCIE_1.1_structure.json")
    
    # Ask your question
    question = '''Instructions:

    Extract Features Only: Identify every distinct module, hardware block, or functional capability mentioned in the specification.

    Keep it Concise: Do not include detailed behavioral descriptions, interfaces, register mappings, or edge cases.

    Be Exhaustive: Ensure no minor features, sub-modules, or distinct operational modes are missed.

Output Format: Provide a clean, bulleted list containing only the names of the identified features and blocks. You may use nested bullets if a major block contains specific sub-features.'''
    
    final_answer = query_vectorless_rag(tree, question)
    
    print("\n================ FINAL ANSWER ================\n")
    print(final_answer)
    print("\n==============================================")