import json
import litellm
import re
import csv
import os
import fitz
from retrival import load_tree, extract_structure_for_prompt

# Open the PDF globally
pdf_doc = fitz.open("UCIE_1.1.pdf")

# Litellm config
litellm.api_base = "http://localhost:8080/v1"
litellm.api_key = "sk-no-key-required"
MODEL = "openai/gemma-4-31b"

TEST_MODE = True
TEST_NODE_ID = "0035" # Die-to-Die Adapter

def clean_json_response(response_text):
    # Extract JSON from markdown blocks if present
    match = re.search(r'```json(.*?)```', response_text, re.DOTALL)
    if match:
        response_text = match.group(1)
    
    # Strip non-json characters (sometimes LLMs add conversational text)
    start_idx = response_text.find('[')
    if start_idx == -1:
        start_idx = response_text.find('{')
    
    end_idx = response_text.rfind(']')
    if end_idx == -1:
        end_idx = response_text.rfind('}')
        
    if start_idx != -1 and end_idx != -1:
        response_text = response_text[start_idx:end_idx+1]
        
    return response_text.strip()

def run_llm_json(prompt, max_tokens=8000):
    response = litellm.completion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
        extra_body={
            "reasoning_budget": 2000,
            "reasoning_format": "none"
        }
    )
    raw_content = response.choices[0].message.content.strip()
    cleaned = clean_json_response(raw_content)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}")
        print(f"Raw Output: {raw_content}")
        return None

def extract_features(text, chapter_title):
    print(f"\n[Stage 1] Extracting Features for {chapter_title}...")
    prompt = f"""You are an expert UVM Verification Engineer.
Analyze the following specification text and identify all distinctly testable features, sub-modules, and capabilities.

Output a JSON array of objects. Each object must have exactly these keys:
- "Feature ID": A unique identifier (e.g., "D2D-001")
- "Feature Name": The high-level name of the feature
- "Sub Feature": The specific sub-feature or operation mode

Specification Text:
{text}

Return ONLY the JSON array. Do not include markdown formatting or explanations.
"""
    return run_llm_json(prompt)

def extract_vp_fields(text, feature):
    print(f"  -> Extracting detailed VP fields for feature: {feature['Feature Name']} ({feature['Sub Feature']})")
    
    fields = [
        "Spec Version", "Requirement Location", "Description", "Verification Goal", 
        "Test Type", "Coverage Method", "UVM Components Involved", "Pass/Fail Criteria", 
        "Sequences Coverage", "Link to Register Coverage", "Traceability", 
        "Test Case Coverage", "Target Register and Justification", "Link to Coverage", 
        "Testcase Range", "Testcase Design", "Testcase Steps", "Testcase Inputs", 
        "Testcase Outputs", "Constraints", "Randomization Constraints and Rationale", 
        "Sequence Implementation Notes", "Scoreboard and Checker", "Link to PDF Coverage"
    ]
    
    prompt = f"""You are an expert UVM Verification Engineer.
Based on the provided specification text, generate detailed verification plan fields for the following feature.

Feature ID: {feature.get('Feature ID')}
Feature Name: {feature.get('Feature Name')}
Sub Feature: {feature.get('Sub Feature')}

Generate a JSON object containing exactly these keys: {json.dumps(fields)}.
If a field is not explicitly mentioned in the text but is a standard UVM requirement (e.g., UVM Components Involved, Scoreboard and Checker), infer it based on standard UVM practices for this type of hardware block. If a field cannot be inferred, write "N/A".

Specification Text:
{text}

Return ONLY the JSON object. Do not include markdown formatting or explanations.
"""
    result = run_llm_json(prompt)
    if result:
        # Merge with base feature info
        return {**feature, **result}
    return feature

def extract_ports_registers(text, chapter_title):
    print(f"\n[Stage 3] Extracting Ports and Registers for {chapter_title}...")
    prompt = f"""You are an expert hardware engineer.
Analyze the following specification text and extract any defined Ports, Signals, or Registers.

Output a JSON array of objects. Each object should have keys like:
- "Type": "Port" or "Register" or "Signal"
- "Name": The name of the port/register
- "Width/Size": Bit width or size
- "Direction": Input/Output/Inout (for ports)
- "Description": Brief description

If no ports or registers are found, return an empty array [].

Specification Text:
{text}

Return ONLY the JSON array.
"""
    return run_llm_json(prompt) or []

def generate_test_cases(vp_details):
    print(f"  -> Generating UVM Test Cases for feature: {vp_details.get('Feature Name')}...")
    
    fields = [
        "Feature ID", "Feature Name", "Sub Feature", "TestCaseName", "Scenario",
        "Traceability.TestName", "Traceability.SequenceName", "Traceability.AgentName",
        "Traceability.CovergroupName", "Covergroup Definition", "Target Register and Justification",
        "Testcase Range Min", "Testcase Range Mid", "Testcase Range Max", "Testcase Design",
        "Testcase Steps", "Testcase Step Description", "Testcase Inputs", "Testcase Outputs",
        "Expected Outputs Description", "Constraints", "Randomization Constraints",
        "Sequence Notes", "Spec Text", "ScoreboardChecks", "New Checkers"
    ]
    
    prompt = f"""You are an expert UVM Verification Engineer.
Based on the following Verification Plan details for a feature, formulate 1 to 3 concrete UVM test cases.

Feature ID: {vp_details.get('Feature ID')}
Feature Name: {vp_details.get('Feature Name')}
Sub Feature: {vp_details.get('Sub Feature')}
Verification Goal: {vp_details.get('Verification Goal')}
Constraints: {vp_details.get('Constraints')}

Output a JSON array of testcase objects. Each object must have exactly these keys: {json.dumps(fields)}.
If a field is not explicitly mentioned or applicable, write "N/A". Make sure to populate fields like TestCaseName, Scenario, Testcase Steps, ScoreboardChecks, etc. accurately based on the feature.

Return ONLY the JSON array.
"""
    return run_llm_json(prompt) or []

def save_csv(filename, data_list):
    if not data_list:
        print(f"No data to save for {filename}")
        return
        
    # Get all unique keys across all dictionaries
    keys = set()
    for d in data_list:
        keys.update(d.keys())
    # Ensure some ordering
    keys = sorted(list(keys))
    
    with open(filename, 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys, restval='N/A', extrasaction='ignore')
        dict_writer.writeheader()
        dict_writer.writerows(data_list)
    print(f"Saved {len(data_list)} rows to {filename}")

def get_node_text(node):
    text = ""
    start = node.get('start_index')
    end = node.get('end_index')
    
    if start is not None and end is not None:
        # fitz is 0-indexed, start/end are 1-indexed (printed pages)
        for i in range(start - 1, end):
            if i < len(pdf_doc):
                text += pdf_doc[i].get_text() + "\n"
                
    # Also grab text from children
    for child in node.get('nodes', []):
        text += get_node_text(child) + "\n"
        
    return text

def main():
    tree = load_tree("./results/UCIE_1.1_structure.json")
    nodes = tree.get('structure', tree) if isinstance(tree, dict) else tree
    
    if TEST_MODE:
        print(f"=== Running in TEST MODE for Node {TEST_NODE_ID} ===")
        # Find the node recursively
        def find_node(nodes_list, target):
            for n in nodes_list:
                if n.get('node_id') == target:
                    return n
                if 'nodes' in n:
                    found = find_node(n['nodes'], target)
                    if found: return found
            return None
        target_node = find_node(nodes, TEST_NODE_ID)
        target_nodes = [target_node] if target_node else []
    else:
        # In FULL MODE, we would iterate over all major chapters
        target_nodes = nodes
        
    all_vp_rows = []
    all_ports_rows = []
    all_test_cases = []
    
    for node in target_nodes:
        chapter_title = node.get('title', 'Unknown')
        node_id = node.get('node_id')
        
        text = get_node_text(node)
        if not text.strip():
            print(f"Skipping empty text for node {node_id}")
            continue
            
        # Stage 1: Feature Extraction
        features = extract_features(text, chapter_title)
        if not features:
            continue
            
        if TEST_MODE:
            print(f"Limiting to first 2 features for TEST_MODE...")
            features = features[:2]
            
        # Stage 2 & 4: Detailed VP and Test Cases
        for feature in features:
            vp_row = extract_vp_fields(text, feature)
            all_vp_rows.append(vp_row)
            
            test_cases = generate_test_cases(vp_row)
            all_test_cases.extend(test_cases)
            
        # Stage 3: Ports and Registers
        ports = extract_ports_registers(text, chapter_title)
        all_ports_rows.extend(ports)
        
    # Save CSVs
    print("\n[Saving Output Files]")
    save_csv("verification_plan.csv", all_vp_rows)
    save_csv("ports_and_registers.csv", all_ports_rows)
    save_csv("test_cases.csv", all_test_cases)

if __name__ == "__main__":
    main()
