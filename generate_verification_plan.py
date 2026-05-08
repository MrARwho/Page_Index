import json
import litellm
import re
import os
import csv
import sys
import fitz
from concurrent.futures import ThreadPoolExecutor, as_completed
from retrival import load_tree, extract_structure_for_prompt

# ── LiteLLM Configuration ─────────────────────────────────────────────────────
litellm.api_base = "http://localhost:8080/v1"
litellm.api_key = "sk-no-key-required"

try:
    from pageindex.utils import ConfigLoader
    loader = ConfigLoader()
    config = vars(loader.load())
except Exception:
    config = {}

MODEL = config.get('vp_model', 'openai/gemma-4-31b')
if_validate_plans = config.get('if_validate_plans', True)
citation_format = config.get('citation_format', 'quote+page')
max_cross_ref_paragraphs = config.get('max_cross_ref_paragraphs', 5)
max_workers = config.get('max_workers', 8)
test_mode = os.environ.get('TEST_MODE', 'no').lower() in ('yes', '1', 'true')
test_node_id = os.environ.get('TEST_NODE_ID', '0006')
test_max_features = int(os.environ.get('TEST_MAX_FEATURES', '2'))

# Open the PDF globally
PDF_PATH = "UCIE_1.1.pdf"
pdf_doc = fitz.open(PDF_PATH)

# ── JSON Helpers ──────────────────────────────────────────────────────────────

def _truncate_text(text, max_chars=12000):
    """Truncate text to max_chars while preserving complete paragraphs."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Cut at last paragraph boundary to avoid half-paragraphs
    last_para = truncated.rfind('\n\n')
    if last_para > max_chars // 2:
        truncated = truncated[:last_para]
    else:
        last_space = truncated.rfind(' ')
        if last_space > max_chars // 2:
            truncated = truncated[:last_space]
    return truncated + '\n\n... [truncated]'

def _escape_newlines_in_strings(text):
    """Walk through text and escape literal newlines/carriage-returns inside JSON strings."""
    result = []
    i = 0
    in_string = False
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == '\\' and i + 1 < len(text):
                result.append(ch)
                result.append(text[i + 1])
                i += 2
                continue
            elif ch == '"':
                in_string = False
                result.append(ch)
            elif ch == '\n' or ch == '\r':
                if ch == '\r' and i + 1 < len(text) and text[i + 1] == '\n':
                    result.append('\\r\\n')
                    i += 2
                    continue
                result.append('\\n')
            else:
                result.append(ch)
        else:
            if ch == '"':
                in_string = True
                result.append(ch)
            else:
                result.append(ch)
        i += 1
    return ''.join(result)


def clean_json_response(response_text):
    # Strip markdown code blocks
    match = re.search(r'```(?:json)?(.*?)```', response_text, re.DOTALL)
    if match:
        response_text = match.group(1)
    
    # Find the outermost JSON array or object
    start_idx = response_text.find('[')
    if start_idx == -1:
        start_idx = response_text.find('{')
    end_idx = response_text.rfind(']')
    if end_idx == -1:
        end_idx = response_text.rfind('}')
        
    if start_idx != -1 and end_idx != -1:
        response_text = response_text[start_idx:end_idx+1]
    
    # Escape literal newlines/carriage-returns inside JSON strings
    response_text = _escape_newlines_in_strings(response_text)
    
    # Remove trailing commas before } or ]
    response_text = re.sub(r',\s*([}\]])', r'\1', response_text)
    # Replace Python True/False/None with JSON equivalents
    response_text = re.sub(r'\bTrue\b', 'true', response_text)
    response_text = re.sub(r'\bFalse\b', 'false', response_text)
    response_text = re.sub(r'\bNone\b', 'null', response_text)
        
    return response_text.strip()


def run_llm_json(prompt, max_tokens=8000, temperature=0.0, extra_kwargs=None):
    kwargs = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra_body": {
            "reasoning_budget": 2000,
            "reasoning_format": "none"
        }
    }
    if extra_kwargs:
        kwargs["extra_body"].update(extra_kwargs)
    
    response = litellm.completion(**kwargs)
    raw_content = response.choices[0].message.content.strip()
    cleaned = clean_json_response(raw_content)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"  [WARN] Failed to decode JSON: {e}")
        print(f"  [RAW] {raw_content[:500]}")
        return None


def find_node_in_tree(nodes_list, target_id):
    """Recursively find a node by node_id in the tree."""
    for node in nodes_list:
        if node.get('node_id') == target_id:
            return node
        if 'nodes' in node:
            found = find_node_in_tree(node['nodes'], target_id)
            if found:
                return found
    return None


def get_node_text(node):
    """Extract raw text from a node's PDF pages (0-indexed internal, 1-indexed externally)."""
    text = ""
    start = node.get('start_index')
    end = node.get('end_index')
    
    if start is not None and end is not None:
        for i in range(start - 1, end):
            if i < len(pdf_doc):
                text += pdf_doc[i].get_text() + "\n"
    
    for child in node.get('nodes', []):
        text += get_node_text(child) + "\n"
        
    return text


def get_node_text_with_line_numbers(node):
    """Extract text with line numbers for citation support."""
    result = []
    start = node.get('start_index')
    end = node.get('end_index')
    
    if start is not None and end is not None:
        for page_idx in range(start - 1, end):
            if page_idx < len(pdf_doc):
                page = pdf_doc[page_idx]
                text = page.get_text()
                lines = text.split('\n')
                page_num = page_idx + 1  # 1-indexed
                for line_num, line in enumerate(lines):
                    stripped = line.strip()
                    if stripped:
                        result.append({
                            'page': page_num,
                            'line': line_num,
                            'content': stripped
                        })
    
    for child in node.get('nodes', []):
        result.extend(get_node_text_with_line_numbers(child))
    
    return result


def find_citation(text_with_lines, quote):
    """Find page and line number for a given quote."""
    if not quote or not text_with_lines:
        return None
    
    quote_norm = quote.strip().lower()
    
    for entry in text_with_lines:
        content_norm = entry['content'].strip().lower()
        if quote_norm == content_norm:
            return {'page': entry['page'], 'line': entry['line'], 'content': entry['content']}
        if quote_norm in content_norm:
            return {'page': entry['page'], 'line': entry['line'], 'content': entry['content']}
    
    return None


def format_citation(citation):
    """Format citation based on config."""
    if not citation:
        return "Not found in spec text"
    
    page = citation['page']
    line = citation['line']
    content = citation['content']
    
    if citation_format == 'quote+page':
        return f'"{content}" (page {page}, line {line})'
    elif citation_format == 'quote+line':
        return f'"{content}" (line {line})'
    else:
        return f'"{content}" (page {page}, line {line})'


# ── Phase 0: Global Register/Port Extraction ─────────────────────────────────

PHASE_0_HEADER = """You are an expert hardware engineer.
Analyze the following specification text and extract all defined Ports, Signals, or Registers.

Output a JSON array of objects. Each object must have EXACTLY these keys:
- "Type": "Port", "Register", or "Signal"
- "Name": The name of the port/register/signal
- "Width/Size": Bit width or size (or "N/A")
- "Direction": "Input", "Output", "Inout", or "N/A"
- "Description": Brief description of purpose
- "Reset_Value": Reset value if register (or "N/A")
- "Citation": Exact quote from text with page/line reference

Rules:
- Extract EVERY port, signal, and register mentioned, even if briefly described
- If a register field is described, create a separate entry for it
- Do NOT merge or deduplicate - extract everything as-is
- For Citation, quote the exact text and include page/line reference

Specification Text:
{text}

Return ONLY the JSON array. Do not include markdown formatting or explanations."""


def phase_0_extract_registers_ports(node, chapter_title):
    """Extract all registers, ports, and signals from a node."""
    print(f"  [Phase 0] Extracting registers/ports for node {node.get('node_id')}: {chapter_title}")
    
    text = get_node_text(node)
    if not text.strip():
        return []
    
    text = _truncate_text(text, max_chars=10000)
    prompt = PHASE_0_HEADER.format(text=text)
    result = run_llm_json(prompt, max_tokens=4000)
    
    if isinstance(result, list):
        return result
    return []


# ── Phase 1: Feature Extraction with Cross-Reference Flagging ────────────────

PHASE_1_FEATURE_PROMPT = """You are an expert UVM Verification Engineer.
Analyze the following specification text and identify ALL distinctly testable features, sub-modules, and capabilities.

Output a JSON array of objects. Each object must have EXACTLY these keys:
- "Feature ID": Unique identifier (e.g., "D2D-001")
- "Feature Name": High-level name of the feature
- "Sub Feature": Specific sub-feature or operation mode
- "Description": Brief description of what this feature does
- "_references": Array of cross-references to OTHER sections/chapters (empty if none)

For each cross-reference, use this format:
{{
  "reference_type": "section" or "module" or "register" or "protocol",
  "reference_name": "Name of the referenced item",
  "reference_context": "Why this reference matters for this feature (1-2 sentences)",
  "_quote": "Exact quote from the text that references the external item",
  "_page": Page number,
  "_line": Line number
}}

Rules:
- Extract EVERY testable feature, capability, or sub-module mentioned
- If a feature is described across multiple paragraphs, include it once with all key aspects in Description
- Be EXHAUSTIVE - do not skip minor features or edge cases
- For each feature, check if it references anything in another section (e.g., "uses the PHY layer", "see Section 4.2", "configured via REG_CTRL")
- If it does, include it in the _references array with the exact quote
- Do NOT extract the referenced content itself - just flag the reference

Specification Text:
{text}

Return ONLY the JSON array. Do not include markdown formatting or explanations."""


def _truncate_text(text, max_chars=12000):
    """Truncate text to max_chars while preserving complete paragraphs."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Cut at last paragraph boundary to avoid half-paragraphs
    last_para = truncated.rfind('\n\n')
    if last_para > max_chars // 2:
        truncated = truncated[:last_para]
    else:
        last_space = truncated.rfind(' ')
        if last_space > max_chars // 2:
            truncated = truncated[:last_space]
    return truncated + '\n\n... [truncated]'


def phase_1_extract_features(node, chapter_title):
    """Extract features and flag cross-references from a node."""
    print(f"  [Phase 1] Extracting features for node {node.get('node_id')}: {chapter_title}")
    
    text = get_node_text(node)
    if not text.strip():
        return {'features': [], 'text': '', 'node_id': node.get('node_id'), 'text_with_lines': []}
    
    text_with_lines = get_node_text_with_line_numbers(node)
    
    text = _truncate_text(text, max_chars=15000)
    prompt = PHASE_1_FEATURE_PROMPT.format(text=text)
    result = run_llm_json(prompt, max_tokens=8000)
    
    if not isinstance(result, list):
        return {'features': [], 'text': text, 'node_id': node.get('node_id'), 'text_with_lines': text_with_lines}
    
    # Validate and clean references
    for feature in result:
        refs = feature.get('_references', [])
        if isinstance(refs, list):
            for ref in refs:
                if isinstance(ref, dict):
                    # Ensure required keys exist
                    ref.setdefault('reference_type', 'unknown')
                    ref.setdefault('reference_name', 'Unknown')
                    ref.setdefault('reference_context', '')
                    ref.setdefault('_quote', '')
                    ref.setdefault('_page', 0)
                    ref.setdefault('_line', 0)
    
    return {
        'features': result,
        'text': text,
        'node_id': node.get('node_id'),
        'text_with_lines': text_with_lines
    }


# ── Phase 1.5: Cross-Reference Resolution ────────────────────────────────────

def resolve_cross_references(all_phase1_results):
    """
    Resolve cross-references between nodes.
    For each feature that references another node, find the relevant text in that node.
    
    Returns: dict mapping (source_node_id, feature_index) -> list of resolved paragraph contexts
    """
    print(f"\n[Phase 1.5] Resolving cross-references...")
    
    # Collect all unique references
    references_to_resolve = []  # (source_node_id, source_feature_idx, ref)
    node_map = {r['node_id']: r for r in all_phase1_results}
    
    for source_id, result in node_map.items():
        for idx, feature in enumerate(result['features']):
            refs = feature.get('_references', [])
            for ref in refs:
                if ref.get('_quote', '').strip():
                    references_to_resolve.append((source_id, idx, ref))
    
    if not references_to_resolve:
        print("  No cross-references to resolve.")
        return {}
    
    print(f"  Found {len(references_to_resolve)} cross-references to resolve.")
    
    # Group by target node to batch processing
    target_nodes_needed = set()
    for source_id, feature_idx, ref in references_to_resolve:
        target_id = ref.get('_target_node_id')
        if target_id:
            target_nodes_needed.add(target_id)
    
    # Also infer target from reference context if no explicit node ID
    # (We need to search the tree structure for matching section names)
    resolved_map = {}  # (source_id, feature_idx) -> list of resolved contexts
    
    for source_id, feature_idx, ref in references_to_resolve:
        target_id = ref.get('_target_node_id')
        ref_name = ref.get('reference_name', '')
        ref_context = ref.get('reference_context', '')
        ref_quote = ref.get('_quote', '')
        
        if not target_id:
            # Try to find target node by matching reference_name against tree structure
            target_id = _infer_target_node(ref_name, ref_context)
        
        if not target_id or target_id == source_id:
            # Can't resolve, skip
            continue
        
        # Get the text from the target node
        target_result = node_map.get(target_id)
        if not target_result:
            continue
        
        target_text = target_result['text']
        target_lines = target_result['text_with_lines']
        
        # Find the relevant paragraph(s) by matching the quote or key terms
        resolved_paragraphs = _find_relevant_paragraphs(
            target_text, target_lines, ref_quote, ref_name, max_paragraphs=max_cross_ref_paragraphs
        )
        
        if resolved_paragraphs:
            key = (source_id, feature_idx)
            if key not in resolved_map:
                resolved_map[key] = []
            resolved_map[key].extend(resolved_paragraphs)
    
    print(f"  Resolved {len(resolved_map)} cross-references.")
    return resolved_map


def _infer_target_node(ref_name, ref_context):
    """Try to find the target node ID by matching reference name against tree structure."""
    try:
        tree = load_tree("./results/UCIE_1.1_structure.json")
        nodes = tree.get('structure', tree) if isinstance(tree, dict) else tree
        
        ref_lower = ref_name.lower()
        ctx_lower = ref_context.lower()
        
        def search(nodes_list):
            best_match = None
            best_score = 0
            for node in nodes_list:
                title = node.get('title', '').lower()
                node_id = node.get('node_id', '')
                summary = node.get('summary', '').lower()
                
                # Score by how well ref_name matches title or summary
                score = 0
                if ref_lower in title or title in ref_lower:
                    score = 10
                elif ref_lower in summary or summary in ref_lower:
                    score = 5
                elif ref_lower in ctx_lower:
                    # Check if node title appears in context
                    if node.get('title', '').lower() in ctx_lower:
                        score = 8
                
                if score > best_score:
                    best_score = score
                    best_match = node_id
                
                if 'nodes' in node:
                    sub_match = search(node['nodes'])
                    if sub_match and best_score == 0:
                        best_match = sub_match
            return best_match
        
        return search(nodes)
    except Exception:
        return None


def _find_relevant_paragraphs(target_text, target_lines, ref_quote, ref_name, max_paragraphs=5):
    """Find the most relevant paragraphs from target text based on quote or key terms."""
    if not ref_quote:
        return []
    
    # Try to find the quote in the target text
    quote_norm = ref_quote.strip().lower()
    
    # Find matching line(s)
    matching_indices = []
    for i, entry in enumerate(target_lines):
        if quote_norm in entry['content'].strip().lower():
            matching_indices.append(i)
            break  # Take first match
    
    if not matching_indices:
        # Fallback: find lines containing key terms from ref_name
        key_terms = ref_name.lower().split()
        for i, entry in enumerate(target_lines):
            content_lower = entry['content'].strip().lower()
            if any(term in content_lower for term in key_terms if len(term) > 3):
                matching_indices.append(i)
                break
    
    if not matching_indices:
        return []
    
    # Extract paragraph around the match
    paragraphs = []
    for match_idx in matching_indices:
        # Get surrounding lines (paragraph context: ~10 lines before and after)
        context_start = max(0, match_idx - 10)
        context_end = min(len(target_lines), match_idx + 10)
        
        context_lines = []
        for j in range(context_start, context_end):
            context_lines.append(f"  Page {target_lines[j]['page']}, Line {target_lines[j]['line']}: {target_lines[j]['content']}")
        
        paragraphs.append({
            'text': '\n'.join(context_lines),
            'source_node_id': None  # Will be filled by caller
        })
        
        if len(paragraphs) >= max_paragraphs:
            break
    
    return paragraphs


# ── Phase 2: VP Generation with Merged Cross-Reference Context ───────────────

VP_FIELDS = [
    "Spec Version", "Requirement Location", "Description", "Verification Goal",
    "Test Type", "Coverage Method", "UVM Components Involved", "Pass/Fail Criteria",
    "Sequences Coverage", "Link to Register Coverage", "Traceability",
    "Test Case Coverage", "Target Register and Justification", "Link to Coverage",
    "Testcase Range", "Testcase Design", "Testcase Steps", "Testcase Inputs",
    "Testcase Outputs", "Constraints", "Randomization Constraints and Rationale",
    "Sequence Implementation Notes", "Scoreboard and Checker", "Link to PDF Coverage"
]


def _build_merged_text(feature, source_text, source_lines, resolved_contexts):
    """Build the text context for VP generation, including resolved cross-references."""
    merged_parts = []
    merged_lines = []
    
    # Original source text (truncated to leave room for LLM output)
    merged_parts.append(f"=== Original Section Text ===\n{_truncate_text(source_text, max_chars=10000)}")
    
    # Add resolved cross-reference contexts
    if resolved_contexts:
        for i, ctx in enumerate(resolved_contexts):
            merged_parts.append(f"=== Resolved Cross-Reference {i+1} ===\n{ctx['text']}")
    
    merged_text = '\n\n'.join(merged_parts)
    
    # Also build merged line list for citation
    merged_lines.extend(source_lines)
    # Cross-ref contexts don't need line numbers in their embedded text
    # (the embedded text already has page/line references)
    
    return merged_text, merged_lines


PHASE_2_VP_PROMPT = """You are an expert UVM Verification Engineer.
Generate a detailed verification plan for the following feature based on the provided specification text.

Feature ID: {feature_id}
Feature Name: {feature_name}
Sub Feature: {sub_feature}
Description: {description}

Generate a JSON object containing EXACTLY these keys: {fields}.

For each field, also include a "_citation" key with the exact quote from the spec text and its page/line reference.
Example structure for each field:
{{
    "Field Name": "Your answer here",
    "_citation": "Exact quote from spec"  // include "page": X, "line": Y if possible
}}

Rules:
- If a field is not explicitly mentioned in the text but is a standard UVM requirement (e.g., UVM Components Involved, Scoreboard and Checker), infer it based on standard UVM practices for this type of hardware block
- If a field cannot be inferred, write "N/A"
- For _citation, ALWAYS include an exact quote from the specification text that supports your answer
- If no direct quote exists for a field, write "_citation": "Inferred from standard UVM practice - no direct quote in spec"
- Be thorough and specific. The verification plan will be used by verification engineers.

Specification Text:
{text}

Return ONLY the JSON object. Do not include markdown formatting or explanations."""


def phase_2_generate_vp(feature, source_text, source_lines, resolved_contexts):
    """Generate VP fields for a single feature with cross-reference context."""
    feature_id = feature.get('Feature ID', 'Unknown')
    feature_name = feature.get('Feature Name', 'Unknown')
    sub_feature = feature.get('Sub Feature', 'Unknown')
    description = feature.get('Description', '')
    
    print(f"    [Phase 2] Generating VP for feature: {feature_name} ({sub_feature})")
    
    merged_text, merged_lines = _build_merged_text(feature, source_text, source_lines, resolved_contexts)
    
    prompt = PHASE_2_VP_PROMPT.format(
        feature_id=feature_id,
        feature_name=feature_name,
        sub_feature=sub_feature,
        description=description,
        fields=json.dumps(VP_FIELDS),
        text=merged_text
    )
    
    result = run_llm_json(prompt, max_tokens=8000, temperature=0.0)
    
    if not result:
        return None
    
    # Add citation validation
    for field_name in VP_FIELDS:
        if field_name in result and isinstance(result[field_name], str):
            # Check if _citation exists
            citation_key = f"_citation_{field_name}" if f"_citation_{field_name}" in result else "_citation"
            # The prompt asks for nested _citation, but let's handle flat structure
            # Look for _citation in the response
            pass
    
    # Merge with base feature info
    vp_with_feature = {**feature, **result}
    vp_with_feature['_source_node_id'] = None  # Will be filled by caller
    vp_with_feature['_resolved_cross_refs'] = resolved_contexts
    
    return vp_with_feature


# ── Test Case Generation ─────────────────────────────────────────────────────

TEST_CASE_PROMPT = """You are an expert UVM Verification Engineer.
Based on the following Verification Plan details for a feature, formulate 1 to 3 concrete UVM test cases.

Feature ID: {feature_id}
Feature Name: {feature_name}
Sub Feature: {sub_feature}
Verification Goal: {verification_goal}
Constraints: {constraints}

Output a JSON array of testcase objects. Each object must have EXACTLY these keys:
- "TestCaseName": Descriptive name of the test case
- "Scenario": Brief scenario description
- "TestType": Type of test (constrained_random, directed, regression, etc.)
- "Testcase Steps": Ordered list of steps (JSON array of strings)
- "Testcase Inputs": Key inputs/configurations
- "Testcase Outputs": Expected outputs
- "ScoreboardChecks": What the scoreboard should check
- "CoveragePoints": Which coverage points this test exercises
- "Constraints": Specific constraints for this test
- "_citation": Quote from spec text supporting this test case design
- "VerificationStatus": "VERIFIED" if citation supports it, "REVIEW_REQUIRED" if not

Return ONLY the JSON array."""


def phase_2_generate_test_cases(vp_details):
    """Generate 1-3 UVM test cases for a feature."""
    feature_name = vp_details.get('Feature Name', 'Unknown')
    feature_id = vp_details.get('Feature ID', 'Unknown')
    sub_feature = vp_details.get('Sub Feature', 'Unknown')
    verification_goal = vp_details.get('Verification Goal', 'N/A')
    constraints = vp_details.get('Constraints', 'N/A')
    
    print(f"    [Phase 2] Generating test cases for feature: {feature_name}")
    
    prompt = TEST_CASE_PROMPT.format(
        feature_id=feature_id,
        feature_name=feature_name,
        sub_feature=sub_feature,
        verification_goal=verification_goal,
        constraints=constraints
    )
    
    result = run_llm_json(prompt, max_tokens=4000, temperature=0.0)
    
    if not isinstance(result, list):
        return []
    
    # Ensure each test case has required keys
    required_keys = [
        "TestCaseName", "Scenario", "TestType", "Testcase Steps",
        "Testcase Inputs", "Testcase Outputs", "ScoreboardChecks",
        "CoveragePoints", "Constraints", "_citation", "VerificationStatus"
    ]
    
    for tc in result:
        for key in required_keys:
            if key not in tc:
                tc[key] = "N/A"
        if tc.get('VerificationStatus') not in ('VERIFIED', 'REVIEW_REQUIRED'):
            tc['VerificationStatus'] = 'REVIEW_REQUIRED'
    
    return result


# ── Phase 3: Two-Pass Validation ─────────────────────────────────────────────

VALIDATION_PROMPT = """You are an expert UVM Verification Engineer reviewing a verification plan.
Your task is to verify that each field in the verification plan is supported by the specification text.

Feature: {feature_name} ({feature_id})
Sub Feature: {sub_feature}

Verification Plan:
{vp_json}

Specification Text:
{text}

For each field in the verification plan:
1. Check if the field's value is supported by the specification text
2. Check if the _citation is accurate and from the correct location
3. Mark the field as "VERIFIED" if the citation supports the claim, or "UNVERIFIED" if not

Output a JSON array of objects. Each object must have:
- "field_name": The field being reviewed
- "current_value": The current value of the field
- "verification_status": "VERIFIED" or "UNVERIFIED"
- "review_comment": Brief explanation of why it was marked verified or unverified
- "suggested_correction": If UNVERIFIED, suggest a corrected value based on the spec text (or "N/A" if cannot correct)

Return ONLY the JSON array."""


def phase_3_validate_plan(vp_details, source_text, source_lines):
    """Validate a single verification plan entry."""
    feature_id = vp_details.get('Feature ID', 'Unknown')
    feature_name = vp_details.get('Feature Name', 'Unknown')
    sub_feature = vp_details.get('Sub Feature', 'Unknown')
    
    print(f"    [Phase 3] Validating VP for feature: {feature_name}")
    
    # Build a clean VP summary for review (exclude test cases to keep it manageable)
    vp_summary = {}
    for key in VP_FIELDS:
        if key in vp_details:
            vp_summary[key] = vp_details[key]
    
    prompt = VALIDATION_PROMPT.format(
        feature_name=feature_name,
        feature_id=feature_id,
        sub_feature=sub_feature,
        vp_json=json.dumps(vp_summary, indent=2),
        text=_truncate_text(source_text, max_chars=10000)
    )
    
    result = run_llm_json(prompt, max_tokens=6000, temperature=0.0)
    
    if not isinstance(result, list):
        print(f"    [WARN] Validation failed to produce JSON, skipping validation.")
        return vp_details
    
    # Apply validation results
    review_status = {}
    for review in result:
        field_name = review.get('field_name', '')
        status = review.get('verification_status', 'UNVERIFIED')
        comment = review.get('review_comment', '')
        correction = review.get('suggested_correction', '')
        
        review_status[field_name] = {
            'status': status,
            'comment': comment,
            'suggested_correction': correction
        }
        
        # If UNVERIFIED with a suggested correction, apply it
        if status == 'UNVERIFIED' and correction and correction != 'N/A':
            if field_name in vp_details:
                vp_details[field_name] = correction
                vp_details[f"review_status_{field_name}"] = f"UNVERIFIED - corrected: {comment}"
            else:
                vp_details[field_name] = correction
                vp_details[f"review_status_{field_name}"] = f"ADDED - was missing: {comment}"
        else:
            vp_details[f"review_status_{field_name}"] = f"{status} - {comment}"
    
    # Add a top-level validation summary
    verified_count = sum(1 for v in review_status.values() if v['status'] == 'VERIFIED')
    total_count = len(review_status)
    vp_details['_validation_summary'] = {
        'verified': verified_count,
        'total': total_count,
        'verified_ratio': round(verified_count / max(total_count, 1), 2)
    }
    
    return vp_details


# ── Phase 4: Hierarchical JSON Output ────────────────────────────────────────

def build_hierarchical_output(all_features, all_vp_results, all_test_cases, all_registers, tree_structure):
    """Build the final hierarchical JSON output."""
    # Group by node
    nodes_map = {}
    
    for vp in all_vp_results:
        node_id = vp.get('_source_node_id', 'unknown')
        if node_id not in nodes_map:
            nodes_map[node_id] = {
                'node_id': node_id,
                'title': '',
                'features': []
            }
        
        # Find tree node for title
        tree_node = find_node_in_tree(tree_structure, node_id)
        title = tree_node.get('title', 'Unknown') if tree_node else 'Unknown'
        nodes_map[node_id]['title'] = title
        
        feature_data = {
            'feature': {
                'Feature ID': vp.get('Feature ID'),
                'Feature Name': vp.get('Feature Name'),
                'Sub Feature': vp.get('Sub Feature'),
                'Description': vp.get('Description')
            },
            'vp_fields': {},
            'test_cases': [],
            'cross_references': []
        }
        
        # Extract VP fields (exclude feature fields and metadata)
        for key in VP_FIELDS:
            if key in vp:
                feature_data['vp_fields'][key] = vp[key]
        
        # Add review status for each field
        for key in VP_FIELDS:
            status_key = f"review_status_{key}"
            if status_key in vp:
                feature_data['vp_fields'][f"review_{key}"] = vp[status_key]
        
        # Validation summary
        if '_validation_summary' in vp:
            feature_data['vp_fields']['_validation_summary'] = vp['_validation_summary']
        
        # Test cases
        node_test_cases = [tc for tc in all_test_cases if tc.get('_source_node_id') == node_id and tc.get('_feature_id') == vp.get('Feature ID')]
        feature_data['test_cases'] = node_test_cases
        
        # Cross references
        if '_resolved_cross_refs' in vp and vp['_resolved_cross_refs']:
            feature_data['cross_references'] = vp['_resolved_cross_refs']
        
        nodes_map[node_id]['features'].append(feature_data)
    
    # Build final structure
    output = {
        'doc_name': 'DUT_Specification',
        'total_features': len(all_vp_results),
        'total_test_cases': len(all_test_cases),
        'validation_config': {
            'if_validate_plans': if_validate_plans,
            'citation_format': citation_format
        },
        'nodes': list(nodes_map.values()),
        'global': {
            'registers_and_ports': all_registers
        }
    }
    
    return output


# ── Main Pipeline ────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("UVM Verification Plan Generation Pipeline")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Validation: {'ON' if if_validate_plans else 'OFF'}")
    print(f"Citation format: {citation_format}")
    print(f"Max cross-ref paragraphs: {max_cross_ref_paragraphs}")
    print(f"Max workers: {max_workers}")
    print("=" * 60)
    
    # Load tree
    print("\nLoading document tree...")
    tree = load_tree("./results/UCIE_1.1_structure.json")
    nodes = tree.get('structure', tree) if isinstance(tree, dict) else tree
    print(f"Loaded {len(nodes)} top-level nodes.")
    
    # Get all nodes (flatten tree)
    all_nodes = []
    def flatten(nodes_list):
        for node in nodes_list:
            all_nodes.append(node)
            if 'nodes' in node:
                flatten(node['nodes'])
    flatten(nodes)
    print(f"Total nodes in tree: {len(all_nodes)}")
    
    # TEST_MODE: filter to single node + limit features
    if test_mode:
        print(f"\n[TEST_MODE] Filtering to node: {test_node_id}, max features: {test_max_features}")
        target_node = find_node_in_tree(all_nodes, test_node_id)
        if not target_node:
            print(f"[ERROR] Node {test_node_id} not found in tree.")
            sys.exit(1)
        all_nodes = [target_node]
        print(f"[TEST_MODE] Processing 1 node (target node {test_node_id}).")
    
    # ── Phase 0: Register/Port Extraction ──────────────────────────────────
    print("\n" + "=" * 60)
    print("[PHASE 0] Global Register/Port Extraction")
    print("=" * 60)
    
    all_registers = []
    
    def run_phase_0(node):
        chapter_title = node.get('title', 'Unknown')
        registers = phase_0_extract_registers_ports(node, chapter_title)
        return registers
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_phase_0, node): node for node in all_nodes}
        for future in as_completed(futures):
            registers = future.result()
            all_registers.extend(registers)
    
    print(f"\n[Phase 0 Complete] Total registers/ports extracted: {len(all_registers)}")
    
    # ── Phase 1: Feature Extraction ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[PHASE 1] Feature Extraction with Cross-Reference Flagging")
    print("=" * 60)
    
    all_phase1_results = []
    
    def run_phase_1(node):
        chapter_title = node.get('title', 'Unknown')
        return phase_1_extract_features(node, chapter_title)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_phase_1, node): node for node in all_nodes}
        for future in as_completed(futures):
            result = future.result()
            if result['features']:
                all_phase1_results.append(result)
    
    total_features = sum(len(r['features']) for r in all_phase1_results)
    print(f"\n[Phase 1 Complete] Total features extracted: {total_features}")
    
    # ── Phase 1.5: Cross-Reference Resolution ──────────────────────────────
    print("\n" + "=" * 60)
    print("[PHASE 1.5] Cross-Reference Resolution")
    print("=" * 60)
    
    resolved_map = resolve_cross_references(all_phase1_results)
    
    # ── Phase 2: VP Generation + Test Cases ────────────────────────────────
    print("\n" + "=" * 60)
    print("[PHASE 2] Verification Plan + Test Case Generation")
    print("=" * 60)
    
    all_vp_results = []
    all_test_cases = []
    
    def run_phase_2(phase1_result):
        node_id = phase1_result['node_id']
        source_text = phase1_result['text']
        source_lines = phase1_result['text_with_lines']
        features = phase1_result['features']
        
        node_results = []
        node_test_cases = []
        
        # TEST_MODE: limit features per node
        if test_mode:
            features = features[:test_max_features]
            print(f"  [TEST_MODE] Limited to {test_max_features} features for node {node_id}.")
        
        for idx, feature in enumerate(features):
            resolved_contexts = resolved_map.get((node_id, idx), [])
            
            # Generate VP
            vp = phase_2_generate_vp(feature, source_text, source_lines, resolved_contexts)
            if not vp:
                continue
            
            vp['_source_node_id'] = node_id
            
            # Generate test cases
            test_cases = phase_2_generate_test_cases(vp)
            for tc in test_cases:
                tc['_source_node_id'] = node_id
                tc['_feature_id'] = feature.get('Feature ID', 'Unknown')
                tc['_feature_name'] = feature.get('Feature Name', 'Unknown')
            
            node_results.append(vp)
            node_test_cases.extend(test_cases)
        
        return node_results, node_test_cases
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_phase_2, r): r for r in all_phase1_results}
        for future in as_completed(futures):
            vp_results, tc_results = future.result()
            all_vp_results.extend(vp_results)
            all_test_cases.extend(tc_results)
    
    print(f"\n[Phase 2 Complete] VP entries: {len(all_vp_results)}, Test cases: {len(all_test_cases)}")
    
    # ── Phase 3: Validation (Optional) ─────────────────────────────────────
    if if_validate_plans:
        print("\n" + "=" * 60)
        print("[PHASE 3] Two-Pass Validation")
        print("=" * 60)
        
        validated_results = []
        
        # Build a lookup from phase1 results
        phase1_lookup = {r['node_id']: r for r in all_phase1_results}
        
        for vp in all_vp_results:
            node_id = vp.get('_source_node_id', '')
            phase1_result = phase1_lookup.get(node_id)
            if phase1_result:
                validated = phase_3_validate_plan(vp, phase1_result['text'], phase1_result['text_with_lines'])
                validated_results.append(validated)
            else:
                validated_results.append(vp)
        
        all_vp_results = validated_results
        
        # Print validation stats
        verified_total = 0
        total_fields = 0
        for vp in all_vp_results:
            summary = vp.get('_validation_summary', {})
            verified_total += summary.get('verified', 0)
            total_fields += summary.get('total', 0)
        
        print(f"\n[Phase 3 Complete] Validation summary:")
        print(f"  Verified: {verified_total}/{total_fields} fields ({round(verified_total/max(total_fields,1)*100, 1)}%)")
    else:
        print("\n[Phase 3 SKIPPED] Validation is disabled in config.")
    
    # ── Phase 4: Output ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[PHASE 4] Building Hierarchical Output")
    print("=" * 60)
    
    output = build_hierarchical_output(
        all_vp_results, all_vp_results, all_test_cases, all_registers, nodes
    )
    
    # Write output
    output_path = "./results/verification_plan_output.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nOutput written to: {output_path}")
    print(f"  Features: {output['total_features']}")
    print(f"  Test cases: {output['total_test_cases']}")
    print(f"  Registers/ports: {len(output['global']['registers_and_ports'])}")
    print(f"  Nodes with features: {len(output['nodes'])}")
    
    # Also save raw CSVs for compatibility with existing workflows
    print("\n[Saving CSV files for backward compatibility...]")
    save_csv("verification_plan.csv", all_vp_results)
    save_csv("ports_and_registers.csv", all_registers)
    save_csv("test_cases.csv", all_test_cases)
    
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)


def save_csv(filename, data_list):
    """Save data list to CSV with all unique keys as columns."""
    if not data_list:
        print(f"No data to save for {filename}")
        return
        
    keys = set()
    for d in data_list:
        keys.update(d.keys())
    keys = sorted(list(keys))
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys, restval='N/A', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(data_list)
    print(f"  Saved {len(data_list)} rows to {filename}")


if __name__ == "__main__":
    main()
