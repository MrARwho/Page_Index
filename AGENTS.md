# Page Index — Agent Guide

## What This Repo Does
Vectorless RAG framework: generates hierarchical JSON trees from PDF/MD using LLMs to infer section boundaries and summaries — no embeddings. Output: `./results/<name>_structure.json`.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Env / API Keys
| Variable | Purpose | Default |
|---|---|---|
| `OPENAI_API_KEY` (also `CHATGPT_API_KEY`) | OpenAI / custom base models | — |
| `ANTHROPIC_API_KEY` | `anthropic/` models | — |
| `NVIDIA_API_KEY` | `nvidia/` models | — |
| `OPENAI_API_BASE` | Base URL for `openai/` or `llamacpp/` models | `http://127.0.0.1:8080/v1` |

Loaded automatically via `python-dotenv`.

## CLI Usage

### PDF
```bash
python run_pageindex.py --pdf_path <file.pdf>
```

### Markdown
```bash
python run_pageindex.py --md_path <file.md>
```

### Flags
| Flag | PDF | MD | Default |
|---|---|---|---|
| `--model` | Y | Y | `gpt-4o-2024-11-20` |
| `--toc-check-pages` | Y | — | `20` |
| `--max-pages-per-node` | Y | — | `10` |
| `--max-tokens-per-node` | Y | — | `20000` |
| `--if-add-node-id` | Y | Y | `yes` |
| `--if-add-node-summary` | Y | Y | `yes` |
| `--if-add-doc-description` | Y | Y | `no` |
| `--if-add-node-text` | Y | Y | `no` |
| `--if-thinning` | — | Y | `no` |
| `--thinning-threshold` | — | Y | `5000` |
| `--summary-token-threshold` | — | Y | `200` |

## Config
Defaults in `pageindex/config.yaml`:
```yaml
model: "gpt-4o-2024-11-20"
retrieve_model: "gpt-5.4"         # defaults to model if unset
toc_check_page_num: 20
max_page_num_each_node: 10
max_token_num_each_node: 20000
if_add_node_id: "yes"
if_add_node_summary: "yes"
if_add_doc_description: "no"
if_add_node_text: "no"
```

## LLM Routing (litellm)
| Prefix | Provider | Details |
|---|---|---|
| (none / `gpt-4o*`) | OpenAI | Standard litellm chat |
| `anthropic/` | Anthropic | litellm chat |
| `ollama/` | Local Ollama | Custom sync/async httpx client |
| `nvidia/` | NVIDIA | `https://integrate.api.nvidia.com/v1`, model remapped to `openai/` |
| `openai/` or `llamacpp/` | Custom base | Uses `OPENAI_API_BASE`; key: `OPENAI_API_KEY` (fallback: `sk-fake`) |

**Quirks:**
- `litellm.drop_params = False` — unknown params cause errors.
- Custom httpx: **3600s read timeout**, 60s connect timeout.
- LLM calls retry **10x** with 1s delay.
- `enable_thinking` kwarg passes `chat_template_kwargs` for llama.cpp.
- `llamacpp/` prefix → remapped to `openai/` before litellm.
- Debug output: `[DEBUG-llm_completion]`, `[DEBUG-llm_acompletion]`, `[DEBUG-llm_output]`.

## JSON Extraction
`utils.extract_json()` fallback chain:
1. Strip ` ```json ` ... ` ``` ` delimiters
2. Replace Python `None` → `null`
3. Normalize whitespace
4. Remove trailing commas before `]`/`}`
5. Return `{}` on failure

## Architecture
| File | Role |
|---|---|
| `run_pageindex.py` | CLI entrypoint, argparse, orchestration |
| `pageindex/page_index.py` | Core PDF tree generation (`page_index_main`) |
| `pageindex/page_index_md.py` | Markdown tree generation (`md_to_tree`, async) |
| `pageindex/retrieve.py` | Tool functions for doc/structure/page retrieval |
| `pageindex/client.py` | `PageIndexClient` — workspace init, async indexing flow |
| `pageindex/utils.py` | LLM wrappers (`llm_completion`, `llm_acompletion`), JSON parsing, config loading, tree helpers (`write_node_id`, `get_leaf_nodes`, `structure_to_list`, `list_to_tree`), token counting |
| `pageindex/config.yaml` | Default configuration |
| `pageindex/__init__.py` | Public API: `md_to_tree`, `get_document`, `get_document_structure`, `get_page_content`, `PageIndexClient` |
| `retrival.py` | Vectorless RAG query — 3-step LLM reasoning over tree structure |
| `generate_verification_plan.py` | UVM verification pipeline — 4-stage feature extraction → CSV output |

## PDF Indexing Flow
1. Parse PDF, scan first `toc_check_pages` for TOC → `doc_description` (LLM)
2. Verify title with LLM → `verified_title` + `verified_description`
3. Split pages into chunks ≤ `max_pages_per_node` pages AND ≤ `max_token_num_each_node` tokens
4. For each chunk: LLM generates section titles + summaries
5. Concurrent execution via `ThreadPoolExecutor` (max_workers from config)
6. Merge chunks → recursive tree structure
7. Write `node_id` to each node if `if_add_node_id=yes`
8. Generate summaries for leaf/prefix nodes if `if_add_node_summary=yes`
9. Output JSON

## MD Indexing Flow
1. Parse markdown: extract headers (`#{1,6}` pattern), skip code blocks
2. Assign text content to each node (from header line to next header)
3. If `if_thinning=yes`: count tokens per node, merge children into parent when parent tokens < `thinning_threshold`
4. Build tree from flat node list using stack-based level comparison
5. If `if_add_node_summary=yes`: async LLM summaries (leaf→`summary`, parent→`prefix_summary`)
6. If `if_add_doc_description=yes`: generate doc description from clean structure
7. Format output per flags, strip fields not requested

## Retrieval
`pageindex/retrieve.py` tool functions:
| Function | Purpose |
|---|---|
| `get_document(documents, doc_id)` | Metadata: doc_id, doc_name, doc_description, type, status, page_count/line_count |
| `get_document_structure(documents, doc_id)` | Tree structure JSON with `text` fields removed (token savings) |
| `get_page_content(documents, doc_id, pages)` | Page content: `pages` = `'5-7'`, `'3,8'`, `'12'`. PDF: physical pages (1-indexed). MD: line numbers matching node headers. |

## Retrieval & Verification
### `retrival.py` — Vectorless RAG Query
- Loads tree JSON, builds lightweight TOC from node summaries
- **3-step pipeline**: (1) LLM reasons over structure to select relevant node IDs, (2) extracts text from selected nodes, (3) generates final answer from extracted text
- Uses `litellm` with `reasoning_budget` and `reasoning_format` kwargs
- Model: `openai/gemma-4-31b` via `http://localhost:8080/v1`
- **Quirks**: hardcodes base URL to `http://localhost:8080/v1` and API key to `sk-no-key-required`; ignores env vars
- Note: filename has typo `retrival.py` (not `retrieval.py`)

### `generate_verification_plan.py` — UVM Verification Pipeline
- 4-stage pipeline per document node/chapter:
  1. **Feature Extraction**: identify testable features, sub-modules, capabilities
  2. **VP Fields**: generate detailed verification plan fields (23 fields including coverage, scoreboards, sequences)
  3. **Ports/Registers**: extract defined ports, signals, registers with metadata
  4. **Test Cases**: generate 1-3 UVM test cases per feature
- Reads PDF text directly via `fitz` (PyMuPDF) using node `start_index`/`end_index`
- Outputs: `verification_plan.csv`, `ports_and_registers.csv`, `test_cases.csv`
- `TEST_MODE` flag limits to single node + 2 features for fast iteration
- **Quirks**: globally opens `UCIE_1.1.pdf` via `fitz`; test mode targets node ID `"0035"`
- Model: `openai/gemma-4-31b` with `reasoning_budget: 2000`

## Output
- Results: `./results/<basename>_structure.json` (2-space indent, ensure_ascii=False)
- Logs: `./logs/` (JSON format via `JsonLogger`)
- Output structure: `{doc_name, line_count/page_count, doc_description?, structure: [{title, node_id, summary?, prefix_summary?, line_num?, text?, nodes?}]}`

## Dependencies
- `litellm` — LLM routing
- `pymupdf` — primary PDF parser
- `PyPDF2` — fallback PDF parser
- `python-dotenv` — env loading
- `pyyaml` — config loading
