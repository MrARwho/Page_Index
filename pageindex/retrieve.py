import json
import PyPDF2
import re

try:
    from .utils import get_number_of_pages, remove_fields
except ImportError:
    from utils import get_number_of_pages, remove_fields


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_pages(pages: str) -> list[int]:
    """Parse a pages string like '5-7', '3,8', or '12' into a sorted list of ints."""
    result = []
    for part in pages.split(','):
        part = part.strip()
        if '-' in part:
            start, end = int(part.split('-', 1)[0].strip()), int(part.split('-', 1)[1].strip())
            if start > end:
                raise ValueError(f"Invalid range '{part}': start must be <= end")
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def _count_pages(doc_info: dict) -> int:
    """Return total page count for a PDF document."""
    if doc_info.get('page_count'):
        return doc_info['page_count']
    if doc_info.get('pages'):
        return len(doc_info['pages'])
    return get_number_of_pages(doc_info['path'])


def _get_pdf_page_content(doc_info: dict, page_nums: list[int]) -> list[dict]:
    """Extract text for specific PDF pages (1-indexed). Prefer cached pages, fallback to PDF."""
    cached_pages = doc_info.get('pages')
    if cached_pages:
        page_map = {p['page']: p['content'] for p in cached_pages}
        return [
            {'page': p, 'content': page_map[p]}
            for p in page_nums if p in page_map
        ]
    path = doc_info['path']
    with open(path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        total = len(pdf_reader.pages)
        valid_pages = [p for p in page_nums if 1 <= p <= total]
        return [
            {'page': p, 'content': pdf_reader.pages[p - 1].extract_text() or ''}
            for p in valid_pages
        ]


def _get_md_page_content(doc_info: dict, page_nums: list[int]) -> list[dict]:
    """
    For Markdown documents, 'pages' are line numbers.
    Find nodes whose line_num falls within [min(page_nums), max(page_nums)] and return their text.
    """
    min_line, max_line = min(page_nums), max(page_nums)
    results = []
    seen = set()

    def _traverse(nodes):
        for node in nodes:
            ln = node.get('line_num')
            if ln and min_line <= ln <= max_line and ln not in seen:
                seen.add(ln)
                results.append({'page': ln, 'content': node.get('text', '')})
            if node.get('nodes'):
                _traverse(node['nodes'])

    _traverse(doc_info.get('structure', []))
    results.sort(key=lambda x: x['page'])
    return results


# ── Tool functions ────────────────────────────────────────────────────────────

def get_document(documents: dict, doc_id: str) -> str:
    """Return JSON with document metadata: doc_id, doc_name, doc_description, type, status, page_count (PDF) or line_count (Markdown)."""
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({'error': f'Document {doc_id} not found'})
    result = {
        'doc_id': doc_id,
        'doc_name': doc_info.get('doc_name', ''),
        'doc_description': doc_info.get('doc_description', ''),
        'type': doc_info.get('type', ''),
        'status': 'completed',
    }
    if doc_info.get('type') == 'pdf':
        result['page_count'] = _count_pages(doc_info)
    else:
        result['line_count'] = doc_info.get('line_count', 0)
    return json.dumps(result)


def get_document_structure(documents: dict, doc_id: str) -> str:
    """Return tree structure JSON with text fields removed (saves tokens)."""
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({'error': f'Document {doc_id} not found'})
    structure = doc_info.get('structure', [])
    structure_no_text = remove_fields(structure, fields=['text'])
    return json.dumps(structure_no_text, ensure_ascii=False)


def get_page_content(documents: dict, doc_id: str, pages: str) -> str:
    """
    Retrieve page content for a document.

    pages format: '5-7', '3,8', or '12'
    For PDF: pages are physical page numbers (1-indexed).
    For Markdown: pages are line numbers corresponding to node headers.

    Returns JSON list of {'page': int, 'content': str}.
    """
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({'error': f'Document {doc_id} not found'})

    try:
        page_nums = _parse_pages(pages)
    except (ValueError, AttributeError) as e:
        return json.dumps({'error': f'Invalid pages format: {pages!r}. Use "5-7", "3,8", or "12". Error: {e}'})

    try:
        if doc_info.get('type') == 'pdf':
            content = _get_pdf_page_content(doc_info, page_nums)
        else:
            content = _get_md_page_content(doc_info, page_nums)
    except Exception as e:
        return json.dumps({'error': f'Failed to read page content: {e}'})

    return json.dumps(content, ensure_ascii=False)


def _get_page_text_with_lines(pdf_reader, page_num_0: int) -> list[dict]:
    """Extract text from a single PDF page with line numbers (0-indexed within page)."""
    page = pdf_reader.pages[page_num_0]
    text = page.extract_text() or ''
    lines = text.split('\n')
    return [
        {'line': i, 'content': line.strip()}
        for i, line in enumerate(lines)
        if line.strip()
    ]


def get_page_content_with_citations(doc_info: dict, node: dict, citation_format: str = 'quote+page') -> dict:
    """
    Extract text from a node's pages with citations (line numbers + page numbers).
    
    Returns a dict with:
    - 'text': raw concatenated text
    - 'text_with_lines': list of dicts with page, line, content for citation lookup
    - 'page_range': (start_page, end_page)
    """
    start = node.get('start_index')
    end = node.get('end_index')
    
    if start is None or end is None:
        return {'text': '', 'text_with_lines': [], 'page_range': (0, 0)}
    
    # 1-indexed page range
    start_page = start
    end_page = end
    
    text_with_lines = []
    all_text_parts = []
    
    # Try cached pages first
    cached_pages = doc_info.get('pages')
    if cached_pages:
        page_map = {p['page']: p['content'] for p in cached_pages}
        for p in range(start_page, end_page + 1):
            if p in page_map:
                content = page_map[p]
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if stripped:
                        text_with_lines.append({
                            'page': p,
                            'line': i,
                            'content': stripped
                        })
                all_text_parts.append(content + '\n')
    else:
        # Fallback to PDF reader
        path = doc_info.get('path', '')
        if not path:
            return {'text': '', 'text_with_lines': [], 'page_range': (start_page, end_page)}
        try:
            with open(path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for p in range(start_page, end_page + 1):
                    if 1 <= p <= len(pdf_reader.pages):
                        page_text_lines = _get_page_text_with_lines(pdf_reader, p - 1)
                        text_with_lines.extend(page_text_lines)
                        all_text_parts.append(page_text_lines[0]['content'] if page_text_lines else '')
        except Exception:
            pass
    
    raw_text = '\n'.join(all_text_parts)
    
    return {
        'text': raw_text,
        'text_with_lines': text_with_lines,
        'page_range': (start_page, end_page)
    }


def find_citation_by_quote(text_with_lines: list[dict], quote: str) -> dict:
    """
    Find the page and line number for a given quote in the extracted text lines.
    Returns {'page': int, 'line': int, 'matched_content': str} or None.
    """
    if not quote or not text_with_lines:
        return None
    
    quote_normalized = quote.strip().lower()
    
    # Try exact match first
    for entry in text_with_lines:
        if entry['content'].strip().lower() == quote_normalized:
            return {
                'page': entry['page'],
                'line': entry['line'],
                'matched_content': entry['content']
            }
    
    # Try substring match (quote appears within a longer line)
    for entry in text_with_lines:
        if quote_normalized in entry['content'].strip().lower():
            return {
                'page': entry['page'],
                'line': entry['line'],
                'matched_content': entry['content']
            }
    
    return None
