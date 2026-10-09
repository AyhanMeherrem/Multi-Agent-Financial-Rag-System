# Turns each downloaded 10-K into section-tagged, retrieval-sized TextNode chunks.
# Run from the repo root: python -m app.parsing.parse_filings
#
# Pipeline per filing: strip iXBRL wrappers -> unstructured.partition_html -> find the real
# "Item N." headings -> drop page headers/footers -> tables become row-per-line text ->
# adjacent paragraphs are merged into ~450-token chunks that never cross a section.

import os
import re
from collections import defaultdict
from functools import lru_cache
from typing import List

import lxml.html
from llama_index.core.schema import TextNode    # For capsulating chunks (which will conver to embedding)

BASE_DIR = os.path.join("data", "raw_filings", "sec-edgar-filings")
EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"

# Chunk size in bge tokens. bge-large truncates at 512 tokens, and the company/year/section
# metadata is prepended to the embedded text, so chunks stay well below that.
MAX_CHUNK_TOKENS = 450
# A new sub-heading starts a new chunk only if the current one already has this much text;
# otherwise the heading is kept inline and small subsections are merged.
MIN_CHUNK_TOKENS = 200

# Metadata that is useful for citations but should not be embedded or shown to the LLM
NON_EMBEDDED_KEYS = ["period_end_date", "accession_number", "cik", "filing_url", "element_type"]

# "Item 1A. Risk Factors": a real heading has the item number followed by a title. Running page
# headers ("Item 1A", "Item 1B, 1C") and cross-references ("see Item 7 of this Form 10-K") do not match.
HEADING_PATTERN = re.compile(r"^item\s+(\d{1,2}[a-c]?)\s*[.:\-–—]\s*\S", re.IGNORECASE)
NOISE_PATTERNS = [
    re.compile(r"^\d{1,3}$"),                                                    # page numbers
    re.compile(r"^part\s+[ivx]+(\s*,\s*[ivx]+)*$", re.IGNORECASE),               # "PART II", "PART II, III"
    re.compile(r"^item\s+\d{1,2}[a-c]?(\s*,\s*\d{1,2}[a-c]?)*$", re.IGNORECASE),  # MSFT page headers "Item 8", "Item 1B, 1C"
    re.compile(r"^.{0,60}\|\s*\d{4}\s*form\s+10-k\s*\|\s*\d+$", re.IGNORECASE),  # AAPL footer "Apple Inc. | 2024 Form 10-K | 17"
    re.compile(r"^(see|refer to) (the )?accompanying notes", re.IGNORECASE),     # statement boilerplate (both companies)
]
HEADER_FIELDS = {
    "accession_number": re.compile(r"ACCESSION NUMBER:\s*(\S+)"),
    "period": re.compile(r"CONFORMED PERIOD OF REPORT:\s*(\d{8})"),
    "cik": re.compile(r"CENTRAL INDEX KEY:\s*(\d+)"),
    "primary_document": re.compile(r"<TYPE>10-K.*?<FILENAME>(\S+)", re.S),
}


@lru_cache(maxsize=1)
def get_tokenizer():
    # The embedding model's own tokenizer, so chunk sizes are measured in the tokens bge truncates at
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)


def count_tokens(text: str) -> int:
    return len(get_tokenizer().encode(text, add_special_tokens=False))


def ensure_nltk_data():
    # unstructured needs these NLTK models to classify elements; downloaded once, not at import time
    import nltk
    nltk.download('punkt_tab', quiet=True)  # For deciding if sentence ended or not
    nltk.download('averaged_perceptron_tagger_eng', quiet=True) # For knowing context of words and its grammaticaly role(verb, noun etc.)


def read_filing_header(submission_txt_path: str) -> dict:
    # The SEC header at the top of full-submission.txt has the fiscal period, accession number,
    # CIK and the real file name of the primary document (sec-edgar-downloader renames it)
    with open(submission_txt_path, "r", encoding="utf-8", errors="ignore") as f:
        head = f.read(200_000)
    found = {}
    for key, pattern in HEADER_FIELDS.items():
        m = pattern.search(head)
        found[key] = next((g for g in m.groups() if g), None) if m else None
    period = found["period"]
    return {
        "year": period[:4] if period else "UNKNOWN_YEAR",
        "period_end_date": f"{period[:4]}-{period[4:6]}-{period[6:]}" if period else None,
        "accession_number": found["accession_number"],
        "cik": found["cik"],
        "primary_document": found["primary_document"],
    }


def filing_url(cik: str, accession_number: str, primary_document: str) -> str | None:
    # EDGAR archive layout: /Archives/edgar/data/<CIK without leading zeros>/<accession without dashes>/<file>
    if not (cik and accession_number and primary_document):
        return None
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_number.replace('-', '')}/{primary_document}"


def strip_ixbrl(html: str) -> str:
    # Inline XBRL wraps tagged text (notes to the financial statements, and the Item 1C cybersecurity
    # disclosure since 2025) in <ix:nonNumeric> elements, and partition_html drops their content. Remove
    # the hidden <ix:header> data block and unwrap every other ix: tag, keeping the text inside.
    html = re.sub(r"<ix:header>.*?</ix:header>", "", html, flags=re.S)
    return re.sub(r"</?ix:[^>]*>", "", html)


def is_noise(text: str) -> bool:
    return any(p.match(text) for p in NOISE_PATTERNS)


UNIT_LINE = re.compile(r"^\(in (millions|thousands|billions)", re.IGNORECASE)


def is_subheading(element, text: str) -> bool:
    # unstructured labels most 10-K headings as plain Text, so use shape: short and not a sentence.
    # Unit lines like "(In millions, except per share amounts)" belong to the heading of the next table.
    if UNIT_LINE.match(text) and len(text) < 160:
        return True
    return type(element).__name__ in ("Title", "Text") and len(text) < 120 and not text.endswith((".", ":", ";"))


def find_section_starts(texts: List[str], is_table: List[bool]) -> dict:
    # Returns {element index: section} for the real "Item N." headings. An item heading can appear
    # several times (table of contents, cross-reference lines); the real one is the occurrence followed
    # by the most text before the next heading, so table-of-contents entries never change the section.
    candidates = [(i, "Item " + m.group(1).upper()) for i, t in enumerate(texts)
                  if not is_table[i] and len(t) <= 200 and (m := HEADING_PATTERN.match(t))]
    best = {}
    for n, (i, section) in enumerate(candidates):
        end = candidates[n + 1][0] if n + 1 < len(candidates) else len(texts)
        following = sum(len(t) for t in texts[i + 1:end])
        if section not in best or following > best[section][1]:
            best[section] = (i, following)
    return {i: section for section, (i, _) in best.items()}


def clean_row(cells: List[str]) -> List[str]:
    # SEC tables put "$", ")" and "%" in their own cells and pad with empty spacer cells
    out = []
    for cell in cells:
        cell = re.sub(r"\s+", " ", cell).strip()
        if not cell or cell == "$":
            continue
        if out and (cell in (")", "%", ")%") or out[-1] == "("):
            out[-1] = (out[-1] + cell).replace("( ", "(")
        else:
            out.append(cell)
    return out


def is_value(cell: str) -> bool:
    # A number cell such as 391,035 / (321) / 46.9% / 6.08 / — ; years (2024) count as header text
    return bool(re.fullmatch(r"\(?-?[\d,]*\.?\d+\)?%?|[—–-]", cell)) and not re.fullmatch(r"(19|20)\d\d", cell)


def table_to_rows(table_html: str) -> tuple:
    # Returns (header_rows, body_rows) as "cell | cell | cell" lines. Header rows are the leading rows
    # without any number in them (e.g. "Years ended" / "September 27, 2025 | September 28, 2024").
    root = lxml.html.fromstring(table_html)
    rows = [clean_row([c.text_content() for c in tr.xpath("./td|./th")]) for tr in root.xpath(".//tr")]
    rows = [" | ".join(r) for r in rows if r]
    split = 0
    while split < len(rows) and not any(is_value(c) for c in rows[split].split(" | ")[1:]):
        split += 1
    if split == len(rows):  # no numbers at all, e.g. a list laid out as a table
        return [], rows
    return rows[:split], rows[split:]


def table_chunks(element, caption: str | None) -> List[str]:
    # One table becomes one or more chunks; each chunk repeats the caption and the header rows
    html = getattr(element.metadata, "text_as_html", None)
    header, body = table_to_rows(html) if html else ([], [str(element)])
    prefix = "\n".join(([f"Table: {caption}"] if caption else []) + header)
    chunks, current = [], []
    for row in body:
        candidate = "\n".join(filter(None, [prefix] + current + [row]))
        if current and count_tokens(candidate) > MAX_CHUNK_TOKENS:
            chunks.append("\n".join(filter(None, [prefix] + current)))
            current = []
        current.append(row)
    if current:
        chunks.append("\n".join(filter(None, [prefix] + current)))
    return chunks


def parse_single_filing(html_path: str, base_metadata: dict) -> List[TextNode]:
    from unstructured.partition.html import partition_html

    print(f"Partitioning HTML for {base_metadata['company']} ({base_metadata['year']}): {html_path}")
    with open(html_path, encoding="utf-8", errors="ignore") as f:
        elements = partition_html(text=strip_ixbrl(f.read()), include_page_breaks=False)
    texts = [str(e).strip() for e in elements]
    is_table = [type(e).__name__ == "Table" for e in elements]
    section_starts = find_section_starts(texts, is_table)

    nodes: List[TextNode] = []
    section = "General"  # cover page and anything before the first item heading
    headings = []        # the latest run of consecutive headings, e.g. ["CONSOLIDATED STATEMENTS OF OPERATIONS", "(In millions...)"]
    after_heading = False
    title = None         # heading prefix of the text chunk being built
    paragraphs, tokens = [], 0

    def emit(text: str, element_type: str):
        metadata = {**base_metadata, "section": section, "element_type": element_type}
        nodes.append(TextNode(text=text, metadata=metadata,
                              excluded_embed_metadata_keys=NON_EMBEDDED_KEYS,
                              excluded_llm_metadata_keys=NON_EMBEDDED_KEYS))

    def flush():
        nonlocal paragraphs, tokens
        if paragraphs:
            emit("\n".join(([title] if title else []) + paragraphs), "text")
        paragraphs, tokens = [], 0

    for i, (element, text) in enumerate(zip(elements, texts)):
        if not text or is_noise(text) or type(element).__name__ == "Image":
            continue
        if i in section_starts:
            flush()
            section, headings, after_heading = section_starts[i], [text], True
        elif is_table[i]:
            flush()
            # Each table chunk is captioned with the headings it sits under
            for chunk in table_chunks(element, " / ".join(headings[-2:]) or None):
                emit(chunk, "table")
            after_heading = False
        elif is_subheading(element, text):
            headings = headings + [text] if after_heading else [text]
            after_heading = True
            if tokens >= MIN_CHUNK_TOKENS:
                flush()
            elif paragraphs:
                paragraphs.append(text)  # small subsection: keep merging, heading stays inline
        else:
            after_heading = False
            n = count_tokens(text)
            if paragraphs and tokens + n > MAX_CHUNK_TOKENS:
                flush()
            if not paragraphs:
                title = " / ".join(headings[-2:]) or None  # a new chunk starts under the latest headings
            paragraphs.append(text)
            tokens += n
    flush()
    return nodes


def parse_all_filings(base_dir: str = BASE_DIR) -> List[TextNode]:
    ensure_nltk_data()
    all_nodes: List[TextNode] = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if not file.endswith((".html", ".htm")):
                continue
            full_path = os.path.join(root, file)
            # Layout: <base_dir>/<TICKER>/10-K/<accession>/primary-document.html
            company = os.path.relpath(full_path, base_dir).split(os.sep)[0]
            header = read_filing_header(os.path.join(root, "full-submission.txt"))
            base_metadata = {
                "company": company,
                "year": header["year"],
                "period_end_date": header["period_end_date"],
                "accession_number": header["accession_number"],
                "cik": header["cik"],
                "filing_url": filing_url(header["cik"], header["accession_number"], header["primary_document"]),
            }
            all_nodes.extend(parse_single_filing(full_path, base_metadata))
    return all_nodes


def section_report(nodes: List[TextNode]) -> dict:
    # {(company, year): {section: [chunks, chars]}}
    report = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for n in nodes:
        entry = report[(n.metadata["company"], n.metadata["year"])][n.metadata["section"]]
        entry[0] += 1
        entry[1] += len(n.text)
    return report


if __name__ == "__main__":
    parsed_nodes = parse_all_filings()
    print(f"\nExtracted {len(parsed_nodes)} chunks.")
    token_counts = sorted(count_tokens(n.text) for n in parsed_nodes)
    print(f"Tokens per chunk: min {token_counts[0]}, median {token_counts[len(token_counts) // 2]}, "
          f"max {token_counts[-1]}, over 480: {sum(t > 480 for t in token_counts)}")

    report = section_report(parsed_nodes)
    for filing, sections in sorted(report.items()):
        print(f"\n--- {filing[0]} FY{filing[1]} ---")
        for sec, (count, chars) in sorted(sections.items(), key=lambda kv: (len(kv[0]), kv[0])):
            print(f"  {sec:8s}: {count:4d} chunks {chars:8d} chars")

    # The sections the router relies on most must have real content in every filing
    for filing, sections in report.items():
        for sec in ("Item 1", "Item 1A", "Item 7", "Item 8"):
            assert sections[sec][1] >= 10_000, f"{filing}: {sec} has only {sections[sec][1]} characters"
    print("\nSection check passed: Items 1, 1A, 7 and 8 have substantial content in every filing.")
