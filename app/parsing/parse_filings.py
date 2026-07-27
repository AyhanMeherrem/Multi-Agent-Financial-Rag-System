# We needed to have filter for raw html and split it into Element trees which unstructured 

import os
import re
from typing import List, Dict, Any

#preloading sentence tokenization modles
import nltk
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
from unstructured.partition.html import partition_html

from llama_index.core.schema import TextNode    # For capsulating chunks (which will conver to embedding)

SECTION_PATTERNS = {
    "Item 1": re.compile(r"item\s+1[\.\s:\–\-]+business", re.IGNORECASE),
    "Item 1A": re.compile(r"item\s+1a[\.\s:\–\-]+risk\s+factors", re.IGNORECASE),
    "Item 1C": re.compile(r"item\s+1c[\.\s:\–\-]+cybersecurity", re.IGNORECASE),
    "Item 2": re.compile(r"item\s+2[\.\s:\–\-]+properties", re.IGNORECASE),
    "Item 3": re.compile(r"item\s+3[\.\s:\–\-]+legal\s+proceedings", re.IGNORECASE),
    "Item 5": re.compile(r"item\s+5[\.\s:\–\-]+market\s+for", re.IGNORECASE),
    "Item 7": re.compile(r"item\s+7[\.\s:\–\-]+management[\'’]?s\s+discussion", re.IGNORECASE),
    "Item 7A": re.compile(r"item\s+7a[\.\s:\–\-]+quantitative", re.IGNORECASE),
    "Item 8": re.compile(r"item\s+8[\.\s:\–\-]+financial\s+statements", re.IGNORECASE),
    "Item 9": re.compile(r"item\s+9[\.\s:\–\-]+changes\s+in", re.IGNORECASE),
    "Item 9A": re.compile(r"item\s+9a[\.\s:\–\-]+controls\s+and\s+procedures", re.IGNORECASE),
}


PERIOD_OF_REPORT_PATTERN = re.compile(r"CONFORMED PERIOD OF REPORT:\s*(\d{8})")
PAGE_FOOTER_PATTERN = re.compile(r"^.{0,60}\|\s*\d{4}\s*Form\s+10-K\s*\|\s*\d+\s*$")


def extract_filing_year(submission_txt_path: str) -> str:


    # Reading Sec header from fullsubmission to find out which fiscal year that pdf is from? If there is no such it we store it as 'unkown year'
    try:
        with open(submission_txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(60):
                line = f.readline()
                if not line:
                    break
                match = PERIOD_OF_REPORT_PATTERN.search(line)
                if match:
                    return match.group(1)[:4]
    except OSError:
        pass
    return "UNKNOWN_YEAR"



# For State machine

def parse_single_filing(html_path: str, company: str, year: str) -> List[TextNode]:
    print(f"Partitioning HTML dom tree for {company} ({year}): {html_path}")
    elements = partition_html(filename=html_path, include_page_breaks=False)    # ALl html
    nodes: List[TextNode] = []
    current_section = "General" # To track which chapter are we in while iterating over 

    for element in elements:
        text = str(element).strip()
        if not text:
            continue

        # Check section headings on every element (even short ones like "Item 2. Properties",
        # which is only 18 chars) before applying the content-length filter below. Otherwise
        # short headings get skipped and the state machine never transitions into that section.
        for sec_name, pattern in SECTION_PATTERNS.items():
            if pattern.search(text[:150]):
                current_section = sec_name
                break

        # Skip short texts ( maybe change len threshold to idk, regards for split)
        if len(text) < 20:
            continue

        if PAGE_FOOTER_PATTERN.match(text):
            continue
        node = TextNode(text=text, metadata={"company": company, "year": year, "section": current_section})
        nodes.append(node)
    return nodes


# Returing all filings inside DB
def parse_all_filings(base_dir: str = "./data/raw_filings/sec-edgar-filings") -> List[TextNode]:
    all_nodes: List[TextNode] = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".html") or file.endswith(".htm"):
                full_path = os.path.join(root, file)
                parts = full_path.replace("\\", "/").split("/")
                company = parts[-4] if len(parts) >= 4 else "UNKNOWN"
                submission_txt_path = os.path.join(root, "full-submission.txt")
                year = extract_filing_year(submission_txt_path)
                filing_nodes = parse_single_filing(full_path, company, year)
                all_nodes.extend(filing_nodes)
    return all_nodes

# Final 3390 Node created (combined from 4 fillings => AAPL(2), MSFT(2)),
#

if __name__ == "__main__":
    parsed_nodes = parse_all_filings()
    print(f"Successfully extracted {len(parsed_nodes)} total section-annotated text nodes.")
    
    from collections import Counter
    section_counts = Counter([n.metadata["section"] for n in parsed_nodes])
    print("\n--- Section Distribution Breakdown ---")
    for sec, count in sorted(section_counts.items()):
        print(f"  {sec:10s}: {count} nodes")