# We needed to have filter for raw html and split it into Element trees which unstructured 

import os
import re
from typing import List, Dict, Any

#preloading sentence tokenization modles
# pyrefly: ignore [missing-import]
import nltk
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
# pyrefly: ignore [missing-import]
from unstructured.partition.html import partition_html

# pyrefly: ignore [missing-import]
from llama_index.core.schema import TextNode    # For capsulating chunks (which will conver to embedding)

SECTION_PATTERNS = {
    "Item 1A": re.compile(r"item\s+1a[\.\s:\–\-]+risk\s+factors", re.IGNORECASE),
    "Item 7": re.compile(r"item\s+7[\.\s:\–\-]+management[\'’]?s\s+discussion", re.IGNORECASE),
    "Item 8": re.compile(r"item\s+8[\.\s:\–\-]+financial\s+statements", re.IGNORECASE),
}

# For State machine

def parse_single_filing(html_path: str, company: str, year: str) -> List[TextNode]:
    print(f"Partitioning HTML dom tree for {company} ({year}): {html_path}")
    elements = partition_html(filename=html_path, include_page_breaks=False)    # ALl html
    nodes: List[TextNode] = []
    current_section = "General" # To track which chapter are we in while iterating over 

    for element in elements:
        text = str(element).strip()
        # Skip for short texts ( maybe change len threshold to idk, regards for split) 
        
        # LOOPING over same patterns
        if not text or len(text) < 20:
            continue
        for sec_name, pattern in SECTION_PATTERNS.items():
            if pattern.search(text[:150]):
                current_section = sec_name
                break
        node = TextNode(text=text, metadata={"company": company, "year": year, "section": current_section})
        nodes.append(node)
    return nodes

def parse_all_filings(base_dir: str = "./data/raw_filings/sec-edgar-filings") -> List[TextNode]:
    all_nodes: List[TextNode] = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".html") or file.endswith(".htm"):
                full_path = os.path.join(root, file)
                parts = full_path.replace("\\", "/").split("/")
                company = parts[-4] if len(parts) >= 4 else "UNKNOWN"
                year = "2024"
                filing_nodes = parse_single_filing(full_path, company, year)
                all_nodes.extend(filing_nodes)
    return all_nodes

# Final 3390 Node created (combined from 4 fillings => AAPL(2), MSFT(2)),
#

if __name__ == "__main__":
    parsed_nodes = parse_all_filings()
    print(f"Successfully extracted {len(parsed_nodes)} total section-annotated text nodes.")