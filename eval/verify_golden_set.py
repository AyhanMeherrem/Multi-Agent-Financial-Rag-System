# Checks every evidence snippet and reference number in golden_set.jsonl against the raw
# filing HTML in data/raw_filings, so no reference value is taken on trust. With --index it
# also reports which snippets are missing from the Qdrant index (a parsing gap, not a
# golden set error).
#
# Usage (from the repo root): python -m eval.verify_golden_set [--index]
import argparse
import json
import os
import re
import sys

import lxml.html

from eval.metrics import normalize_text, parse_numbers

GOLDEN_SET = os.path.join("eval", "golden_set.jsonl")
FILINGS_DIR = os.path.join("data", "raw_filings", "sec-edgar-filings")
PERIOD_OF_REPORT = re.compile(r"CONFORMED PERIOD OF REPORT:\s*(\d{4})")


def load_filing_texts() -> dict:
    # {(company, year): [text variants]}. iXBRL splits text across many inline tags, so the
    # text is joined two ways: with spaces (keeps table cells apart) and without (keeps
    # words that were split across tags intact). A snippet counts if it is in either.
    texts = {}
    for company in sorted(os.listdir(FILINGS_DIR)):
        form_dir = os.path.join(FILINGS_DIR, company, "10-K")
        if not os.path.isdir(form_dir):
            continue
        for accession in sorted(os.listdir(form_dir)):
            folder = os.path.join(form_dir, accession)
            with open(os.path.join(folder, "full-submission.txt"), encoding="utf-8", errors="ignore") as f:
                match = PERIOD_OF_REPORT.search(f.read(20000))
            year = match.group(1) if match else "UNKNOWN_YEAR"
            root = lxml.html.parse(os.path.join(folder, "primary-document.html")).getroot()
            for hidden in root.xpath("//script|//style|//*[contains(translate(@style, ' ', ''), 'display:none')]"):
                hidden.drop_tree()
            parts = list(root.itertext())
            texts[(company, year)] = [normalize_text(" ".join(parts)), normalize_text("".join(parts))]
    return texts


def load_index_texts() -> dict:
    from qdrant_client import QdrantClient

    client = QdrantClient(path=os.path.join("data", "qdrant_db"))
    try:
        points, _ = client.scroll("financial_filings", limit=100000, with_payload=True)
    finally:
        client.close()
    texts = {}
    for point in points:
        key = (point.payload["company"], point.payload["year"])
        texts.setdefault(key, []).append(normalize_text(json.loads(point.payload["_node_content"])["text"]))
    return texts


def found_in(snippet: str, texts: dict, companies: list, years: list) -> bool:
    target = normalize_text(snippet)
    for (company, year), variants in texts.items():
        if companies and company not in companies:
            continue
        if years and year not in years:
            continue
        if any(target in v for v in variants):
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", action="store_true", help="also check which snippets are in the Qdrant index")
    args = parser.parse_args()

    with open(GOLDEN_SET, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    raw = load_filing_texts()
    index = load_index_texts() if args.index else None

    errors, missing_from_index = 0, []
    for item in items:
        companies, years = item["expected_companies"], item["expected_years"]
        for snippet in item["evidence_snippet"]:
            if not found_in(snippet, raw, companies, years):
                errors += 1
                print(f"ERROR {item['id']}: evidence not in raw filing text: {snippet!r}")
            elif index is not None and not found_in(snippet, index, companies, years):
                missing_from_index.append((item["id"], snippet))
        for number in item["reference_numbers"]:
            # The digits as written in the filing, e.g. "391,035 million" -> "391,035", "46.9%" -> "46.9"
            digits = re.match(r"[\d,.]+", number).group(0)
            if not parse_numbers(number) or not found_in(digits, raw, companies, years):
                errors += 1
                print(f"ERROR {item['id']}: reference number not in raw filing text: {number!r}")

    print(f"\n{len(items)} items checked, {errors} errors against the raw filings.")
    if index is not None:
        print(f"{len(missing_from_index)} evidence snippets are in the raw filings but not in the index:")
        for item_id, snippet in missing_from_index:
            print(f"  {item_id}: {snippet!r}")
    review = [i["id"] for i in items if i["needs_human_review"]]
    print(f"needs_human_review: {', '.join(review) or 'none'}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
