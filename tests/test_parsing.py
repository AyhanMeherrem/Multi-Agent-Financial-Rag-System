import app.parsing.parse_filings as parsing
from app.parsing.parse_filings import (clean_row, filing_url, find_section_starts, is_noise, read_filing_header,
                                       strip_ixbrl, table_to_rows)

TABLE_HTML = """<table>
<tr><td></td><td>Years ended</td></tr>
<tr><td></td><td>September 27, 2025</td><td></td><td>September 28, 2024</td></tr>
<tr><td>Net sales:</td><td></td></tr>
<tr><td>Total net sales</td><td>$</td><td>416,161</td><td></td><td>$</td><td>391,035</td></tr>
<tr><td>Other income/(expense), net</td><td>(</td><td>321</td><td>)</td><td>269</td></tr>
<tr><td>Gross margin percentage</td><td>46.9</td><td>%</td><td>46.2</td><td>%</td></tr>
</table>"""

# Table of contents first (with no text after each entry), then the real headings with content
FILING_HTML = """<html><body>
<p>Item 1. Business</p><p>Item 1A. Risk Factors</p><p>Item 7. Management's Discussion and Analysis</p>
<p>Item 1. Business</p>
<p>The Company designs, manufactures and markets smartphones, personal computers and tablets worldwide.</p>
<p>Apple Inc. | 2025 Form 10-K | 3</p>
<p>Item 1A. Risk Factors</p>
<p>The Company's operations depend on complex global supply chains that are subject to disruption.</p>
<p>Item 7. Management's Discussion and Analysis</p>
<p>Total net sales increased during the year due to higher net sales of iPhone and Services.</p>
<table><tr><td>Total net sales</td><td>$</td><td>416,161</td></tr></table>
</body></html>"""


def test_clean_row_merges_sec_table_cells():
    assert clean_row(["Total", "$", "416,161", "", "(", "321", ")", "46.9", "%"]) == ["Total", "416,161", "(321)", "46.9%"]


def test_table_to_rows_keeps_rows_and_separates_header():
    header, body = table_to_rows(TABLE_HTML)
    assert header == ["Years ended", "September 27, 2025 | September 28, 2024", "Net sales:"]
    assert body == ["Total net sales | 416,161 | 391,035",
                    "Other income/(expense), net | (321) | 269",
                    "Gross margin percentage | 46.9% | 46.2%"]


def test_table_of_contents_entries_do_not_start_sections():
    texts = ["Item 1. Business", "Item 1A. Risk Factors",            # table of contents
             "Item 1. Business", "long business text " * 20,          # real Item 1
             "Item 1A. Risk Factors", "long risk text " * 20]          # real Item 1A
    starts = find_section_starts(texts, [False] * len(texts))
    assert starts == {2: "Item 1", 4: "Item 1A"}


def test_running_headers_and_cross_references_are_not_headings():
    texts = ["Item 8", "Item 1B, 1C", "See Item 7 of this Form 10-K for details.", "ITEM 8. FINANCIAL STATEMENTS", "x" * 50]
    assert find_section_starts(texts, [False] * len(texts)) == {3: "Item 8"}


def test_noise_patterns_cover_both_companies():
    for text in ["Apple Inc. | 2024 Form 10-K | 17", "85", "PART II", "PART II, III", "Item 8", "Item 1B, 1C",
                 "See accompanying Notes to Consolidated Financial Statements.", "Refer to accompanying notes."]:
        assert is_noise(text), text
    for text in ["Item 7. Management's Discussion", "Net sales increased 6%.", "See Note 5 for details."]:
        assert not is_noise(text), text


def test_strip_ixbrl_keeps_text_blocks_and_drops_hidden_header():
    html = ('<div><ix:header><ix:hidden>secret</ix:hidden></ix:header>'
            '<ix:nonNumeric name="cyd:Block" escape="true"><p>Board oversees cybersecurity.</p></ix:nonNumeric></div>')
    cleaned = strip_ixbrl(html)
    assert "Board oversees cybersecurity." in cleaned
    assert "secret" not in cleaned and "ix:" not in cleaned


def test_read_filing_header_and_url(tmp_path):
    submission = tmp_path / "full-submission.txt"
    submission.write_text("ACCESSION NUMBER:\t\t0000320193-25-000079\nCONFORMED PERIOD OF REPORT:\t20250927\n"
                          "\t\tCENTRAL INDEX KEY:\t\t\t0000320193\n<DOCUMENT>\n<TYPE>10-K\n<SEQUENCE>1\n"
                          "<FILENAME>aapl-20250927.htm\n<TYPE>EX-4.1\n<FILENAME>exhibit.htm\n")
    header = read_filing_header(str(submission))
    assert header == {"year": "2025", "period_end_date": "2025-09-27", "accession_number": "0000320193-25-000079",
                      "cik": "0000320193", "primary_document": "aapl-20250927.htm"}
    assert filing_url(header["cik"], header["accession_number"], header["primary_document"]) == \
        "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm"


def test_parse_single_filing_end_to_end(tmp_path, monkeypatch):
    # Word count instead of the bge tokenizer, so the test needs no model download
    monkeypatch.setattr(parsing, "count_tokens", lambda text: len(text.split()))
    path = tmp_path / "primary-document.html"
    path.write_text(FILING_HTML, encoding="utf-8")
    nodes = parsing.parse_single_filing(str(path), {"company": "AAPL", "year": "2025"})

    by_section = {}
    for n in nodes:
        by_section.setdefault(n.metadata["section"], []).append(n)
    assert "smartphones" in by_section["Item 1"][0].text
    assert "supply chains" in by_section["Item 1A"][0].text
    assert any(n.metadata["element_type"] == "table" and "Total net sales | 416,161" in n.text for n in by_section["Item 7"])
    assert all("Form 10-K |" not in n.text for n in nodes)  # footer removed
    assert "filing_url" in nodes[0].excluded_embed_metadata_keys
