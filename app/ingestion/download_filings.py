# Run from the repo root: python -m app.ingestion.download_filings
import os
import re
import shutil
import tempfile
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from sec_edgar_downloader import Downloader

load_dotenv()

TARGET_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META"]
# Every company covers the same fiscal years so comparisons line up. "Latest two filings" would
# not: companies close their fiscal years in different months, so on a given day MSFT and NVDA
# may already have filed FY2026 while the others have not.
FISCAL_YEARS = {"2024", "2025"}
# Enough recent filings to contain both fiscal years for every company
FILINGS_PER_TICKER = 3
DOWNLOAD_DIR = os.path.join("data", "raw_filings")
PERIOD_OF_REPORT = re.compile(r"CONFORMED PERIOD OF REPORT:\s*(\d{4})")


def fiscal_year(filing_dir: str) -> str | None:
    # The fiscal year a 10-K reports on, from the EDGAR header of its full submission
    with open(os.path.join(filing_dir, "full-submission.txt"), encoding="utf-8", errors="ignore") as f:
        match = PERIOD_OF_REPORT.search(f.read(20000))
    return match.group(1) if match else None


def download_all_filings():

    # SEC's fair-access policy requires a real name and contact email in the User-Agent of every
    # request, so they come from the environment instead of being hardcoded placeholders
    name = os.getenv("SEC_USER_AGENT_NAME")
    email = os.getenv("SEC_USER_EMAIL")
    if not name or not email:
        raise SystemExit(
            "Set SEC_USER_AGENT_NAME and SEC_USER_EMAIL (your name and contact email, see .env.example) "
            "before downloading from SEC EDGAR."
        )

    # Download into a temporary folder and copy only the wanted fiscal years, so filings for other
    # years never reach data/raw_filings (parsing indexes everything it finds there). Filings
    # that are already present are left untouched, so the script can be rerun safely.
    with tempfile.TemporaryDirectory() as staging:
        downloader = Downloader(company_name=name, email_address=email, download_folder=staging)
        for ticker in TARGET_TICKERS:
            print(f"Downloading 10-K filings for {ticker}...")
            downloader.get("10-K", ticker, limit=FILINGS_PER_TICKER, download_details=True)

            source_dir = os.path.join(staging, "sec-edgar-filings", ticker, "10-K")
            target_dir = os.path.join(DOWNLOAD_DIR, "sec-edgar-filings", ticker, "10-K")
            found = set()
            for accession in sorted(os.listdir(source_dir)):
                year = fiscal_year(os.path.join(source_dir, accession))
                if year not in FISCAL_YEARS:
                    continue
                found.add(year)
                target = os.path.join(target_dir, accession)
                if os.path.exists(target):
                    print(f"  FY{year} {accession}: already present, skipped")
                    continue
                shutil.copytree(os.path.join(source_dir, accession), target)
                print(f"  FY{year} {accession}: added")
            if found != FISCAL_YEARS:
                print(f"  WARNING: {ticker} is missing fiscal years {sorted(FISCAL_YEARS - found)}")
    print("Download complete.")

# We extract 2 files, 1 is with html other with txt. We do parsing for html because of its nice format
# but we will be searching for metadata infos in txt file

if __name__ == "__main__":
    download_all_filings()
