# Run from the repo root: python -m app.ingestion.download_filings
import os
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from sec_edgar_downloader import Downloader

load_dotenv()


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
    downloader = Downloader(
        company_name=name,
        email_address=email,
        download_folder="./data/raw_filings"
    )

    target_tickers = ["AAPL", "MSFT"] # Apple and Microsoft so far, change as needed

    for ticker in target_tickers:
        print(f"Downloading 10-K filings for {ticker}...")
        downloader.get("10-K", ticker, limit=2, download_details=True)  # limit: is for how many latest pdf you wanna see. We did 2 so we can see 2025&2024
    print("Download complete.")

# We extract 2 files, 1 is with html other with txt. We do parsing for html because of its nice format 
# but we will be searching for metadata infos in txt file

if __name__ == "__main__":
    download_all_filings()
