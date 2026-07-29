import os
# pyrefly: ignore [missing-import]
from sec_edgar_downloader import Downloader


def download_all_filings():

    # Our company infos to match Sec Edgars policies
    downloader = Downloader(
        company_name="MultiAgentFinancialRAG",
        email_address="admin@financialrag.com",
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
