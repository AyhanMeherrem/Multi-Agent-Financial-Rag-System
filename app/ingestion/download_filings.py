import os
# pyrefly: ignore [missing-import]
from sec_edgar_downloader import Downloader

downloader = Downloader(
    company_name="MultiAgentFinancialRAG",
    email_address="[EMAIL_ADDRESS]",
    download_folder="./data/raw_filings"
)

target_tickers = ["AAPL", "MSFT"]         # Apple and Microsoft so far, change as needed


for ticker in target_tickers:
    print(f"Downloading 10-K filings for {ticker}...")
    downloader.get("10-K", ticker, limit=2, download_details=True)
print("Download complete.")
