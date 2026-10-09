import os

import pytest

# app.main reads these at import time
os.environ["BACKEND_API_KEY"] = "test-key"
os.environ["TRUST_PROXY_HEADERS"] = "true"


@pytest.fixture
def catalog():
    from app.router.router_agent import IndexCatalog
    return IndexCatalog(companies=("AAPL", "MSFT"), years=("2024", "2025"),
                        sections=("Item 1", "Item 1A", "Item 1C", "Item 3", "Item 7", "Item 8"))
