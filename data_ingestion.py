import os
from sec_api import QueryApi, ExtractorApi
from rag_manager import add_documents_to_db, vector_db
from dotenv import load_dotenv

load_dotenv()

query_api = QueryApi(api_key = os.getenv("SEC_API_KEY"))
extractor_api = ExtractorApi(api_key = os.getenv("SEC_API_KEY"))


def is_data_ingested(ticker: str, form_type: str) -> bool:
    """Check Chroma for existing ticker + formType metadata."""
    # Filter to find if any chunks exist
    results = vector_db.get(
        where = {"$and": [{"ticker": ticker}, {"form_type": form_type}]},
        limit = 1
    )
    return len(results["ids"]) > 0


def process_ticker(ticker: str):
    sections = {
        "10-K": ["7", "1A"],  # MD&A, Risks
        "10-Q": ["part1item2", "part2item1a"]  # MD&A, Risks
    }

    for form_type, items in sections.items():
        # Skip if already in DB
        if is_data_ingested(ticker, form_type):
            print(f"--- Skipping {ticker} {form_type}: Already Ingested ---")
            continue

        # Get Latest Filing URL
        query = f"ticker:{ticker} AND formType:\"{form_type}\""
        response = query_api.get_filings({"query": query, "size": "1", "sort": [{"filedAt": {"order": "desc"}}]})

        if not response['filings']:
            continue

        filing_url = response['filings'][0]['linkToFilingDetails']

        # Extract and Add
        for item in items:
            print(f"--- Extracting {ticker} {form_type} Item {item} ---")
            content = extractor_api.get_section(filing_url, item, "text")

            if content:
                # Add to DB using rag_manager logic
                metadata = {"ticker": ticker, "form_type": form_type, "item": item}
                add_documents_to_db(ticker, content, metadata = metadata)

    return f"Finished processing {ticker}"