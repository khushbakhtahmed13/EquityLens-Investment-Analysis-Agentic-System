import os
from fastmcp import FastMCP
import yfinance as yf
from tavily import AsyncTavilyClient
from rag_manager import search_docs
import asyncio


# FastMCP SERVER
mcp = FastMCP(
      name = "FinancialToolsServer"
)

# TOOLS
@mcp.tool()
async def retrieve_market_filings(query: str, ticker: str) -> str:
    """
    Retrieves information from 10-K and 10-Q SEC filings based on natural language queries.
    Covers: revenue trends, profitability (EBITDA, margins, net income), risk factors,
    liquidity/capital resources, geographic/segment breakdowns, management discussion.
    Use targeted queries combining financial dimensions with specific angles
    (e.g., "AAPL revenue China exposure", "MSFT risk factors regulatory").
    """
    # Calls the chroma database
    results = await search_docs(query, ticker)

    if not results:
        return "No relevant financial data found in company filings."

    return results


@mcp.tool()
async def get_market_data(ticker: str) -> dict:
    """
    Retrieves current market data and valuation metrics for a stock ticker.
    Returns: price, P/E ratio, EPS, and other key financial ratios.
    Use for assessing current valuation and market positioning.
    """
    # Force uppercase to ensure yfinance compatibility
    symbol = ticker.upper()
    stock = yf.Ticker(symbol)

    try:
        # Fetching the 'info' dict which contains the metrics
        info = await asyncio.to_thread(lambda: stock.info)

        # Check (yfinance sometimes returns empty dicts for invalid tickers)
        if not info or ('regularMarketPrice' not in info and 'currentPrice' not in info):
            return {"error": f"Ticker '{symbol}' not found or no data available."}

        return {
            "ticker": symbol,
            "price": info.get('currentPrice') or info.get('regularMarketPrice'),
            "pe_ratio": info.get('trailingPE'),
            "eps": info.get('trailingEps'),
            "currency": info.get('currency', 'USD')
        }
    except Exception as e:
        return {"error": f"Failed to retrieve data for {symbol}: {str(e)}"}

client = AsyncTavilyClient(api_key = os.getenv("TAVILY_API_KEY"))
@mcp.tool()
async def web_search(query: str) -> str:
    """
    Searches the web for recent news, analyst reports, and market developments.
    Use for: earnings results, analyst upgrades/downgrades, product launches,
    M&A activity, regulatory news, market sentiment, and macro events.
    Supports any time constraint (e.g., "last 30 days", "recent", "2025 Q4").
    """
    response = await client.search(
        query = query,
        search_depth = "advanced", # For better sentiment analysis
        max_results = 5,
        include_answer = True  # Gives a nice summary for the Analyst
    )

    # Extract the AI-generated answer if available, else the snippets
    if response.get("answer"):
        return f"Summary: {response['answer']}\n\nSources: {response['results']}"

    results = [f"- {r['title']}: {r['content']}" for r in response['results']]
    return "\n".join(results)

# RUN SERVER
if __name__ == "__main__":
    mcp.run(
        host = "0.0.0.0",
        port = 8000,
        stateless_http = True,
        transport = "http"
    )
