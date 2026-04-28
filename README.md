## EquityLens Investment Analysis Agentic System
EquityLens is an agentic AI ecosystem that synthesizes SEC filings, real-time market data, and news sentiment to provide customized investment analysis tailored to a client's specific Investment Policy Statement (IPS).

---

### Project Overview

**1. Problem Solving**
It bridges the gap between raw financial data and personalized advice by cross-referencing market volatility and news with a client's specific risk tolerance and portfolio constraints.

**2. End User**
The system serves Wealth Managers and Portfolio Advisors who need to reconcile institutional-grade market data with individual client mandates.

**3. Agent Roles & LLMs**
*   **Researcher:** **Sonnet 4.6** (Excellent tool calling) – Scrapes SEC filings, stock prices, news feeds and client's policy. 
*   **Analyzer:** **Sonnet 4.6** (Smart reasoning) – Performs final comparison between financial fundamentals, sentiment, and the client's IPS to give recommendation.
*   **Summarizer:** **Haiku 4.5** (Fast synthesis) – Condenses research findings into mid-process briefs to validate information before the next phase.

**4. MCP Servers**
These act as the bridge to external tools, allowing agents to execute complex queries across market data providers and local client databases:
*   **FastMCP**: Custom server hosting tools for ChromaDB chunk retrieval, valuation metric calculation, and web search for news/sentiment.
*   **Google Drive MCP**: Specifically used to securely fetch and read the client’s IPS file.

**5. APIs and Purpose**
*   **Financial Data APIs (e.g., Yahoo Finance/Alpha Vantage):** Used to pull real-time stock prices and valuation metrics.
*   **News/Sentiment APIs:** Fetches headlines and product launches to gauge controversies and market momentum.
*   **SEC API**: Programmatically extracts targeted sections (e.g., Risk Factors, MD&A) from 10-K and 10-Q reports.
*   **Anthropic API**: Serves as the primary intelligence engine for reasoning, tool use, and natural language synthesis.

**6. Data Sources**
*   **Multi-Source Feed:** Combines SEC documents (truth) stored in ChromaDB, news/social feeds (sentiment), and price action (metrics).
*   **Investment Policy Statement (IPS):** The primary benchmark for all analysis to ensure compliance with client-specific risks.

**7. LangGraph Workflow Control**

The system employs a Research-to-Summarizer loop: the Researcher pulls data via tools, the Summarizer briefs the findings, and the graph loops back if information is missing. Once complete, it transitions to the Analyzer for the final investment decision.

**8. Memory Types**
*   **Short-term (Thread State):** Holds the current "conversation" and data variables for a specific client query.
*   **Long-term (VectorStoreRetrieverMemory):** Stores historical news events and previous IPS versions to track sentiment shifts and portfolio evolution over time.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Add your API keys to a `.env` file.
3. Run the system: `python main.py`