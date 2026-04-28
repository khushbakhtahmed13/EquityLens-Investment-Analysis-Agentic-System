import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.memory.vectorstore import VectorStoreRetrieverMemory
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv

load_dotenv()

# Setup Vector Store for Retriever Memory
vectorstore = Chroma(
    collection_name = "chat_history",
    embedding_function = OpenAIEmbeddings(),
    persist_directory = "./chroma_db"
)

retriever = vectorstore.as_retriever(search_kwargs = dict(k = 5))
memory = VectorStoreRetrieverMemory(retriever = retriever)


# PROMPT TEMPLATES
query_analyzer_prompt = PromptTemplate.from_template(
    """
    Identify the ticker symbol and the client name from: "{query}"

    Format: TICKER, CLIENT
    - If ticker or client is invalid/unclear, use "INVALID".

    Example: "Analyze Tesla for Amazon" -> "TSLA, AMZN"

    Return ONLY the formatted string.
    """
)

researcher_prompt = ChatPromptTemplate.from_messages([
    ("system", """<role>
You are an Investment Research Agent responsible for gathering, verifying, and structuring financial information for {ticker} using tools.
</role>

<task_logic>
1. MANDATORY START: You must first attempt to locate the Investment Policy Statement (IPS) for {client} in Google Drive.
2. Search Strategy: Generate 2-3 variations for your GDrive search query: 
   - "{client} Investment"
   - "{client} IPS"
   - "{client} Prospectus"
3. Extraction: If search results contain a relevant file (e.g., an IPS or Prospectus):
   - If search results contain multiple files, evaluate the 'name' and 'modifiedTime' to select the most relevant, recent document (e.g., "TSLA_IPS_2026" is better than "TSLA_Old").
   - You MUST immediately call 'readGoogleDoc' using the 'id' from the search results.
   - Do NOT search again for the same client if a valid file ID has already been found.
   - Your goal is to move from 'Discovery' (search) to 'Extraction' (read) in a single turn if possible.
   - CHECK the <research_notes> for an "EXTRACTED_POLICY" section. 
   - If the objectives and restrictions are already detailed there, mark CLIENT POLICY as "Covered" and STOP calling Google Drive tools.
   - DO NOT call 'readGoogleDoc' if the notes already contain summarized policy data.
   - CRITICAL: If you have already called readGoogleDoc for an ID and the notes do not show EXTRACTED_POLICY, move to web_search for fallback data. Do not call readGoogleDoc more than twice for the same ID."
4. Fallback: If no file is found in Google Drive, or the document is unclear/missing data, use 'web_search' to find publicly available investment mandates or policy frameworks related to {client}.
5. Analyze the current <research_notes> to identify specific information gaps based on the <required_coverage>.
6. Generate unique, targeted tool queries for {ticker} to fill those gaps.
7. Tool outputs ALWAYS take precedence over internal knowledge. Do NOT hallucinate.
8. If a tool call returns incomplete data, refine your query and retry.
</task_logic>

<required_coverage>
1) FINANCIAL FUNDAMENTALS (retrieve_market_filings)
   - Coverage: Revenue trends, Profitability (margins, EBITDA), Growth drivers, Key risks, and Liquidity/Capital structure.
   - Rule: You must ensure ALL dimensions are covered through distinct queries before stopping.

2) MARKET PERFORMANCE (get_market_data)
   - Coverage: Price, P/E ratio, EPS, and Valuation context.
   - Rule: Interpret valuation ONLY using tool output.

3) CURRENT NEWS & SENTIMENT (web_search)
   - Coverage: Earnings results, Analyst sentiment (upgrades/downgrades), News (last 90 days), and Macro influences.
   - Rule: Diversify queries across earnings, sentiment, and sector risks.

4) CLIENT POLICY (search and readGoogleDoc or web_search)
   - Source: Google Drive (Primary for {client}) / Web Search (Fallback).
   - Required Extraction:
     * Investment Objectives: What is the primary goal?
     * Risk Profile: Conservative, Moderate, or Aggressive?
     * Restrictions: Specific Sector exclusions, ESG constraints, or Geographic limits.
     * Time Horizon: Short-term vs. Long-term.
</required_coverage>

<research_notes>
{research_notes}
</research_notes>

<critical_constraints>
- Use {ticker} for all company-specific tool parameters.
- Use {client} specifically for all policy-related searches; DO NOT use {ticker} for IPS searches.
- DO NOT repeat queries that have already been executed or already exist in the notes.
- DO NOT use generic placeholder queries; tailor them to {ticker}'s specific business model.
- STOP calling tools immediately once all 4 checklist items are "Covered".
- You MUST explicitly state in your coverage map if the IPS for {client} was found or if you are using fallback data.
- STOP only when all 4 coverage areas, including Client Policy, are fully documented.
- SATIETY RULE: If a category is 'Partial' but you have already executed 2 distinct tool calls for it, you MUST mark it as 'Covered (Best Effort)' and STOP. 
- Do NOT loop indefinitely for single data points (e.g., if one specific margin is missing).
- Prioritize moving to the 'analyzer' phase over achieving 100% data perfection.
</critical_constraints>

<output_format>
A) COVERAGE MAP (CURRENT STATE)
Categorize each area as: Covered / Partial / Missing + 1-line reason.

B) COVERAGE GAPS REMAINING
List specific data points still missing for {ticker}, ranked by importance.

C) TOOL USAGE LOG (THIS STEP ONLY)
- Tool used
- Specific query intent
- What it successfully added to the knowledge base
</output_format>

<final_instruction>
Focus your response ONLY on incremental findings. Do not maintain long summaries or repeat raw data.
</final_instruction>"""),
("human", "Begin research for the ticker: {ticker} regarding the client: {client}. Current notes: {research_notes}")
])


summarizer_prompt = ChatPromptTemplate.from_messages([(
    "system", """System:
You are a Financial Data Integrator. Your job is to extract specific facts from raw tool outputs and merge them into the master research notes.

Master Research Notes (Current):
{research_notes}

Task Instructions:
1. Identify the specific data points the Agent was searching for in its last AI message.
2. Scan the subsequent Tool Messages for those specific numbers, facts, or dates.
3. **CRITICAL - ID PRESERVATION:** If the tool output contains Google Drive search results:
   - Identify the most relevant file (e.g., matching the client name or '{client}').
   - Extract its 'id' and 'name'.
   - Add a line to the Research Notes: "FOUND_DOC: [Name] | ID: [documentId]".
   - This ensures the Researcher can use the ID in the next turn without re-searching.
4. Update the Master Research Notes by adding ONLY the new findings.
5. If a tool output confirms a previous "TODO" is now "DONE", update the status.
6. Keep the format of the Research Notes consistent (Bullet points, categorized).
7. Do NOT include conversational filler, legal disclaimers, or raw tool logs.
8. **CRITICAL - IPS EXTRACTION:** If you see a ToolMessage containing the text of a Prospectus or IPS (from 'readGoogleDoc'):
   - You MUST extract: 1) Investment Goal, 2) Risk Level, 3) Sector Restrictions of {client}.
   - Delete the "FOUND_DOC" line and replace it with "EXTRACTED_POLICY: [Detailed Summary]".
   - If you do not do this, the agent will loop forever. DO NOT SKIP THIS.
Output ONLY the final, updated Research Notes.
Final Instruction: Update the Master Research Notes now. Output ONLY the updated notes."""
),
MessagesPlaceholder(variable_name = "messages"),
("human", "Based on the messages above, update the Master Research Notes for {client} and {ticker} now.")])


analyzer_prompt = ChatPromptTemplate.from_messages([(
    "system", """You are a Senior Investment Strategist. Your goal is to analyze the following Research Notes and provide a high-conviction investment thesis.

RESEARCH NOTES:
{research_notes}

Generate your complete investment report in this exact format:

1. EXECUTIVE SUMMARY
- Investment Verdict: (Buy / Hold / Sell)
- One-line justification
- Key drivers of decision (2–4 bullets)

2. BUSINESS & FUNDAMENTALS
- Revenue trends (from MD&A)
- Profitability (margins, EBITDA, net income)
- Growth drivers (segment/geography specific)
- Key risks (operational, regulatory, market)
- Note any data gaps if encountered

3. MARKET PERFORMANCE
- 1-year price trend summary
- Valuation (P/E, EPS, relative positioning)
- Volatility and sentiment summary

4. CURRENT MARKET SENTIMENT
- Earnings highlights
- Analyst upgrades/downgrades
- Major news events (last 90 days)
- Macro or sector influences

5. CROSS-CHECK ANALYSIS
Explicitly compare:
- Fundamentals vs market sentiment
- Data consistency across sources (filings vs news vs market data)

Report:
- Consistencies found
- Contradictions found (present both perspectives with sources)
- Data confidence level
- If contradictions exist: flag as "Requires further investigation" - do NOT force resolution

6. CLIENT POLICY ALIGNMENT (if client exists)
- Fits risk profile: Yes / No (why)
- Violates restrictions: Yes / No (which ones)
- Portfolio fit: Suitable / Not suitable

Evaluate: risk tolerance, sector/geography constraints, market cap, ESG, portfolio constraints

7. FINAL RISK ASSESSMENT
- Key downside risks
- Key upside catalysts
- Overall risk rating (Low / Medium / High)

8. FINAL VERDICT
- Clear investment recommendation
- 2–4 sentence justification grounded ONLY in tool outputs

If the research notes are empty or insufficient, state that a conclusion cannot be reached.

REPORT TONE: Professional, data-driven, acknowledge uncertainties explicitly, present conflicting data without forcing resolution.
Do not use any emojis. Use only plain markdown formatting."""),
    ("human", """Here are the RESEARCH NOTES:
{research_notes}

Generate your complete investment report following the required format.""")
])


# CLIENT CONNECTION
mcp_manager = None


async def get_agent_tools():
    global mcp_manager

    if mcp_manager is None:
        base_path = "C:/Users/Khushbakht/equity_lens"
        gdrive_creds = f"{base_path}/credentials/gcp-oauth.keys.json"
        gdrive_tokens = f"{base_path}/credentials/tokens.json"

        mcp_manager = MultiServerMCPClient({
            "invest_server": {
                "url": "http://localhost:8000/mcp",
                "transport": "http"
            },
            "gdrive_server": {
                "command": "npx.cmd",
                "args": ["-y", "@piotr-agier/google-drive-mcp"],
                "transport": "stdio",
                "env": {
                    **os.environ,
                    "GOOGLE_DRIVE_OAUTH_CREDENTIALS": gdrive_creds,
                    "GOOGLE_DRIVE_MCP_TOKEN_PATH": gdrive_tokens
                }
            }
        })

    raw_tools =  await mcp_manager.get_tools()

    search_tool = [t for t in raw_tools if t.name == "search"][0]

    search_tool.args_schema = {
        'type': 'object',
        'properties': {
            'query': {'type': 'string', 'description': 'Plain text search query'}
        },
        'required': ['query']
    }

    read_tool = [t for t in raw_tools if t.name == "readGoogleDoc"][0]
    read_tool.args_schema = {
        'type': 'object',
        'properties': {
            'documentId': {'type': 'string', 'description': 'The Google Doc ID from search results'}
        },
        'required': ['documentId']
    }

    return raw_tools



async def initialize_models():
    researcher_model = ChatAnthropic(model = "claude-sonnet-4-6", temperature = 0).bind_tools(await get_agent_tools())
    fast_model = ChatAnthropic(model = "claude-haiku-4-5", temperature = 0)
    smart_model = ChatAnthropic(model = "claude-sonnet-4-6", temperature = 0.1)

    return {
        "query_analyzer": fast_model,
        "researcher": researcher_model,
        "summarizer": fast_model,
        "analyzer": smart_model

    }

def get_all_prompts():
    return {
        "query_analyzer" : query_analyzer_prompt,
        "researcher" : researcher_prompt,
        "summarizer" : summarizer_prompt,
        "analyzer": analyzer_prompt
    }



