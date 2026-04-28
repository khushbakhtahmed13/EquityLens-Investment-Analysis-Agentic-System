import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from data_ingestion import process_ticker
from Graph import run_pipeline


async def run_full_audit(ticker, query):
    if not ticker or not query:
        yield "Error: Please enter both a Ticker and a Research Query."
        return

    yield f"🔍Fetching and indexing data for {ticker.upper()}..."
    try:
        # Step 1: Data Ingestion
        process_ticker(ticker.upper())

        yield f"🧠Data ready. Analyzing '{query}' using LangGraph..."

        # Step 2: Agentic Analysis
        result = await run_pipeline(query = query)
        final_answer = result["messages"][-1].content

        yield final_answer

    except Exception as e:
        yield f"❌ Pipeline Error: {str(e)}"


def launch_ui():
    with gr.Blocks(title = "EquityLens Analysis", theme = gr.themes.Soft()) as demo:
        gr.Markdown("# 🏢 EquityLens Investment Analysis Agent")

        with gr.Row():
            # Input Column
            with gr.Column(scale = 1):
                ticker_input = gr.Textbox(
                    label = "Stock Ticker",
                    placeholder = "e.g., NVDA, ARKK..."
                )
                query_input = gr.Textbox(
                    label = "Query",
                    placeholder = "Enter your thesis or audit request...",
                    lines = 5
                )

                with gr.Row():
                    clear_btn = gr.Button("Clear")
                    run_btn = gr.Button("Start Analysis", variant="primary")

                gr.Examples(
                    examples = [
                        ["NVDA", "Analyze NVIDIA for ARKK."],
                        ["AAPL", "Supply chain risks in the latest 10-K."],
                    ],
                    inputs = [ticker_input, query_input]
                )

            # Output Column (The "Report" Block)
            with gr.Column(scale = 2):
                gr.Markdown("### 📊 Investment Research Report")
                with gr.Group():
                    output_display = gr.Markdown(
                        value = "_Results will appear here after analysis starts..._"
                    )

        # Logic
        run_btn.click(
            fn = run_full_audit,
            inputs = [ticker_input, query_input],
            outputs = output_display
        )

        # Button
        clear_btn.click(
            lambda: ("", "", "_Results will appear here after analysis starts..._"),
            outputs = [ticker_input, query_input, output_display]
        )

    demo.launch()

if __name__ == "__main__":

    launch_ui()