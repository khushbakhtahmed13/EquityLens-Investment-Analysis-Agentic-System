from typing import TypedDict, Annotated, Sequence

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage
from langchain_core.tracers import ConsoleCallbackHandler

from agents import initialize_models, get_agent_tools, get_all_prompts, memory


# STATE
class AgentState(TypedDict):
    ticker:               str
    client:               str
    messages:             Annotated[Sequence[BaseMessage], add_messages]
    relevant_history:     str
    research_notes:       str



# MEMORY

# Buffer Memory — MemorySaver checkpoints full message history per thread_id
checkpointer = MemorySaver()


# BUILD GRAPH

async def build_graph():

    prompts = get_all_prompts()
    models = await initialize_models()
    tools = await get_agent_tools()


    # NODE FUNCTIONS
    def query_analyzer(state: AgentState):
        """Validates user query"""
        prompt = prompts["query_analyzer"]
        llm = models["query_analyzer"]
        chain = prompt | llm

        query = state["messages"][-1].content
        history = memory.load_memory_variables({"prompt": query}) # Search memory for context related to the current query
        response = chain.invoke({"query": query}).content
        parts = [p.strip() for p in response.split(",")]
        ticker = parts[0]
        client = parts[1]

        if ticker == "INVALID":
            return {
                "messages": [AIMessage("Please provide a valid company name or ticker symbol.")]
            }

        return {"ticker": ticker,
                "client": client,
                "relevant_history": history.get("history", "")}


    def researcher(state: AgentState):
        prompt = prompts["researcher"]
        llm = models ["researcher"]
        chain = prompt | llm

        response = chain.invoke(state)

        return {"messages": [response]}

    def summarizer(state: AgentState):
        prompt = prompts["summarizer"]
        llm = models["summarizer"]
        chain = prompt | llm

        last_messages = []
        for msg in reversed(state["messages"]):
            last_messages.append(msg)
            if isinstance(msg, AIMessage) and msg.tool_calls:
                break
        last_messages.reverse()

        updated_notes = chain.invoke(
            {
                "research_notes": state["research_notes"],
                "messages": last_messages,
                "client": state["client"]
            }
        ).content

        # Clear message history
        messages = state["messages"]
        delete_cmds = [
            RemoveMessage(id = m.id)
            for m in messages
            if not isinstance(m, HumanMessage)
        ]

        return {
            "research_notes": updated_notes,
            "messages": delete_cmds + [
             AIMessage(content = "I have updated the research notes with the latest findings. I will now evaluate the remaining gaps.")]
        }

    def analyzer(state):
        prompt = prompts["analyzer"]
        llm = models["analyzer"]
        chain = prompt | llm

        report_message = chain.invoke(state)

        return {"messages": [report_message]}


    # INITIALIZE TOOL NODES
    tool_node = ToolNode(tools)


    # ROUTER FUNCTIONS
    def route_query_analyzer(state: AgentState) -> str:
        ticker = state.get("ticker")

        # If no ticker OR ticker is invalid
        if not ticker or ticker == "INVALID":
            return END

        return "researcher"

    def route_researcher(state):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "analyzer"


    # BUILD GRAPH
    builder = StateGraph(AgentState)

    # ADD NODES
    builder.add_node("query_analyzer", query_analyzer)
    builder.add_node("researcher", researcher)
    builder.add_node("summarizer", summarizer)
    builder.add_node("analyzer", analyzer)
    builder.add_node("tools", tool_node)

    # ENTRY POINT
    builder.set_entry_point("query_analyzer")

    # EDGES
    builder.add_conditional_edges(
        "query_analyzer",
        route_query_analyzer,
        {
            "researcher": "researcher",
            END: END
        }
    )

    builder.add_conditional_edges(
        "researcher",
        route_researcher,
        {
            "tools": "tools",
            "analyzer": "analyzer"
        }
    )

    builder.add_edge("tools", "summarizer")
    builder.add_edge("summarizer", "researcher")

    builder.add_edge("analyzer", END)

    # COMPILE
    return builder.compile(checkpointer = checkpointer)


# RUN
async def run_pipeline(query: str):

    app = await build_graph()


    initial_state = {
        "messages": [HumanMessage(content = query)],
        "research_notes": "",
        "ticker": "",
        "client": "",
        "relevant_history": ""
    }

    config = {
        "configurable": {"thread_id": "analysis_01"},
        #"recursion_limit": 20,
        "callbacks": [ConsoleCallbackHandler()]
    }

    print(f"--- Starting Analysis ---")
    result = await app.ainvoke(initial_state, config = config)
    memory.save_context(
        {"input": query},
        {"output": result.get("research_notes", "Analysis complete.")}
    )

    return result




