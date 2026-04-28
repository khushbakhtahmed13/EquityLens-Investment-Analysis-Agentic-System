import hashlib
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# PERSISTENT DATABASE
DB_PATH = "./chroma_db"
embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")

vector_db = Chroma(
    persist_directory = DB_PATH,
    embedding_function = embeddings,
    collection_name = "market_filings"
)


async def search_docs(query: str, ticker: str) -> str:
    """
    The Retriever: Uses MMR to find 5 diverse chunks.
    """
    # Configure retriever with MMR
    retriever = vector_db.as_retriever(
        search_type = "mmr",
        search_kwargs = {
            "k": 10,  # Final number of chunks
            "fetch_k": 20,  # Candidates to pick from for diversity
            "lambda_mult": 0.5,  # 0.5 is balanced diversity
            "filter": {"ticker": ticker}
        }
    )

    docs = await retriever.ainvoke(query)

    if not docs:
        return ""

    # Join the page content of all retrieved chunks
    return "\n\n---\n\n".join([doc.page_content for doc in docs])


def add_documents_to_db(ticker: str, text: str, metadata: dict = None):
    """
    The Ingestor: Chunks text, generates unique IDs, and saves to Chroma.
    """
    # SPLITTER
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 512,
        chunk_overlap = 100,
        add_start_index = True  # Helpful for tracking original positions
    )

    # DOCUMENT
    base_doc = Document(page_content = text, metadata = metadata or {"ticker": ticker})
    chunks = text_splitter.split_documents([base_doc])

    # Generate Unique IDs to prevent duplicates
    # Hash the content + ticker so the same chunk won't be added twice
    ids = []
    for chunk in chunks:
        identifier = f"{ticker}_{hashlib.md5(chunk.page_content.encode()).hexdigest()}"
        ids.append(identifier)
        chunk.metadata.update(base_doc.metadata)  # Ensure metadata for filtering

    # Add to DB
    vector_db.add_documents(documents = chunks, ids = ids)
    return f"Successfully added {len(chunks)} chunks for {ticker}."