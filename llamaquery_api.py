from fastapi import FastAPI, Request
from llama_cloud_services import LlamaCloudIndex
from llama_index.core import RouterRetriever
from llama_index.core.tools import RetrieverTool
from llama_index.core.selectors import PydanticSingleSelector
from llama_index.llms.openai import OpenAI
import httpx
import asyncio
import os
import uvicorn
import logging

# -------------------------------
# Setup Logging & App Metadata
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("router_api")

app = FastAPI(
    title="The Beast Router API",
    description=(
        "Routes queries between multiple Llama Cloud Indexes using an AI Router. "
        "Automatically selects the most relevant retriever."
    ),
    version="2.0.0",
)

# -------------------------------
# Environment Configuration
# -------------------------------
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not LLAMA_API_KEY:
    raise ValueError("Missing LLAMA_API_KEY environment variable")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY environment variable")

# -------------------------------
# Initialize LLM Router
# -------------------------------
llm = OpenAI(model="gpt-4-turbo", api_key=OPENAI_API_KEY)

# Custom HTTP client (shared)
timeout = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)
http_client = httpx.AsyncClient(timeout=timeout)


# -------------------------------
# Lazy Index Loader
# -------------------------------
def build_index(name: str) -> LlamaCloudIndex:
    """Factory function for Llama Cloud indexes."""
    return LlamaCloudIndex(
        name=name,
        project_name="The BEAST",
        organization_id="8ff953cd-9c16-49f2-93a4-732206133586",
        api_key=LLAMA_API_KEY,
        client=http_client,
    )


# -------------------------------
# Define Retriever Tools
# -------------------------------
deal_index = build_index("Sharepoint Deal Pipeline")
work_index = build_index("SharePoint Thematic Work")

deal_tool = RetrieverTool.from_defaults(
    retriever=deal_index.as_retriever(),
    description=(
        "Use for queries about company pipelines, deals, transactions, or financial diligence materials."
    ),
)

work_tool = RetrieverTool.from_defaults(
    retriever=work_index.as_retriever(),
    description=(
        "Use for thematic research, strategy notes, and market intelligence content."
    ),
)

# -------------------------------
# Build Router Retriever
# -------------------------------
router_retriever = RouterRetriever(
    selector=PydanticSingleSelector.from_defaults(llm=llm),
    retriever_tools=[deal_tool, work_tool],
)

# -------------------------------
# API Endpoint
# -------------------------------
@app.post("/llamaquery")
async def llamaquery(request: Request):
    """
    Routes the incoming query to the most relevant Llama Cloud Index.
    Returns structured text chunks and metadata.
    """
    data = await request.json()
    query = data.get("query")
    if not query:
        return {"error": "Missing 'query' in request body"}

    logger.info(f"Received query: {query}")

    # Retry logic for transient API issues
    for attempt in range(3):
        try:
            nodes = await asyncio.to_thread(router_retriever.retrieve, query)
            break
        except (httpx.RemoteProtocolError, httpx.ReadTimeout) as e:
            if attempt < 2:
                logger.warning(f"Retry {attempt + 1}/3 after error: {e}")
                await asyncio.sleep(2)
            else:
                return {"error": f"Llama Cloud connection failed: {str(e)}"}

    # Build structured results
    results = []
    for node in nodes or []:
        node_obj = getattr(node, "node", node)
        metadata = getattr(node_obj, "metadata", {}) or {}
        file_name = (
            metadata.get("file_name")
            or metadata.get("filename")
            or metadata.get("document_title")
        )
        web_url = metadata.get("web_url")
        text = getattr(node, "text", "")

        results.append({
            "text": text,
            "file_name": file_name,
            "web_url": web_url,
        })

    # Combine text chunks into a single concatenated string
    combined_text = "\n".join([r["text"] for r in results if r["text"]])

    logger.info(f"Returning {len(results)} chunks for query: {query}")

    return {
        "query": query,
        "text": combined_text.strip(),
        "results": results,
    }


# -------------------------------
# Entrypoint
# -------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
