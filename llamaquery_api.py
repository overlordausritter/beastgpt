from fastapi import FastAPI, Request
from llama_cloud_services import (
    LlamaCloudIndex,
    LlamaCloudCompositeRetriever,
)
from llama_cloud import CompositeRetrievalMode
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.prompts import PromptTemplate
import httpx
import asyncio
import os
import uvicorn


# Initialize FastAPI app
app = FastAPI(
    title="The Beast API",
    description=(
        "Composite Retrieval API combining deal and thematic indices. "
        "Queries across both Llama Cloud indices: Sharepoint Deal Pipeline and SharePoint Thematic Work."
    ),
    version="2.0.0",
)


@app.post("/llamaquery")
async def llamaquery(request: Request):
    """
    Handles queries against both LlamaIndex indices using Composite Retrieval (FULL mode).
    Returns structured text chunks and metadata.
    """
    data = await request.json()
    query = data.get("query")
    if not query:
        return {"error": "Missing 'query' in request body"}

    llama_api_key = os.getenv("LLAMA_API_KEY")
    if not llama_api_key:
        return {"error": "Missing LLAMA_API_KEY environment variable"}

    # Custom HTTP client with extended timeouts
    timeout = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        project_name = "The BEAST"

        # Initialize both indices
        deal_index = LlamaCloudIndex(
            name="Sharepoint Deal Pipeline",
            project_name=project_name,
            organization_id="8ff953cd-9c16-49f2-93a4-732206133586",
            api_key=llama_api_key,
            client=client,
        )

        thematic_index = LlamaCloudIndex(
            name="SharePoint Thematic Work",
            project_name="The BEAST",
            organization_id="8ff953cd-9c16-49f2-93a4-732206133586",
            api_key=llama_api_key,
            client=client,
        )

        # Composite retriever
        composite_retriever = LlamaCloudCompositeRetriever(
            name="The Beast Composite Retriever",
            project_name="The BEAST",
            organization_id="8ff953cd-9c16-49f2-93a4-732206133586",
            api_key=llama_api_key,
            client=client,
            create_if_not_exists=True,
            mode=CompositeRetrievalMode.FULL,
            rerank_top_n=6,
        )

        composite_retriever.add_index(
            deal_index,
            description="Deal-specific materials such as data rooms, pitch decks, and company diligence files.",
        )
        composite_retriever.add_index(
            thematic_index,
            description="Market research, news, and sectoral analysis supporting deal context.",
        )

        # Retry logic for transient network issues
        for attempt in range(3):
            try:
                nodes = await asyncio.to_thread(composite_retriever.retrieve, query)
                break
            except (httpx.RemoteProtocolError, httpx.ReadTimeout) as e:
                if attempt < 2:
                    print(f"Retry {attempt + 1}/3 after error: {e}")
                    await asyncio.sleep(2)
                else:
                    return {"error": f"Llama Cloud connection failed: {str(e)}"}

    # Build structured chunk-level results
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

        results.append(
            {
                "text": text,
                "file_name": file_name,
                "web_url": web_url,
            }
        )

    # --- Truncate overly long text using LlamaIndex helpers ---
    text_chunks = [r["text"] for r in results if r["text"]]
    prompt_helper = PromptHelper(context_window=32000, num_output=1024)
    prompt = PromptTemplate("Your prompt: {text}")
    truncated_chunks = prompt_helper.truncate(prompt=prompt, text_chunks=text_chunks)

    combined_text = "\n".join(truncated_chunks)

    return {
        "query": query,
        "text": combined_text.strip(),
        "results": results,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
