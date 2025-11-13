from fastapi import FastAPI, Request
from llama_cloud_services import (
    LlamaCloudIndex,
    LlamaCloudCompositeRetriever,
)
from llama_cloud import CompositeRetrievalMode
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.schema import NodeWithScore, Node
from llama_index.llms.openai import OpenAI
import httpx
import asyncio
import os
import uvicorn


# Initialize FastAPI app
app = FastAPI(
    title="The Beast API",
    description=(
        "Composite Retrieval + LLM Response API. "
        "Combines retrieval from SharePoint Deal Pipeline and Thematic indices "
        "and synthesizes a GPT-5 response."
    ),
    version="2.1.0",
)


@app.post("/llamaquery")
async def llamaquery(request: Request):
    """
    Handles queries across multiple Llama Cloud indices, synthesizes an LLM answer
    using GPT-5 via LlamaIndex response synthesizer.
    """
    data = await request.json()
    query = data.get("query")
    query = query + " return all citations with web_url and file_name"
    if not query:
        return {"error": "Missing 'query' in request body"}

    llama_api_key = os.getenv("LLAMA_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not llama_api_key or not openai_api_key:
        return {"error": "Missing one or more required API keys"}

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
            project_name=project_name,
            organization_id="8ff953cd-9c16-49f2-93a4-732206133586",
            api_key=llama_api_key,
            client=client,
        )

        # Composite retriever (FULL mode for merged retrieval)
        composite_retriever = LlamaCloudCompositeRetriever(
            name="The Beast Composite Retriever",
            project_name=project_name,
            organization_id="8ff953cd-9c16-49f2-93a4-732206133586",
            api_key=llama_api_key,
            client=client,
            create_if_not_exists=True,
            mode=CompositeRetrievalMode.FULL,
            rerank_top_n=6,
        )

        composite_retriever.add_index(
            deal_index,
            description="Deal-specific materials (data rooms, decks, diligence).",
        )
        composite_retriever.add_index(
            thematic_index,
            description="Market research and sector analyses supporting deal context.",
        )

        # Retry for transient retrieval issues
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

    if not nodes:
        return {"query": query, "response": "No relevant documents found.", "results": []}

    # Initialize GPT-5 LLM
    llm = OpenAI(model="gpt-5", api_key=openai_api_key, temperature=0.2)

    # Initialize response synthesizer (compact mode)
    response_synthesizer = get_response_synthesizer(
        llm=llm,
        response_mode=ResponseMode.REFINE,
        structured_answer_filtering=True,
    )

    # Generate synthesized answer
    synthesized = await asyncio.to_thread(
        response_synthesizer.synthesize, query, nodes
    )
    print(synthesized.response)
    # Structure chunk-level metadata
    results = []
    for node in nodes:
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
            {"text": text, "file_name": file_name, "web_url": web_url}
        )

    return {
        "query": query,
        "response": synthesized.response  # synthesized LLM answer
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
