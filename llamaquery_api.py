from fastapi import FastAPI, Request
import os
import asyncio
import httpx
import uvicorn

from llama_index.core import Settings, RouterQueryEngine
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.cloud import LlamaCloudIndex  # <-- updated import


app = FastAPI(
    title="The Beast Router API",
    description=(
        "Routes user queries intelligently between two Llama Cloud indices — "
        "'SharePoint Deal Pipeline' and 'SharePoint Thematic Work' — using an "
        "LLM-based RouterQueryEngine."
    ),
    version="3.0.0",
)


def create_query_engine(index: LlamaCloudIndex) -> RetrieverQueryEngine:
    retriever = index.as_retriever()
    return RetrieverQueryEngine(retriever=retriever)


@app.post("/llamaquery")
async def llamaquery(request: Request):
    """Routes query between indices using an LLM-based router."""
    data = await request.json()
    query = data.get("query")

    if not query:
        return {"error": "Missing 'query' in request body"}

    llama_api_key = os.getenv("LLAMA_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not llama_api_key or not openai_api_key:
        return {"error": "Missing LLAMA_API_KEY or OPENAI_API_KEY in environment"}

    timeout = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            # --- Connect to both indices ---
            deal_index = LlamaCloudIndex(
                name="SharePoint Deal Pipeline",
                project_name="The BEAST",
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

            # --- Configure router LLM ---
            Settings.llm = OpenAI(model="gpt-4o-mini", api_key=openai_api_key)

            # --- Build query engines ---
            deal_engine = create_query_engine(deal_index)
            thematic_engine = create_query_engine(thematic_index)

            # --- Build RouterQueryEngine ---
            router_engine = RouterQueryEngine.from_defaults(
                query_engine_tools=[
                    {
                        "query_engine": deal_engine,
                        "description": (
                            "Useful for company-specific questions, diligence docs, "
                            "financials, and deal materials."
                        ),
                    },
                    {
                        "query_engine": thematic_engine,
                        "description": (
                            "Useful for market, sector, or thematic research and newsletters."
                        ),
                    },
                ]
            )

            # --- Execute routed query ---
            response = await asyncio.to_thread(router_engine.query, query)
            selected_tool = (
                response.metadata.get("selectorResult", {}).get("selected_tool", "Unknown")
            )

            return {
                "query": query,
                "selected_index": selected_tool,
                "response": response.response.strip(),
                "metadata": response.metadata,
            }

        except Exception as e:
            return {"error": f"Router query failed: {str(e)}"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
