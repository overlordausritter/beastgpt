from fastapi import FastAPI, Request
from llama_cloud_services import LlamaCloudIndex
import httpx
import asyncio
import os
import uvicorn

app = FastAPI(
    title="The Beast API",
    description="API for querying the SharePoint Deal Pipeline index via Llama Cloud.",
    version="1.0.0",
)

@app.post("llamaquery")
async def llamaquery(request: Request):
    """
    Handles general queries against the LlamaIndex index.
    Returns structured text chunks with their source metadata.
    """
    data = await request.json()
    query = data.get("query")
    if not query:
        return {"error": "Missing 'query' in request body"}

    llama_api_key = os.getenv("LLAMA_API_KEY")
    if not llama_api_key:
        return {"error": "Missing LLAMA_API_KEY environment variable"}

    timeout = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        index = LlamaCloudIndex(
            name="Sharepoint Deal Pipeline",
            project_name="The BEAST",
            organization_id="8ff953cd-9c16-49f2-93a4-732206133586",
            api_key=llama_api_key,
            client=client,
        )

        for attempt in range(3):
            try:
                retriever = index.as_retriever()
                nodes = await asyncio.to_thread(retriever.retrieve, query)
                break
            except (httpx.RemoteProtocolError, httpx.ReadTimeout) as e:
                if attempt < 2:
                    print(f"Retry {attempt + 1}/3 after error: {e}")
                    await asyncio.sleep(2)
                else:
                    return {"error": f"Llama Cloud connection failed: {str(e)}"}

    if not nodes:
        return {"query": query, "results": [], "message": "No relevant documents found."}

    # Build structured results and concatenated text
    results = []
    combined_text = ""
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
        combined_text += text + "\n"
        results.append({
            "text": text,
            "file_name": file_name,
            "web_url": web_url,
        })

    return {
        "query": query,
        "text": combined_text.strip(),  # new field matching your spec
        "results": results
    }

@app.get("/")
def healthcheck():
    return {"message": "The Beast API is running", "endpoint": "/llamaquery"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
