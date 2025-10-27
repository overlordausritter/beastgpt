from fastapi import FastAPI, Request
from llama_cloud_services import LlamaCloudIndex
import httpx
import asyncio
import os
import uvicorn
import re

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

app = FastAPI()


def normalize_variants(value: str) -> list[str]:
    """
    Generate normalized variants for the company name to handle encoding differences.
    Example: 'Blue Ocean' → ['blue ocean', 'blue%20ocean', 'blue_ocean', 'blueocean']
    """
    if not value:
        return []
    v = value.strip().lower()
    return list({v, v.replace(" ", "%20"), v.replace(" ", "_"), v.replace(" ", "")})


@app.post("/llamaquery")
async def llamaquery(request: Request):
    data = await request.json()
    query = data.get("query")
    if not query:
        return {"error": "Missing 'query' in request body"}

    # Pull the company name if passed in via filters or payload
    filters_payload = data.get("filters") or data.get("preFilters")
    company_value = None
    if filters_payload and "filters" in filters_payload:
        first = filters_payload["filters"][0]
        company_value = first.get("value")
    if not company_value:
        company_value = data.get("company") or ""  # fallback
    if not company_value:
        return {"error": "Missing 'company' name in payload"}

    company_variants = normalize_variants(company_value)

    # Custom HTTP client with extended timeouts
    timeout = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)
    client = httpx.Client(timeout=timeout)

    # Initialize the LlamaCloud index
    index = LlamaCloudIndex(
        name="Sharepoint Deal Pipeline",
        project_name="The BEAST",
        organization_id="8ff953cd-9c16-49f2-93a4-732206133586",
        api_key=os.getenv("LLAMA_API_KEY"),
        client=client,
    )

    # Retry logic for transient network issues
    for attempt in range(3):
        try:
            nodes = await asyncio.to_thread(index.as_retriever().retrieve, query)
            break
        except (httpx.RemoteProtocolError, httpx.ReadTimeout) as e:
            if attempt < 2:
                print(f"Retry {attempt + 1}/3 after error: {e}")
                await asyncio.sleep(2)
                continue
            else:
                return {"error": f"Llama Cloud connection failed: {str(e)}"}

    # Build structured chunk-level results
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

        results.append({
            "text": text,
            "file_name": file_name,
            "web_url": web_url,
        })

    # --- Filter chunks that do NOT match the company name ---
    def match_company(text, file_name, web_url):
        text_lower = (text or "").lower()
        file_lower = (file_name or "").lower()
        url_lower = (web_url or "").lower()
        return any(v in text_lower or v in file_lower or v in url_lower for v in company_variants)

    filtered_results = [
        r for r in results if match_company(r["text"], r["file_name"], r["web_url"])
    ]
    # ---------------------------------------------------------

    # If nothing matches, return notice
    if not filtered_results:
        return {
            "company": company_value,
            "results": [],
            "message": f"No relevant chunks found for '{company_value}'."
        }

    # Return clean structured data
    return {
        "company": company_value,
        "results": filtered_results
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
