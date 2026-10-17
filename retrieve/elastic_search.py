from elasticsearch import Elasticsearch
import os
from dotenv import load_dotenv

load_dotenv()

ES_ENDPOINT = os.getenv("ES_ENDPOINT")
ES_API_KEY  = os.getenv("ES_API_KEY")

FIELDS = ["tenchude", "tendemuc", "tenchuong", "tendieu", "noidung"]

def retrieve_top_20_results(index_name: str, query: str):
    es = Elasticsearch(ES_ENDPOINT, api_key=ES_API_KEY)

    if not es.ping():
        raise RuntimeError("❌ Failed to connect to Elasticsearch.")

    search_body = {
        "_source": FIELDS,  # chỉ lấy đúng các field cần in
        "query": {
            "multi_match": {
                "query": query,
                "fields": [
                    "tenchude^1",
                    "tendemuc^1.5",
                    "tenchuong^2",
                    "tendieu^10",
                    "noidung^9"
                ]
            }
        }
    }

    response = es.search(index=index_name, body=search_body, size=20)
    hits = response.get("hits", {}).get("hits", [])

    results = []
    for hit in hits:
        src = hit.get("_source", {}) or {}

        nd = src.get("noidung", "")
        if isinstance(nd, list):
            noidung_text = " ".join(map(str, nd))
        else:
            noidung_text = str(nd)

        results.append({
            "_id": hit.get("_id"),
            "_score": hit.get("_score", 0.0),
            "tenchude": src.get("tenchude", ""),
            "tendemuc": src.get("tendemuc", ""),
            "tenchuong": src.get("tenchuong", ""),
            "tendieu": src.get("tendieu", ""),
            "noidung": noidung_text,
        })

    return results

if __name__ == "__main__":
    index_name = "law_data"
    query = "Hành vi chống đối người thi hành công vụ sẽ bị xử phạt như thế nào?"
    results = retrieve_top_20_results(index_name, query)

    for i, r in enumerate(results, start=1):
        print(f"\n{i}: ==========================================")
        print(f"_id: {r.get('_id', '')} | _score: {r.get('_score', 0.0):.4f}")
        print(f"Chủ đề : {r.get('tenchude', '')}")
        print(f"Đề mục : {r.get('tendemuc', '')}")
        print(f"Chương : {r.get('tenchuong', '')}")
        print(f"Điều   : {r.get('tendieu', '')}")
        print("- Nội dung:")
        print(r.get("noidung", ""))
