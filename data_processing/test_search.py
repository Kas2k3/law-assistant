from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection, utility
)
from openai import OpenAI
import pandas as pd
import json

# from services.utils import embed_batch

# Connect OpenAI
client = OpenAI()

# Connect Milvus
connections.connect("default", host="localhost", port="19530")
print("Connected to Milvus")

collection_name = "phapdien_simple_tendieu"

# Chỉ cần load collection đã tạo từ trước
collection = Collection(collection_name)
collection.load()

print("Loaded collection:", collection_name)

path= "data.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

def embed_batch(texts):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in resp.data]

class TestSearch:
    def __init__(self, collection: Collection, client: OpenAI):
        self.collection = collection
        self.client = client

    def select_noidung(self, offset):
        rec = data[offset]

        ten_chude = (rec.get("tenchude") or "").strip()
        ten_demuc = (rec.get("tendemuc") or "").strip()
        ten_chuong = (rec.get("tenchuong") or "").strip()
        ten_dieu = (rec.get("tendieu") or "").strip()

        nd = rec.get("noidung") or []
        if isinstance(nd, list):
            noidung = "\n".join(nd)
        else:
            noidung = str(nd)

        noidung = noidung.strip()

        return noidung, ten_dieu, ten_chude, ten_chuong, ten_demuc

    def search(self, query: str, top_k: int = 5):
        """Thực hiện semantic search với query truyền từ ngoài vào."""
        query_vec = embed_batch([query])[0]

        results = self.collection.search(
            data=[query_vec],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            output_fields=["metadata"]
        )

        items = []
        for hit in results[0]:
            offset = hit.id
            noidung, ten_dieu, ten_chude, ten_chuong, ten_demuc = self.select_noidung(offset)

            items.append({
                "offset": offset,
                "score": float(hit.distance),
                "ten": ten_dieu,
                "demuc": ten_demuc,
                "chuong": ten_chuong,
                "chude": ten_chude,
                "noidung": noidung,
            })

        return items


if __name__ == "__main__":
    retriever = TestSearch(collection=collection, client=client)

    # Bạn có thể đổi query ở đây, hoặc sau này truyền từ CLI / API
    query = "Hành vi chống đối người thi hành công vụ sẽ bị xử phạt như thế nào?"
    results = retriever.search(query=query, top_k=20)


    print("\n=== Search Results ===")
    for hit in results:
        print(f"ID={hit['offset']}, score={hit['score']}")
        print("   chude:", hit["chude"])
        print("   chuong:", hit['chuong'])
        print("   demuc :", hit['demuc'])
        print("   ten      :", hit['ten'])
        print("   noidung  :", hit['noidung'], "...")
        print("---------------------")
