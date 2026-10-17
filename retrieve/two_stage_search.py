from typing import List, Dict, Any
import re
import pandas as pd

from pymilvus import (
    connections,
    Collection,
)
from openai import OpenAI
from sqlalchemy import create_engine

# from hybrid_search import HybridRetriever 


# OpenAI client (dùng cho cả embedding & rerank LLM)
client = OpenAI()

# MySQL engine (dùng cho HybridRetriever nếu cần)
engine = create_engine("mysql+mysqlconnector://root:123456789@localhost:3306/law")

# Milvus
connections.connect("default", host="localhost", port="19530")
print("Connected to Milvus")

collection_name = "phapdien_simple_tendieu"
collection = Collection(collection_name)
collection.load()
print("Loaded collection:", collection_name)


class TwoStageRetriever:
    """
    Chiến lược Hybrid + Re-ranking:

    - B1: Dùng HybridRetriever để lấy top_N ứng viên (hybrid_search).
    - B2: Với từng ứng viên, dùng LLM chấm điểm mức độ liên quan (0-1).
    - B3: Sort lại theo rerank_score, lấy top_k cuối cùng.
    """

    def __init__(
        self,
        # hybrid_retriever: HybridRetriever,
        client: OpenAI,
        rerank_model: str = "gpt-4o-mini",
    ):
        # self.hybrid_retriever = hybrid_retriever
        self.client = client
        self.rerank_model = rerank_model

    def _score_pair(self, query: str, passage: str) -> float:
        """
        Dùng LLM để chấm điểm mức độ liên quan giữa query và passage.
        Trả về số thực từ 0 đến 1. Nếu parse lỗi thì trả 0.0.
        """
        system_prompt = (
            "Bạn là một hệ thống chấm điểm truy hồi văn bản phục vụ cho trả lời truy vấn pháp luật. "
            "Nhiệm vụ của bạn là đánh giá mức độ đoạn văn (passage) trả lời được câu hỏi (query). "
            "Đoạn văn nào trực tiếp cần dùng để trả lời câu hỏi thì điểm cao."
            "Hãy trả về MỘT số thực trong khoảng từ 0 đến 1:\n"
            "- 0: hoàn toàn không liên quan\n"
            "- 1: rất liên quan, trả lời trực tiếp câu hỏi\n"
            "Chỉ in ra đúng MỘT số, không giải thích gì thêm."
        )

        user_content = f"Query:\n{query}\n\nPassage:\n{passage}"

        resp = self.client.chat.completions.create(
            model=self.rerank_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=8,
        )

        text = resp.choices[0].message.content.strip()
        # Tìm số thực trong output
        match = re.search(r"([01](?:\.\d+)?)", text)
        if not match:
            return 0.0
        try:
            score = float(match.group(1))
            # ép vào [0,1] cho chắc
            score = max(0.0, min(1.0, score))
            return score
        except ValueError:
            return 0.0

    def _embed_batch(self, texts):
        resp = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [item.embedding for item in resp.data]

    def _select_noidung(self, offset):
        query = f"""
            SELECT 
                p.*,
                cd.ten  AS ten_chude,
                dm.ten  AS ten_demuc,
                ch.ten  AS ten_chuong
            FROM pddieu p
            LEFT JOIN pdchude  cd ON cd.id = p.chude_id
            LEFT JOIN pddemuc  dm ON dm.id = p.demuc_id
            LEFT JOIN pdchuong ch ON ch.mapc = p.chuong_id
            LIMIT 1 OFFSET {offset - 1};
        """
        df = pd.read_sql(query, engine).fillna("")

        noidung = df["noidung"].iloc[0]
        ten_dieu = df["ten"].iloc[0]
        ten_chude = df["ten_chude"].iloc[0]
        ten_demuc = df["ten_demuc"].iloc[0]
        ten_chuong = df["ten_chuong"].iloc[0]
        vbpl = df["vbqppl"].iloc[0]
        return noidung, ten_dieu, ten_chude, ten_chuong, ten_demuc, vbpl

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        candidate_k: int = 30,
        alpha: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """
        - query: câu hỏi người dùng
        - top_k: số kết quả cuối cùng trả về sau khi rerank
        - candidate_k: số ứng viên lấy từ hybrid_search trước khi rerank
        - alpha: tham số hybrid bên dưới (tuỳ HybridRetriever của bạn dùng)

        Trả về list dict, mỗi dict là 1 chunk (Điều) đã được gắn thêm key 'rerank_score'.
        """

        query_vec = self._embed_batch([query])[0]

        results = collection.search(
            data=[query_vec],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=candidate_k,
            output_fields=["metadata"]
        )

        if not results:
            return []

        candidates = []
        # B2: chấm điểm rerank từng ứng viên
        for hit in results[0]:
            # passage = cand.get("noidung", "") or ""
            offset = hit.id
            noidung, ten_dieu, ten_chude, ten_chuong, ten_demuc, vbpl = self._select_noidung(offset)
            
            passage = (
                    f"Chủ đề: {ten_chude}\n"
                    f"Đề mục: {ten_demuc}\n"
                    f"Chương: {ten_chuong}\n"
                    f"Điều: {ten_dieu}\n"
                    f"{noidung}\n"
                    f"Căn cứ theo {vbpl}"
                )

            rerank_score = self._score_pair(query, passage)

            candidates.append({
                "rerank_score": rerank_score,
                "offset": offset,
                "score": float(hit.distance),
                "ten": ten_dieu,
                "demuc": ten_demuc,
                "chuong": ten_chuong,
                "chude": ten_chude,
                "vbqppl": vbpl,
                "noidung": noidung,
                "passage": passage
            })

        # B3: sort theo rerank_score giảm dần
        candidates = sorted(candidates, key=lambda x: x.get("rerank_score", 0.0), reverse=True)

        # B4: cắt top_k
        return candidates[:top_k]
    

import csv
import os
import json

def export_results_to_csv_grouped(all_rows, filename):
    """
    Lưu CSV gồm các cột:
    - cau_hoi
    - id
    - noidung
    Trong đó mỗi câu hỏi có 20 bản ghi (20 dòng).
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cau_hoi", "id", "noidung"])

        for row in all_rows:
            writer.writerow([row["cau_hoi"], row["id"], row["noidung"]])

    print(f"Đã ghi file: {filename} | Tổng dòng: {len(all_rows)}")

def retrieve_query(query):
    rerank_retriever = TwoStageRetriever(
        # hybrid_retriever=hybrid_retriever,
        client=client,
        rerank_model="gpt-4o-mini",  # đổi model nếu muốn
    )

    results = rerank_retriever.retrieve(
        query=query,
        top_k=20,
        candidate_k=30,
        alpha=0.6,
    )

    return results

def run_test():
    with open("cauhoi.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for chu_de, questions in data.items():
        print(f"\n=== Xử lý chủ đề: {chu_de} ===")

        all_rows = []

        for question in questions:
            print(f" ->.  Đang xử lý câu hỏi: {question}")

            results = retrieve_query(question)

            for r in results:
                offset = r.get("offset", None)

                block = (
                    f"Chủ đề: {r.get('chude', '')}\n"
                    f"Đề mục: {r.get('demuc', '')}\n"
                    f"Chương: {r.get('chuong', '')}\n"
                    f"{r.get('ten', '')}\n"
                    f"{r.get('noidung', '')}\n"
                    f"Căn cứ theo {r.get('vbqppl', '')}"
                )

                all_rows.append({
                    "cau_hoi": question,
                    "id": offset,
                    "noidung": block
                })

        safe_filename = chu_de.replace("/", "-").replace("\\", "-")
        output_path = f"ketqua/{safe_filename}.csv"

        export_results_to_csv_grouped(all_rows, output_path)

if __name__ == "__main__":
    run_test()