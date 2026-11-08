"""
Xuất toàn bộ dữ liệu đang dùng để chunking (từ MySQL, cùng nguồn + cùng hàm
chunking với seed_data_batch.py) ra 1 file JSON, đồng thời đánh giá chất lượng
recursive chunking: tìm các trường hợp cắt giữa từ (mid-word), chunk rỗng/quá
nhỏ, và các "điều" bất thường (noidung quá dài) có nguy cơ làm sai lệch ngữ
nghĩa vì header (tên điều) được gắn vào MỌI chunk dù nội dung thực tế của chunk
đó có thể không còn liên quan trực tiếp tới tên điều.

Chạy: python data_processing/export_and_evaluate_chunks.py
Output:
  evaluation/results/chunk_dataset_export.json   (toàn bộ chunk hiện dùng để seed)
  evaluation/results/chunk_quality_report.json   (báo cáo đánh giá)
"""
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(__file__))
from seed_data_batch import (  # noqa: E402
    load_data_from_mysql,
    build_chunk_records,
    _noidung_text,
    _ENCODER,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "evaluation", "results")
os.makedirs(OUT_DIR, exist_ok=True)
EXPORT_PATH = os.path.join(OUT_DIR, "chunk_dataset_export.json")
REPORT_PATH = os.path.join(OUT_DIR, "chunk_quality_report.json")

# Ngưỡng coi là "điều" bất thường (noidung quá dài -> nghi ngờ lỗi crawl / bảng
# phụ lục bị gộp vào 1 điều) — số chunk sinh ra quá nhiều khiến header (tên
# điều) không còn mô tả đúng nội dung từng chunk.
SUSPICIOUS_CHUNK_COUNT = 50
TINY_CHUNK_TOKENS = 15
WORD_CHAR_RE = re.compile(r"\w", re.UNICODE)


def is_mid_word_cut(original: str, chunk: str, search_from: int) -> tuple[bool, int]:
    """Tìm vị trí `chunk` trong `original` (tìm từ search_from trở đi). Trả về
    (bị_cắt_giữa_từ, vị_trí_tìm_thấy). Nếu không tìm thấy nguyên văn, trả về
    (False, -1) — không kết luận được (có thể do overlap/ghép khác biệt nhỏ)."""
    pos = original.find(chunk, max(0, search_from - len(chunk)))
    if pos == -1:
        return False, -1
    if pos == 0:
        return False, pos
    prev_char = original[pos - 1]
    first_char = chunk[0] if chunk else ""
    mid_word = bool(WORD_CHAR_RE.match(prev_char) and WORD_CHAR_RE.match(first_char))
    return mid_word, pos


def main():
    print("Đang tải dữ liệu từ MySQL...")
    data = load_data_from_mysql()
    total_articles = len(data)
    print(f"Tổng số điều: {total_articles}")

    export_records = []
    token_lengths = []
    chunks_per_article = []
    suspicious_articles = []
    mid_word_cases = []
    tiny_chunk_cases = []
    empty_chunk_count = 0

    for art in data:
        chunk_recs = build_chunk_records(art)
        n_chunks = len(chunk_recs)
        chunks_per_article.append(n_chunks)

        original_full = _noidung_text(art)
        search_cursor = 0

        if n_chunks > SUSPICIOUS_CHUNK_COUNT:
            suspicious_articles.append({
                "mapc": art.get("mapc"),
                "tendieu": art.get("tendieu"),
                "noidung_chars": len(original_full),
                "so_chunk": n_chunks,
            })

        for rec in chunk_recs:
            body = (rec.get("noidung_chunk") or "").strip()
            n_tok = len(_ENCODER.encode(body)) if body else 0
            token_lengths.append(n_tok)

            if not body:
                empty_chunk_count += 1
            elif n_tok < TINY_CHUNK_TOKENS and rec["chunk_index"] != rec["chunk_total"] - 1:
                tiny_chunk_cases.append({
                    "mapc": art.get("mapc"),
                    "tendieu": art.get("tendieu"),
                    "chunk_index": rec["chunk_index"],
                    "chunk_total": rec["chunk_total"],
                    "n_token": n_tok,
                    "text": body[:120],
                })

            mid_word, pos = is_mid_word_cut(original_full, body, search_cursor)
            if pos != -1:
                search_cursor = pos + len(body)
            if mid_word and len(mid_word_cases) < 50:
                mid_word_cases.append({
                    "mapc": art.get("mapc"),
                    "tendieu": art.get("tendieu"),
                    "chunk_index": rec["chunk_index"],
                    "boundary_context": original_full[max(0, pos - 30): pos + 30],
                })

            export_records.append({
                "mapc": rec.get("mapc"),
                "tenchude": rec.get("tenchude"),
                "tendemuc": rec.get("tendemuc"),
                "tenchuong": rec.get("tenchuong"),
                "tendieu": rec.get("tendieu"),
                "chunk_index": rec.get("chunk_index"),
                "chunk_total": rec.get("chunk_total"),
                "n_token": n_tok,
                "noidung_chunk": body,
            })

    print(f"Tổng số chunk sinh ra: {len(export_records)}")
    print("Đang ghi file export...")
    with open(EXPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(export_records, f, ensure_ascii=False)

    def pct(data_list, p):
        if not data_list:
            return 0
        s = sorted(data_list)
        idx = min(len(s) - 1, int(len(s) * p / 100))
        return s[idx]

    suspicious_total_chunks = sum(a["so_chunk"] for a in suspicious_articles)
    report = {
        "tong_so_dieu": total_articles,
        "tong_so_chunk": len(export_records),
        "chunk_token_length": {
            "min": min(token_lengths) if token_lengths else 0,
            "max": max(token_lengths) if token_lengths else 0,
            "mean": round(statistics.mean(token_lengths), 1) if token_lengths else 0,
            "median": statistics.median(token_lengths) if token_lengths else 0,
            "p95": pct(token_lengths, 95),
            "p99": pct(token_lengths, 99),
        },
        "so_chunk_tren_moi_dieu": {
            "min": min(chunks_per_article) if chunks_per_article else 0,
            "max": max(chunks_per_article) if chunks_per_article else 0,
            "mean": round(statistics.mean(chunks_per_article), 2) if chunks_per_article else 0,
            "median": statistics.median(chunks_per_article) if chunks_per_article else 0,
        },
        "canh_bao": {
            "chunk_rong": empty_chunk_count,
            "chunk_qua_nho_bat_thuong": {
                "nguong_token": TINY_CHUNK_TOKENS,
                "so_luong": len(tiny_chunk_cases),
                "vi_du": tiny_chunk_cases[:10],
            },
            "cat_giua_tu_mid_word": {
                "so_luong_phat_hien": len(mid_word_cases),
                "ghi_chu": "chỉ quét 50 trường hợp đầu tiên tìm thấy để lấy mẫu, không phải tổng số thực",
                "vi_du": mid_word_cases[:10],
            },
            "dieu_bat_thuong_qua_nhieu_chunk": {
                "nguong_so_chunk": SUSPICIOUS_CHUNK_COUNT,
                "so_dieu_bi_canh_bao": len(suspicious_articles),
                "tong_chunk_tu_cac_dieu_nay": suspicious_total_chunks,
                "ty_le_tren_tong_chunk_pct": round(100 * suspicious_total_chunks / max(1, len(export_records)), 2),
                "danh_sach": sorted(suspicious_articles, key=lambda a: -a["so_chunk"])[:30],
            },
        },
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("✅ Xong.")
    print(f"  Export: {EXPORT_PATH}")
    print(f"  Report: {REPORT_PATH}")
    print(json.dumps(report["canh_bao"]["dieu_bat_thuong_qua_nhieu_chunk"], ensure_ascii=False, indent=2)[:1500])


if __name__ == "__main__":
    main()
