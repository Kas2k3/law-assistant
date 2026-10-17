from elasticsearch import Elasticsearch
# from retrieve.hybrid_search import retrieve_top_20_results

# Kết nối bằng endpoint (không cần cloud_id)
# endpoint = "https://my-elasticsearch-project-b6206b.es.asia-south1.gcp.elastic.cloud:443"
# api_key = "V21GYUI1b0JiRTFmNGdnYXg3VHg6bEFMeFJvNHFjVHJEdkhvSjRBUlVoQQ=="

ES_ENDPOINT="https://my-elasticsearch-project-bacbca.es.asia-south1.gcp.elastic.cloud:443"
ES_API_KEY="TlpFYWxab0JOYmlQSGlkQ0d2UWc6OU9wN2dQRVlYVlVzWVh6d2NKZThXQQ=="

es = Elasticsearch(ES_ENDPOINT, api_key=ES_API_KEY)

def get_es_client() -> Elasticsearch:
    """Tạo client Elasticsearch dùng endpoint + API key."""
    if not ES_ENDPOINT or not ES_API_KEY:
        raise ValueError("Chưa set ES_ENDPOINT hoặc ES_API_KEY trong .env")

    es = Elasticsearch(ES_ENDPOINT, api_key=ES_API_KEY)

    if not es.ping():
        raise ConnectionError("❌ Failed to connect to Elasticsearch (ping thất bại).")

    return es


def retrieve_top_20_results(index_name, query):
    es = Elasticsearch(ES_ENDPOINT, api_key=ES_API_KEY)

    if not es.ping():
        print("❌ Failed to connect to Elasticsearch.")

    search_body = {
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

    noidung_list = []
    for hit in hits:
        nd = hit.get("_source", {}).get("noidung", "")
        if isinstance(nd, list):
            text = " ".join(nd)
        else:
            text = str(nd)
        noidung_list.append(text)
    return noidung_list


def main():
    index_name = "law_data"   # TODO: sửa lại tên index thực tế
    query = "Thủ tục và giấy tờ đăng ký kết hôn tại Việt Nam"        # TODO: sửa query test

    print(f"🔎 Đang search trên index='{index_name}' với query='{query}'")

    try:
        results = retrieve_top_20_results(index_name, query)
        print(f"✅ Lấy được {len(results)} kết quả.")
        for i, txt in enumerate(results[:5], start=1):
            print(f"\n----- Kết quả {i} -----")
            print(txt[:500], "...")
    except ConnectionError:
        print("⏰ Lỗi ConnectionTimeout: Elasticsearch mất quá lâu để trả kết quả.")
        print("Hãy thử:")
        print("- Kiểm tra ES có bị quá tải không (CPU/RAM).")
        print("- Giảm độ phức tạp query hoặc thử với index nhỏ hơn.")
        print("- Tiếp tục tăng request_timeout (ví dụ 120).")
    except Exception as e:
        print("❌ Có lỗi xảy ra:", repr(e))


if __name__ == "__main__":
    main()