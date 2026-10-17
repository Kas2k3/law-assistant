# ingest.py (phần ensure_index)
import json, os
from elasticsearch import helpers
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

load_dotenv()

ES_ENDPOINT = os.getenv("ES_ENDPOINT")
ES_API_KEY  = os.getenv("ES_API_KEY")

def ensure_index(es, index_name: str):
    # nếu index đã tồn tại thì không làm gì (tránh ghi đè analyzer không thể update trực tiếp)
    if es.indices.exists(index=index_name):
        return

    # settings (giữ nguyên từ notebook)
    settings = {
        "settings": {
            "analysis": {
                "analyzer": {
                    "legal_vi_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": [
                            "lowercase",
                            "custom_stopwords_filter",
                            "synonym_filter",
                            "remove_punctuation"
                        ]
                    }
                },
                "filter": {
                    "custom_stopwords_filter": {
                        "type": "stop",
                        "stopwords": [
                            "chứ", "đó", "này", "là", "vậy", "cứ", "gì"
                        ]
                    },
                    "synonym_filter": {
                        "type": "synonym",
                        "synonyms": [
                            "luật, pháp luật, quy định, điều lệ, quy tắc, luật pháp",
                            "hợp pháp, hợp lệ, đúng luật, chính đáng, chính thống, đúng quy định",
                            "trái phép, phạm pháp, phi pháp, trái luật",
                            "pháp lý, tư pháp, pháp chế, pháp quyền, pháp định",
                            "điều luật, khoản luật, mục luật, điều khoản",
                            "thẩm quyền, quyền hạn, quyền lực",
                            "vi phạm, xâm phạm, trái luật",
                            "hình phạt, xử phạt, án phạt, chế tài",
                            "tội phạm, tội đồ, kẻ vi phạm, kẻ phạm pháp",
                            "nghĩa vụ, bổn phận, phận sự",
                            "bằng chứng, chứng cứ, vật chứng",
                            "phán quyết, quyết định, bản án",
                            "tòa án, pháp đình, phiên tòa",
                            "luật sư, luật gia",
                            "tranh chấp, xung đột, mâu thuẫn",
                            "hòa giải, thương lượng, đàm phán, giải hòa",
                            "kháng cáo, khiếu nại, kháng nghị",
                            "quyền lợi, lợi ích, phúc lợi",
                            "hợp đồng, giao kèo, cam kết",
                            "chứng thực, công chứng, xác minh",
                            "truy tố, buộc tội, cáo buộc",
                            "công lý, lẽ phải, chính nghĩa",
                            "thủ tục, quy trình, cách thức",
                            "quyền sở hữu, quyền sử dụng",
                            "thời hiệu, thời hạn, kỳ hạn",
                            "bắt giữ, giam giữ, giam cầm",
                            "tự do, độc lập, tự trị",
                            "xét xử, phán xét, xử án",
                            "hình thức, biện pháp, phương án",
                            "án lệ, tiền lệ",
                            "khởi kiện, tố cáo, kiện tụng",
                            "thi hành, áp dụng, chấp hành",
                            "phạm nhân, tù nhân, bị cáo",
                            "lệnh bắt, lệnh giam",
                            "tạm giam, giam lỏng",
                            "giám định, kiểm định, thẩm định",
                            "chế tài, xử phạt",
                            "quy định, điều khoản, nội quy",
                            "hành vi, hành động, cử chỉ",
                            "truy cứu, điều tra",
                            "biên bản, báo cáo, hồ sơ",
                            "khiếu nại, tố cáo",
                            "chứng từ, tài liệu",
                            "bảo vệ, giữ gìn"
                        ]
                    },
                    "remove_punctuation": {
                        "type": "pattern_replace",
                        "pattern": "[\\p{Punct}]",
                        "replacement": ""
                    }
                }
            }
        }
    }

    # mappings (giữ dynamic_templates như gốc)
    mappings = {
        "mappings": {
            "dynamic_templates": [
                {
                    "texts_with_legal_vi_analyzer": {
                        "match_mapping_type": "string",
                        "mapping": {
                            "type": "text",
                            "analyzer": "legal_vi_analyzer"
                        }
                    }
                }
            ]
        }
    }

    # tạo index với settings + mappings
    es.indices.create(index=index_name, body={**settings, **mappings})
    # print(f"Index {index_name} has been created with settings and mappings.")

def ingest_data(es, index_name: str = "law_data", file_path: str = "data.json"):
    """
    Nạp dữ liệu từ file JSON vào Elasticsearch theo batch.
    """
    # đọc dữ liệu
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Number of records: {len(data)}")

    # chuẩn bị actions cho bulk
    actions = [{"_index": index_name, "_source": doc} for doc in data]

    try:
        success, failed = helpers.bulk(es, actions, chunk_size=500)
        print(f"✅ Đã nạp thành công {success} tài liệu vào Elasticsearch!")
    except Exception as e:
        print(f"❌ Lỗi khi nạp dữ liệu: {e}")

if __name__ == "__main__":
    # Kết nối bằng endpoint (không cần cloud_id)
    es = Elasticsearch(ES_ENDPOINT, api_key=ES_API_KEY)

    INDEX_NAME = "law_data"
    FILE_PATH = "data.json"

    print("🚀 Bắt đầu ingest dữ liệu vào Elasticsearch...")
    ensure_index(es, INDEX_NAME)
    ingest_data(es, INDEX_NAME, FILE_PATH)
    print("🎉 Hoàn tất ingest!")