#!/usr/bin/env bash
# Theo dõi tiến trình seed Milvus bằng BGE-M3 (data_processing/seed_data_batch_bge.py)
# — chỉ đọc, không ảnh hưởng tiến trình. Ctrl-C để thoát.
set -euo pipefail

CKPT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data_processing/.seed_checkpoint_bge.json"
TOTAL=66362

while true; do
    if [ -f "$CKPT" ]; then
        OFFSET=$(grep -o '"doc_offset": [0-9]*' "$CKPT" | grep -o '[0-9]*')
        NEXT_ID=$(grep -o '"next_id": [0-9]*' "$CKPT" | grep -o '[0-9]*')
    else
        OFFSET=0
        NEXT_ID=1
    fi
    PCT=$(( OFFSET * 100 / TOTAL ))
    FILLED=$(( PCT / 2 ))
    BAR=$(printf "%${FILLED}s" | tr " " "#")
    EMPTY=$(printf "%$((50 - FILLED))s" | tr " " "-")

    clear
    echo "== Seed Milvus bằng BGE-M3 (phapdien_bge_m3) =="
    printf "Điều đã xử lý : %s / %s  (%s%%)\n" "$OFFSET" "$TOTAL" "$PCT"
    printf "Vector đã tạo : %s\n" "$((NEXT_ID - 1))"
    printf "[%s%s]\n" "$BAR" "$EMPTY"
    echo
    echo "(Ctrl-C để thoát — không ảnh hưởng tiến trình seed)"
    sleep 3
done
