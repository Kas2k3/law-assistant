#!/usr/bin/env bash
# Theo dõi tiến trình crawl Pháp Điển vào MySQL (law-crawler/main.py) — chỉ đọc,
# không ảnh hưởng tiến trình crawl. Ctrl-C để thoát.
set -euo pipefail

while true; do
    OUT=$(docker exec law-crawler-law-mysql-1 mysql -uroot -p123456789 -N -s law \
        -e "SELECT COUNT(*), (SELECT COUNT(DISTINCT demuc_id) FROM pddieu) FROM pddieu;" 2>/dev/null || echo "0	0")
    N=$(echo "$OUT" | cut -f1)
    D=$(echo "$OUT" | cut -f2)
    PCT=$(( D * 100 / 306 ))
    FILLED=$(( PCT / 2 ))
    BAR=$(printf "%${FILLED}s" | tr " " "#")
    EMPTY=$(printf "%$((50 - FILLED))s" | tr " " "-")

    clear
    echo "== Crawl Pháp Điển → MySQL =="
    printf "Điều luật đã crawl : %s\n" "$N"
    printf "Đề mục             : %s / 306  (%s%%)\n" "$D" "$PCT"
    printf "[%s%s]\n" "$BAR" "$EMPTY"
    echo
    echo "(Ctrl-C để thoát — không ảnh hưởng crawl)"
    sleep 2
done
