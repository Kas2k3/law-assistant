#!/usr/bin/env bash
# Quản lý toàn bộ hệ thống chatbot RAG pháp luật: hạ tầng Docker (MySQL,
# Milvus, Neo4j), backend FastAPI, frontend Next.js, dashboard đánh giá.
#
# Dùng:
#   ./run.sh start            # dựng docker infra + chạy backend + frontend
#   ./run.sh stop              # dừng backend/frontend/dashboard (giữ docker)
#   ./run.sh restart           # stop rồi start lại (giữ docker)
#   ./run.sh down               # stop toàn bộ + docker compose down 2 stack
#   ./run.sh status             # xem tiến trình nào đang chạy
#   ./run.sh logs [service]     # tail log (không truyền service = tail tất cả)
#   ./run.sh dashboard          # chạy dashboard đánh giá (evaluation/visualization.py)
#   ./run.sh open               # mở trình duyệt tới UI + dashboard đang chạy

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT/logs"
PID_DIR="$ROOT/.run"
mkdir -p "$LOG_DIR" "$PID_DIR"

FRONTEND_URL="http://localhost:3000"
BACKEND_URL="http://127.0.0.1:8001"
DASHBOARD_URL="http://localhost:8501"

# ───────────────────────── helpers ─────────────────────────

pid_file() { echo "$PID_DIR/$1.pid"; }
log_file() { echo "$LOG_DIR/$1.log"; }

is_running() {
    local name="$1" pf
    pf="$(pid_file "$name")"
    [[ -f "$pf" ]] || return 1
    local pid
    pid="$(cat "$pf")"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

start_proc() {
    # start_proc <name> <workdir> <command...>
    local name="$1" workdir="$2"
    shift 2
    if is_running "$name"; then
        echo "  [$name] đã chạy (pid $(cat "$(pid_file "$name")")), bỏ qua."
        return 0
    fi
    (
        cd "$workdir"
        # setsid để toàn bộ process group con (vd next dev spawn thêm tiến
        # trình) bị kill gọn khi stop, không để sót tiến trình zombie.
        setsid nohup "$@" >"$(log_file "$name")" 2>&1 &
        echo $! > "$(pid_file "$name")"
    )
    sleep 1
    if is_running "$name"; then
        echo "  [$name] started (pid $(cat "$(pid_file "$name")")), log: $(log_file "$name")"
    else
        echo "  [$name] KHỞI ĐỘNG LỖI — xem $(log_file "$name")"
    fi
}

stop_proc() {
    local name="$1"
    if ! is_running "$name"; then
        rm -f "$(pid_file "$name")"
        echo "  [$name] không chạy."
        return 0
    fi
    local pid
    pid="$(cat "$(pid_file "$name")")"
    echo "  [$name] stopping (pid $pid)…"
    kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 10); do
        is_running "$name" || break
        sleep 0.5
    done
    if is_running "$name"; then
        echo "  [$name] vẫn sống, kill -9…"
        kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    fi
    rm -f "$(pid_file "$name")"
    echo "  [$name] stopped."
}

open_url() {
    local url="$1"
    if command -v explorer.exe >/dev/null 2>&1; then
        explorer.exe "$url" >/dev/null 2>&1 || true
    elif command -v wslview >/dev/null 2>&1; then
        wslview "$url" >/dev/null 2>&1 || true
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$url" >/dev/null 2>&1 || true
    else
        echo "  (không tự mở được trình duyệt) → $url"
        return 0
    fi
    echo "  → mở $url"
}

wait_http_ok() {
    # wait_http_ok <url> <timeout_s> <match> — chờ HTTP 200 (hoặc body chứa $match)
    local url="$1" timeout="${2:-60}" match="${3:-}" waited=0
    while (( waited < timeout )); do
        local body
        body="$(curl -fsS --max-time 2 "$url" 2>/dev/null || true)"
        if [[ -n "$body" ]]; then
            if [[ -z "$match" || "$body" == *"$match"* ]]; then
                return 0
            fi
        fi
        sleep 2
        waited=$((waited + 2))
    done
    return 1
}

venv_python_ready() {
    [[ -x "$ROOT/.venv/bin/python" ]]
}

# ───────────────────────── docker infra ─────────────────────────

docker_up() {
    echo "[docker] law-crawler (MySQL + phpMyAdmin)…"
    (cd "$ROOT/law-crawler" && docker compose up -d)
    echo "[docker] infra (Milvus + Neo4j)…"
    (cd "$ROOT/infra" && docker compose up -d)

    echo -n "[docker] chờ Milvus healthz… "
    if wait_http_ok "http://localhost:9091/healthz" 120 "OK"; then
        echo "OK"
    else
        echo "TIMEOUT — kiểm tra: cd infra && docker compose logs milvus"
    fi
}

docker_down() {
    echo "[docker] infra down…"
    (cd "$ROOT/infra" && docker compose down) || true
    echo "[docker] law-crawler down…"
    (cd "$ROOT/law-crawler" && docker compose down) || true
}

docker_status() {
    echo "[docker] law-crawler:"
    (cd "$ROOT/law-crawler" && docker compose ps) || true
    echo "[docker] infra:"
    (cd "$ROOT/infra" && docker compose ps) || true
}

# ───────────────────────── app services ─────────────────────────

start_backend() {
    if ! venv_python_ready; then
        echo "  [backend] .venv chưa có — chạy: python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
        return 1
    fi
    start_proc backend "$ROOT" \
        "$ROOT/.venv/bin/python" -m uvicorn backend.rag_api:app --host 0.0.0.0 --port 8001
}

start_frontend() {
    if [[ ! -d "$ROOT/frontend/node_modules" ]]; then
        echo "  [frontend] node_modules chưa có — chạy: cd frontend && npm install"
        return 1
    fi
    start_proc frontend "$ROOT/frontend" npm run dev
}

start_dashboard() {
    if ! venv_python_ready; then
        echo "  [dashboard] .venv chưa có — xem './run.sh start' để biết cách cài."
        return 1
    fi
    start_proc dashboard "$ROOT/evaluation" \
        "$ROOT/.venv/bin/python" -m streamlit run visualization.py \
        --server.headless true --server.port 8501
}

# ───────────────────────── commands ─────────────────────────

cmd_start() {
    docker_up
    echo "[backend] khởi động…"
    start_backend || true
    echo "[frontend] khởi động…"
    start_frontend || true

    echo -n "[backend] chờ /health… "
    if wait_http_ok "$BACKEND_URL/health" 60; then
        echo "OK"
    else
        echo "chưa sẵn sàng (xem logs/backend.log)"
    fi

    echo
    echo "Sẵn sàng:"
    echo "  Frontend : $FRONTEND_URL"
    echo "  Backend  : $BACKEND_URL (docs: $BACKEND_URL/docs)"
    echo "  phpMyAdmin: http://localhost:8081"
    echo "  Neo4j    : http://localhost:7474"
    open_url "$FRONTEND_URL"
}

cmd_stop() {
    echo "[app] dừng backend/frontend/dashboard (docker vẫn giữ nguyên)…"
    stop_proc backend
    stop_proc frontend
    stop_proc dashboard
}

cmd_restart() {
    cmd_stop
    cmd_start
}

cmd_down() {
    cmd_stop
    docker_down
}

cmd_status() {
    echo "== Tiến trình app =="
    for name in backend frontend dashboard; do
        if is_running "$name"; then
            echo "  [$name] RUNNING (pid $(cat "$(pid_file "$name")"))"
        else
            echo "  [$name] stopped"
        fi
    done
    echo
    echo "== Docker =="
    docker_status
}

cmd_logs() {
    local svc="${1:-}"
    if [[ -n "$svc" ]]; then
        local lf
        lf="$(log_file "$svc")"
        [[ -f "$lf" ]] || { echo "Không thấy log: $lf (service chưa chạy lần nào?)"; exit 1; }
        tail -n 100 -f "$lf"
    else
        local files=()
        for name in backend frontend dashboard; do
            [[ -f "$(log_file "$name")" ]] && files+=("$(log_file "$name")")
        done
        if [[ ${#files[@]} -eq 0 ]]; then
            echo "Chưa có log nào — chạy './run.sh start' trước."
            exit 1
        fi
        # tail -f nhiều file: tự in header "==> file <==" khi có dòng mới
        tail -n 20 -f "${files[@]}"
    fi
}

cmd_dashboard() {
    start_dashboard
    echo -n "[dashboard] chờ Streamlit… "
    if wait_http_ok "$DASHBOARD_URL" 30; then
        echo "OK"
    else
        echo "chưa sẵn sàng (xem logs/dashboard.log)"
    fi
    open_url "$DASHBOARD_URL"
}

cmd_open() {
    is_running frontend && open_url "$FRONTEND_URL"
    is_running dashboard && open_url "$DASHBOARD_URL"
}

usage() {
    cat <<'EOF'
Dùng: ./run.sh <command>

  start       Dựng docker (MySQL/Milvus/Neo4j) + chạy backend + frontend
  stop        Dừng backend/frontend/dashboard (docker vẫn chạy)
  restart     stop rồi start (docker vẫn chạy, không down/up lại)
  down        stop toàn bộ + docker compose down (law-crawler + infra)
  status      Trạng thái tiến trình app + docker compose ps
  logs [name] Tail log (name: backend|frontend|dashboard; bỏ trống = tất cả)
  dashboard   Chạy dashboard đánh giá (evaluation/visualization.py, port 8501)
  open        Mở trình duyệt tới các URL đang chạy
EOF
}

main() {
    local cmd="${1:-}"
    shift || true
    case "$cmd" in
        start) cmd_start ;;
        stop) cmd_stop ;;
        restart) cmd_restart ;;
        down) cmd_down ;;
        status) cmd_status ;;
        logs) cmd_logs "${1:-}" ;;
        dashboard) cmd_dashboard ;;
        open) cmd_open ;;
        *) usage; exit 1 ;;
    esac
}

main "$@"
