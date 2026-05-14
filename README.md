# AI Server

FastAPI server chạy AI backend kết hợp với `app.ai_core` bên trong `medical_voice_rag_app/ai_core`.

---

## Cấu trúc thư mục

```
ai_server/
├── app/
│   ├── main.py                   # FastAPI entry point, lifespan, mounts routers
│   ├── api/
│   │   ├── router_ai.py          # Các endpoint AI (STT, TTS, RAG, predict)
│   │   ├── deps.py               # Dependency injection (auth, db)
│   │   ├── router_chat.py        # (Old) POST /api/v1/chat  (REST + SSE stream)
│   │   ├── router_websocket.py   # (Old) WS  /ws/chat
│   │   └── router_session.py     # (Old) CRUD /api/v1/sessions
│   ├── core/
│   │   ├── config.py             # Cấu hình từ .env
│   │   └── security.py           # API key auth + rate limiter
│   ├── db/
│   │   ├── database.py           # SQLAlchemy engine + get_db
│   │   └── models.py             # ORM: APIClient, User, Session, Message
│   ├── schemas/
│   │   └── schemas.py            # Pydantic request/response models
│   └── services/
│       └── session_manager.py    # Session/user CRUD
├── tests/
│   └── test_api.py               # Integration tests
├── .env.example
└── requirements.txt
```

---

## Cài đặt

```bash
# Tạo môi trường ảo và cài dependencies
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Copy file .env và cấu hình (đặc biệt là DATABASE_URL, API_KEYS và các MODEL_PATH)
cp .env.example .env

# Tạo user MySQL trước khi chạy server.
# App sẽ tự tạo database healthcare_db nếu user này có quyền CREATE.
mysql -u root -p <<'SQL'
CREATE USER IF NOT EXISTS 'healthcare_user'@'localhost' IDENTIFIED BY 'healthcare_password';
GRANT ALL PRIVILEGES ON healthcare_db.* TO 'healthcare_user'@'localhost';
GRANT CREATE ON *.* TO 'healthcare_user'@'localhost';
FLUSH PRIVILEGES;
SQL

# Chạy server
# Tại thư mục gốc của project (medical_voice_rag_app/ai_core)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Ví dụ cấu hình MySQL trong `.env`:

```env
DATABASE_URL=mysql+pymysql://healthcare_user:healthcare_password@localhost:3306/healthcare_db?charset=utf8mb4
```

---

## API Endpoints Mới (AI Module)

Các API này gọi trực tiếp vào `app.ai_core` core logic.

### Health
```
GET /health
```
Kiểm tra trạng thái server, cấu hình mô hình LLM, Neo4j, TTS, STT.

### Predict / Chat
```
POST /api/predict
POST /api/rag/query
```
Body:
```json
{
  "message": "Đái tháo đường type 2 có triệu chứng gì?",
  "session_id": "default",
  "enable_tts": false
}
```

### STT (Speech To Text)
```
POST /api/stt
```
Dạng `multipart/form-data`. Upload file audio qua trường `file`.

### TTS (Text To Speech)
```
POST /api/tts?text=Câu+muốn+nói
```

```
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
chmod +x cloudflared
./cloudflared tunnel --url http://127.0.0.1:8000
```

---

## Chạy Server & Test (Swagger UI)

Khi server đang chạy, truy cập vào `http://localhost:8000/docs` để sử dụng Swagger UI trực tiếp test các API.
Frontend có thể gọi đến backend bằng API_KEYS tùy theo cấu hình `.env` của bạn.

---

## Chạy Bằng Docker CUDA

Image backend dùng NVIDIA CUDA `13.2.1`:

```dockerfile
FROM nvidia/cuda:13.2.1-cudnn-runtime-ubuntu24.04
```

Yêu cầu máy host:

```bash
nvidia-smi
docker --version
docker compose version
```

Nếu tự quản lý máy Linux, cần cài NVIDIA Container Toolkit để Docker thấy GPU.
Trên Vast.ai thường image GPU đã có sẵn driver/NVIDIA runtime; chỉ cần kiểm tra `nvidia-smi`.

Compose mặc định dùng `.env.docker.example`. Nếu muốn chỉnh cấu hình riêng:

```bash
cp .env.docker.example .env
```

Build và chạy API + MySQL:

```bash
docker compose up --build
```

Chạy với `.env` riêng:

```bash
ENV_FILE=.env docker compose up --build
```

Chạy nền:

```bash
docker compose up -d --build
```

Xem log:

```bash
docker compose logs -f api
```

Kiểm tra GPU trong container:

```bash
docker compose exec api nvidia-smi
```

Kiểm tra API:

```bash
curl http://localhost:8000/health
```

Swagger UI:

```text
http://localhost:8000/docs
```

Dừng service:

```bash
docker compose down
```

Xóa cả dữ liệu MySQL volume:

```bash
docker compose down -v
```
