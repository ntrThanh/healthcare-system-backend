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

# Copy file .env và cấu hình (đặc biệt là API_KEYS và các MODEL_PATH)
cp .env.example .env

# Chạy server
# Tại thư mục gốc của project (medical_voice_rag_app/ai_core)
uvicorn ai_server.app.main:app --host 0.0.0.0 --port 8000 --reload
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
