import json
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import rag

load_dotenv()
BASE_DIR = Path(__file__).parent


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Warm up the embedding model in the background so the first upload is fast.
    threading.Thread(target=rag.get_embeddings, daemon=True).start()
    yield


app = FastAPI(title="RetrivAI", lifespan=lifespan)


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1)
    model: str = rag.GROQ_MODELS[0]
    temperature: float = Field(0.2, ge=0, le=1)
    top_k: int = Field(4, ge=1, le=10)


@app.get("/api/config")
def config():
    return {
        "models": rag.GROQ_MODELS,
        "extensions": rag.SUPPORTED_EXTENSIONS,
    }


@app.get("/api/session/{session_id}")
def get_session_state(session_id: str):
    s = rag.get_session(session_id)
    return {"files": s.files, "messages": s.messages}


@app.post("/api/upload")
def upload(
    session_id: str = Form(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(150),
    files: list[UploadFile] = File(...),
):
    payload = [(f.filename or "file", f.file.read()) for f in files]
    session = rag.get_session(session_id)
    try:
        return rag.ingest_files(session, payload, chunk_size, chunk_overlap)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/chat")
def chat(req: ChatRequest):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="Missing Groq API key. Set GROQ_API_KEY in .env.",
        )
    session = rag.get_session(req.session_id)

    def event_stream():
        try:
            for event in rag.stream_answer(
                session, req.message, api_key, req.model, req.temperature, req.top_k
            ):
                yield json.dumps(event) + "\n"
        except Exception as e:  # noqa: BLE001 - surface API/model errors in the UI
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.delete("/api/session/{session_id}/chat")
def clear_chat(session_id: str):
    rag.clear_chat(rag.get_session(session_id))
    return {"ok": True}


@app.delete("/api/session/{session_id}/documents")
def clear_documents(session_id: str):
    rag.clear_documents(rag.get_session(session_id))
    return {"ok": True}


# Frontend (must be mounted last so /api routes take priority)
app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
