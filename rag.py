"""Core RAG logic: file ingestion, retrieval and streaming answers.

Stack: LangChain + Groq (chat LLM) + local HuggingFace embeddings + FAISS.
Sessions are kept in memory, keyed by a session id sent from the browser.
"""

import os
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from threading import Lock

from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Groq model names change over time - see https://console.groq.com/docs/models
# Use the requested model as the default choice for chat requests.
GROQ_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_HISTORY_MESSAGES = 6

LOADERS = {
    ".pdf": lambda p: PyPDFLoader(p),
    ".txt": lambda p: TextLoader(p, encoding="utf-8"),
    ".md": lambda p: TextLoader(p, encoding="utf-8"),
    ".docx": lambda p: Docx2txtLoader(p),
    ".csv": lambda p: CSVLoader(p, encoding="utf-8"),
}
SUPPORTED_EXTENSIONS = sorted(LOADERS)

CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the chat history and the latest user question, rewrite the "
            "question so it can be understood on its own. Do NOT answer it. "
            "If it is already standalone, return it unchanged.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers questions using the context "
            "taken from the user's uploaded documents.\n"
            "- Base your answer on the context below.\n"
            "- If the answer is not in the context, say you couldn't find it in the "
            "documents instead of guessing.\n"
            "- Be concise and clear.\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

PLAIN_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)


# ------------------------------ sessions -------------------------------------
@dataclass
class Session:
    vectorstore: FAISS | None = None
    files: list = field(default_factory=list)
    messages: list = field(default_factory=list)  # {"role", "content", "sources"}
    lock: object = field(default_factory=Lock)


_SESSIONS: dict[str, Session] = {}
_SESSIONS_LOCK = Lock()


def get_session(session_id: str) -> Session:
    with _SESSIONS_LOCK:
        if session_id not in _SESSIONS:
            _SESSIONS[session_id] = Session()
        return _SESSIONS[session_id]


def clear_chat(session: Session) -> None:
    session.messages = []


def clear_documents(session: Session) -> None:
    with session.lock:
        session.vectorstore = None
        session.files = []


# ------------------------------ ingestion ------------------------------------
@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def ingest_files(session: Session, files, chunk_size: int = 1000, chunk_overlap: int = 150):
    """files: list of (filename, bytes). Adds them to the session's vector store."""
    docs, loaded_names, skipped = [], [], []

    for name, data in files:
        suffix = Path(name).suffix.lower()
        if suffix not in LOADERS:
            skipped.append(f"{name} (unsupported type)")
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            path = tmp.name
        try:
            loaded = LOADERS[suffix](path).load()
            for d in loaded:
                d.metadata["source"] = name  # real filename instead of temp path
            if any(d.page_content.strip() for d in loaded):
                docs.extend(loaded)
                loaded_names.append(name)
            else:
                skipped.append(f"{name} (no extractable text)")
        except Exception as e:  # noqa: BLE001 - report per-file problems to the UI
            skipped.append(f"{name} ({e})")
        finally:
            os.remove(path)

    if not docs:
        detail = f" Skipped: {'; '.join(skipped)}" if skipped else ""
        raise ValueError("No readable text found in the uploaded files." + detail)

    chunk_overlap = max(0, min(chunk_overlap, chunk_size // 2))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]
    if not chunks:
        raise ValueError("No text could be extracted from the uploaded files.")

    with session.lock:
        if session.vectorstore is None:
            session.vectorstore = FAISS.from_documents(chunks, get_embeddings())
        else:
            session.vectorstore.add_documents(chunks)
        for n in loaded_names:
            if n not in session.files:
                session.files.append(n)
        all_files = list(session.files)

    return {
        "files": all_files,
        "added": loaded_names,
        "chunks": len(chunks),
        "skipped": skipped,
    }


# ------------------------------ answering ------------------------------------
def _format_context(docs) -> str:
    parts = []
    for d in docs:
        label = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        if page is not None:
            label += f", page {page + 1}"
        parts.append(f"[{label}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def _build_sources(docs) -> list:
    seen, out = set(), []
    for d in docs:
        label = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        if page is not None:
            label += f" (page {page + 1})"
        if label in seen:
            continue
        seen.add(label)
        out.append({"label": label, "snippet": " ".join(d.page_content.split())[:220]})
    return out


def _to_lc_history(messages) -> list:
    history = []
    for m in messages[-MAX_HISTORY_MESSAGES:]:
        cls = HumanMessage if m["role"] == "user" else AIMessage
        history.append(cls(content=m["content"]))
    return history


def stream_answer(
    session: Session,
    question: str,
    api_key: str,
    model: str,
    temperature: float = 0.2,
    top_k: int = 4,
):
    """Generator of events: {"type": "sources"|"token"|"done", ...}."""
    llm = ChatGroq(api_key=api_key, model=model, temperature=temperature)
    history = _to_lc_history(session.messages)
    sources: list = []

    if session.vectorstore is not None:
        # 1) make the follow-up question standalone using the chat history
        standalone = question
        if history:
            standalone = (CONDENSE_PROMPT | llm | StrOutputParser()).invoke(
                {"chat_history": history, "question": question}
            )
        # 2) retrieve relevant chunks
        retriever = session.vectorstore.as_retriever(search_kwargs={"k": top_k})
        retrieved = retriever.invoke(standalone)
        sources = _build_sources(retrieved)
        # 3) answer from the retrieved context
        chain = RAG_PROMPT | llm | StrOutputParser()
        payload = {
            "context": _format_context(retrieved),
            "chat_history": history,
            "question": question,
        }
    else:
        chain = PLAIN_PROMPT | llm | StrOutputParser()
        payload = {"chat_history": history, "question": question}

    yield {"type": "sources", "sources": sources}

    parts = []
    for token in chain.stream(payload):
        parts.append(token)
        yield {"type": "token", "text": token}

    session.messages.append({"role": "user", "content": question, "sources": []})
    session.messages.append(
        {"role": "assistant", "content": "".join(parts), "sources": sources}
    )
    session.messages = session.messages[-40:]
    yield {"type": "done"}
