"""HTTP API for document storage, retrieval, and one answer endpoint."""

import asyncio
import hmac
import json
import logging
import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRouter

from answer_service import AnswerService
from document_service import DocumentService
from index_store import IndexStore
from models import AskRequest, ImportRequest, MultiSearchRequest, ParseBatchRequest, ParseRequest, SearchRequest
from retrieval import RetrievalIndex
from source_security import MAX_SOURCE_BYTES, fetch_remote_document
from text_extractor import EnhancedTextExtractionService


load_dotenv()
load_dotenv(dotenv_path=".env.local")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("docura")


def configured_tokens() -> dict[str, str]:
    tokens = {}
    legacy = os.getenv("DOCURA_API_TOKEN")
    if legacy:
        tokens["default"] = legacy
    configured = os.getenv("DOCURA_API_TOKENS_JSON")
    if configured:
        parsed = json.loads(configured)
        if not isinstance(parsed, dict) or not all(
            isinstance(owner, str) and owner and isinstance(token, str) and token
            for owner, token in parsed.items()
        ):
            raise ValueError("DOCURA_API_TOKENS_JSON must map owners to nonempty tokens")
        tokens.update(parsed)
    if not tokens or len(set(tokens.values())) != len(tokens):
        raise ValueError("Configure at least one unique Docura API token")
    return tokens


async def cleanup_expired(app: FastAPI):
    while True:
        await asyncio.sleep(300)
        try:
            removed = await asyncio.to_thread(app.state.documents.cleanup_expired)
            app.state.indexes.invalidate(removed)
        except Exception as error:
            logger.warning("Expiry cleanup failed: %s", type(error).__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.tokens = configured_tokens()
    data_dir = os.getenv("DOCURA_DATA_DIR", "./data")
    extractor = EnhancedTextExtractionService()
    app.state.documents = DocumentService(
        data_dir, extractor=extractor,
        ttl_seconds=int(os.getenv("DOCURA_DOCUMENT_TTL_SECONDS", "86400")),
    )
    shared_models = RetrievalIndex()
    app.state.indexes = IndexStore(
        app.state.documents, shared_models=shared_models,
        max_entries=int(os.getenv("DOCURA_MAX_CACHED_INDEXES", "8")),
        ttl_seconds=int(os.getenv("DOCURA_INDEX_TTL_SECONDS", "3600")),
    )
    app.state.answers = AnswerService()
    removed = await asyncio.to_thread(app.state.documents.cleanup_expired)
    app.state.indexes.invalidate(removed)
    cleanup_task = asyncio.create_task(cleanup_expired(app))
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Docura API", version="3.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("DOCURA_ALLOWED_ORIGINS", "").split(",")
                   if origin.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)


def verify_token(request: Request, authorization: str = Header(default="")) -> str:
    owner = None
    for candidate, token in request.app.state.tokens.items():
        if hmac.compare_digest(authorization, f"Bearer {token}"):
            owner = candidate
    if owner is None:
        raise HTTPException(status_code=401, detail="Invalid authorization token")
    request.state.owner = owner
    return owner


router = APIRouter(prefix="/api/v1", dependencies=[Depends(verify_token)])


def owner_of(request: Request) -> str:
    return request.state.owner


async def index_for(request: Request, sources: list[str]) -> RetrievalIndex:
    try:
        index, _ = await asyncio.to_thread(
            request.app.state.indexes.get_or_build, owner_of(request), sources,
        )
        return index
    except LookupError:
        raise HTTPException(status_code=404, detail="Document not found or expired")
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))


async def parse_document(request: Request, document_id: str) -> dict:
    try:
        record = request.app.state.documents.get_records(owner_of(request), [document_id])[0]
        result = await asyncio.to_thread(request.app.state.documents.extract, record)
    except LookupError:
        raise HTTPException(status_code=404, detail="Document not found or expired")
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    if not result.success or not result.text.strip():
        raise HTTPException(status_code=422, detail=result.error or "No text extracted")
    return {"text": result.text, "metadata": result.metadata, "status": "success"}


@router.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    data = bytearray()
    while chunk := await file.read(64 * 1024):
        data.extend(chunk)
        if len(data) > MAX_SOURCE_BYTES:
            raise HTTPException(status_code=413, detail="Document exceeds the size limit")
    try:
        record = await asyncio.to_thread(
            request.app.state.documents.store_bytes,
            owner_of(request), file.filename or "", bytes(data),
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    return {"id": f"upload:{record.document_id}", "filename": record.filename,
            "size": record.size, "expires_at": record.expires_at}


@router.post("/documents/import")
async def import_document(request: Request, payload: ImportRequest):
    try:
        data, filename = await asyncio.to_thread(fetch_remote_document, payload.url)
        record = await asyncio.to_thread(
            request.app.state.documents.store_bytes, owner_of(request), filename, data,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        logger.warning("Remote import failed: %s", type(error).__name__)
        raise HTTPException(status_code=502, detail="Remote document import failed")
    return {"id": f"upload:{record.document_id}", "filename": record.filename,
            "size": record.size, "expires_at": record.expires_at}


@router.delete("/documents/{document_id}")
async def delete_document(request: Request, document_id: str):
    source = document_id if document_id.startswith("upload:") else f"upload:{document_id}"
    try:
        deleted = await asyncio.to_thread(
            request.app.state.documents.delete, owner_of(request), source,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    request.app.state.indexes.invalidate([source.removeprefix("upload:")])
    return {"deleted": True}


@router.post("/parse")
async def parse_single_document(request: Request, payload: ParseRequest):
    return await parse_document(request, payload.document_id)


@router.post("/parse/batch")
async def parse_batch_documents(request: Request, payload: ParseBatchRequest):
    async def parse_one(source: str):
        try:
            return {"document_id": source, **await parse_document(request, source)}
        except HTTPException as error:
            return {"document_id": source, "status": "failed", "error": error.detail}
    results = await asyncio.gather(*(parse_one(source) for source in payload.document_ids))
    return {"results": results, "successful": sum(item["status"] == "success" for item in results)}


@router.post("/search")
async def search_documents(request: Request, payload: SearchRequest):
    index = await index_for(request, payload.documents)
    return await asyncio.to_thread(index.advanced_search, payload.query,
                                   payload.strategy, payload.top_k)


@router.post("/search/multi")
async def multi_search_documents(request: Request, payload: MultiSearchRequest):
    index = await index_for(request, payload.documents)
    results = []
    for query in payload.queries:
        result = await asyncio.to_thread(index.advanced_search, query,
                                         payload.strategy, payload.top_k)
        results.append({"query": query, "results": result})
    return {"results": results, "total_queries": len(results)}


@router.get("/search/strategies")
async def search_strategies():
    return {"strategies": ["semantic", "lexical", "hybrid", "ensemble"],
            "default": "ensemble"}


@router.get("/search/stats")
async def search_stats(request: Request):
    return request.app.state.indexes.stats(owner_of(request))


@router.get("/cache/stats")
async def cache_stats(request: Request):
    return request.app.state.documents.stats(owner_of(request))


@router.post("/cache/clear")
async def clear_cache(request: Request):
    owner = owner_of(request)
    await asyncio.to_thread(request.app.state.documents.clear_extractions, owner)
    request.app.state.indexes.clear(owner)
    return {"cleared": True}


@router.post("/documents/cleanup-expired")
async def clear_expired_documents(request: Request):
    removed = await asyncio.to_thread(request.app.state.documents.cleanup_expired)
    request.app.state.indexes.invalidate(removed)
    return {"removed": len(removed)}


@router.post("/ask")
async def ask(request: Request, payload: AskRequest):
    if len(payload.history) > 12:
        raise HTTPException(status_code=413, detail="Too many conversation turns")
    index = await index_for(request, payload.documents)
    try:
        answer = await request.app.state.answers.ask(
            index, payload.question, payload.search_strategy, payload.history,
        )
    except RuntimeError as error:
        logger.warning("Answer generation failed: %s", type(error).__name__)
        raise HTTPException(status_code=502, detail="Answer generation failed")
    return {"answer": answer, "documents": payload.documents}


@router.get("/health")
async def api_health():
    return {"status": "healthy", "timestamp": time.time()}


app.include_router(router)


@app.middleware("http")
async def request_metrics(request: Request, call_next):
    if request.url.path == "/api/v1/upload":
        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit() and int(content_length) > MAX_SOURCE_BYTES + 1024 * 1024:
            return JSONResponse(status_code=413, content={"detail": "Document exceeds the size limit"})
    start = time.monotonic()
    response = await call_next(request)
    logger.info("%s %s %s %.3fs", request.method, request.url.path,
                response.status_code, time.monotonic() - start)
    return response


@app.get("/")
async def root():
    return {"name": "Docura", "answer_endpoint": "/api/v1/ask",
            "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": time.time()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
