"""Retrieval API - Hybrid search service."""

from fastapi import FastAPI
from .retrieve import router as retrieve_router
from .bm25_endpoint import router as bm25_router

app = FastAPI(title="HRChat Retrieval API")

app.include_router(retrieve_router)
app.include_router(bm25_router)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "service": "hrchat-retrieval"}
