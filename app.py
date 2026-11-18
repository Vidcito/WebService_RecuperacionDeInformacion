import os
from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ir_engine import SearchEngine, load_corpus_jsonl

CORPUS_PATH = os.environ.get("CORPUS_PATH", "corpus.jsonl")
CORPUS_NAME = os.environ.get("CORPUS_NAME", "scifact")

app = FastAPI(title="Servicio de recuperación de información")
engine = SearchEngine(load_corpus_jsonl(CORPUS_PATH))


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    model: Literal["bm25", "tfidf"] = "bm25"
    top_k: int = Field(10, ge=1, le=1000)


@app.get("/health")
def health():
    return {"status": "ok", "corpus": CORPUS_NAME, "documents": engine.N}


@app.post("/search")
def search(req: SearchRequest):
    return {"model": req.model, "results": engine.search(req.query, req.model, req.top_k)}
