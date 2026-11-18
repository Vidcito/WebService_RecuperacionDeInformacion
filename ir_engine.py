"""Motor de recuperación de información: preprocesamiento"""
import json
import math
import re
from collections import Counter
from functools import lru_cache

import numpy as np
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOPWORDS = frozenset(ENGLISH_STOP_WORDS)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_stemmer = PorterStemmer()


@lru_cache(maxsize=None)
def _stem(token):
    return _stemmer.stem(token)


def preprocess(text):
    """minúsculas -> tokens alfanuméricos -> sin stopwords -> stemming de Porter."""
    tokens = _TOKEN_RE.findall((text or "").lower())
    return [_stem(t) for t in tokens if t not in STOPWORDS and len(t) > 1]


def load_corpus_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

class InvertedIndex:
    def __init__(self, docs):
        """Construye el vocabulario e índice invertido conservando los IDs originales."""
        self.doc_ids = [str(d["_id"]) for d in docs]
        self.titles = [d.get("title") or "" for d in docs]
        self.texts = [d.get("text") or "" for d in docs]
        self.N = len(docs)

        # Vocabulario e índice invertido crudo: término -> {doc_idx: tf}
        raw = {}
        self.doc_len = np.zeros(self.N)
        for i in range(self.N):
            tokens = preprocess(self.titles[i] + " " + self.texts[i])
            self.doc_len[i] = len(tokens)
            for term, tf in Counter(tokens).items():
                raw.setdefault(term, {})[i] = tf

        self.vocab = {t: j for j, t in enumerate(sorted(raw))}
        self.avgdl = float(self.doc_len.mean())

        # Postings vectorizados en numpy: (doc_idx, tf)
        self.postings, self.df = {}, {}
        for term, d in raw.items():
            idx = np.fromiter(d.keys(), dtype=np.int64, count=len(d))
            tf = np.fromiter(d.values(), dtype=np.float64, count=len(d))
            self.postings[term] = (idx, tf)
            self.df[term] = len(d)

class TFIDFSearchEngine(InvertedIndex):
    def __init__(self, docs):
        super().__init__(docs)
        self.idf_tfidf = {t: math.log((1 + self.N) / (1 + df)) + 1 for t, df in self.df.items()}

        # Pre-cálculo de normas L2 de cada documento para similitud coseno
        sq = np.zeros(self.N)
        for t, (idx, tf) in self.postings.items():
            w = (1 + np.log(tf)) * self.idf_tfidf[t]
            np.add.at(sq, idx, w * w)
        self.doc_norm = np.sqrt(sq)
        self.doc_norm[self.doc_norm == 0] = 1.0

    def score(self, q_terms):
        scores = np.zeros(self.N)
        q_w = {t: (1 + math.log(c)) * self.idf_tfidf[t] for t, c in q_terms.items()}
        q_norm = math.sqrt(sum(w * w for w in q_w.values())) or 1.0
        for t, wq in q_w.items():
            idx, tf = self.postings[t]
            scores[idx] += wq * (1 + np.log(tf)) * self.idf_tfidf[t]
        return scores / (self.doc_norm * q_norm)

    def search(self, query, top_k=10):
        if top_k <= 0:
            return []
        q_terms = Counter(t for t in preprocess(query) if t in self.postings)
        if not q_terms:
            return []
        scores = self.score(q_terms)
        k = min(top_k, self.N)
        cand = np.argpartition(-scores, k - 1)[:k]
        cand = cand[np.lexsort((cand, -scores[cand]))]
        out = []
        for i in cand:
            if scores[i] <= 0:
                break
            out.append({
                "rank": len(out) + 1,
                "document_id": self.doc_ids[i],
                "score": round(float(scores[i]), 6),
                "title": self.titles[i],
                "snippet": self.texts[i][:300],
            })
        return out

class BM25SearchEngine(InvertedIndex):
    def __init__(self, docs, k1=1.5, b=0.75):
        super().__init__(docs)
        self.k1, self.b = k1, b
        self.idf_bm25 = {t: math.log(1 + (self.N - df + 0.5) / (df + 0.5)) for t, df in self.df.items()}

    def score(self, q_terms):
        scores = np.zeros(self.N)
        for t, qtf in q_terms.items():
            idx, tf = self.postings[t]
            denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[idx] / self.avgdl)
            scores[idx] += qtf * self.idf_bm25[t] * tf * (self.k1 + 1) / denom
        return scores

    def search(self, query, top_k=10):
        if top_k <= 0:
            return []
        q_terms = Counter(t for t in preprocess(query) if t in self.postings)
        if not q_terms:
            return []
        scores = self.score(q_terms)
        k = min(top_k, self.N)
        cand = np.argpartition(-scores, k - 1)[:k]
        cand = cand[np.lexsort((cand, -scores[cand]))]
        out = []
        for i in cand:
            if scores[i] <= 0:
                break
            out.append({
                "rank": len(out) + 1,
                "document_id": self.doc_ids[i],
                "score": round(float(scores[i]), 6),
                "title": self.titles[i],
                "snippet": self.texts[i][:300],
            })
        return out

class SearchEngine:
    """Fachada unificada que expone ambos modelos (TF-IDF y BM25) sobre un único índice compartido."""
    def __init__(self, docs, k1=1.5, b=0.75):
        self.tfidf = TFIDFSearchEngine(docs)
        self.bm25 = BM25SearchEngine(docs, k1=k1, b=b)
        # Reutiliza el mismo índice invertido en memoria para evitar duplicación
        self.bm25.postings = self.tfidf.postings
        self.bm25.vocab = self.tfidf.vocab
        self.N = self.tfidf.N

    def search(self, query, model="bm25", top_k=10):
        if model == "bm25":
            return self.bm25.search(query, top_k=top_k)
        return self.tfidf.search(query, top_k=top_k)

# ---- Métricas de evaluación ----
def precision_at_k(ranked, rel, k=10):
    return sum(1 for d in ranked[:k] if rel.get(d, 0) > 0) / k


def recall_at_k(ranked, rel, k=10):
    n_rel = sum(1 for v in rel.values() if v > 0)
    return sum(1 for d in ranked[:k] if rel.get(d, 0) > 0) / n_rel if n_rel else 0.0


def ndcg_at_k(ranked, rel, k=10):
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0
