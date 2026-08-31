"""
Semantic intent detection via Qdrant + FastEmbed.

Architecture
------------
- Seed a small catalogue of labelled example questions into a Qdrant collection.
- At query time, embed the user question and retrieve top-k nearest neighbours.
- Vote (score-weighted) across neighbour intents to produce a semantic label + confidence.
- Designed to work with:
    * QdrantClient(":memory:")          — zero infra, process-local (default)
    * QdrantClient(path="...")          — local on-disk persistence
    * QdrantClient(url=..., api_key=...)— remote / Qdrant Cloud

Falls back gracefully when qdrant-client / fastembed are unavailable.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "intent_examples"
# FastEmbed default BAAI/bge-small-en-v1.5 → 384 dims
EMBEDDING_DIM = 384

# ---------------------------------------------------------------------------
# Seed catalogue — extend freely; each example is embedded once at startup
# ---------------------------------------------------------------------------
INTENT_EXAMPLES: list[dict[str, str]] = [
    # Career
    {"text": "Should I consider changing my job in the next few months?", "intent": "career"},
    {"text": "Is this a good time to switch careers?", "intent": "career"},
    {"text": "Will I get a promotion at work this year?", "intent": "career"},
    {"text": "Should I resign from my current position?", "intent": "career"},
    {"text": "How is my professional life looking ahead?", "intent": "career"},
    {"text": "Is starting a business favourable right now?", "intent": "career"},
    {"text": "What does my chart say about my workplace prospects?", "intent": "career"},
    {"text": "Should I accept the new job offer?", "intent": "career"},
    # Relationship
    {"text": "How does this month look for my relationship?", "intent": "relationship"},
    {"text": "Will my marriage improve soon?", "intent": "relationship"},
    {"text": "Is this a good time for love and dating?", "intent": "relationship"},
    {"text": "How can I improve communication with my partner?", "intent": "relationship"},
    {"text": "Are we compatible according to the stars?", "intent": "relationship"},
    {"text": "When will I find a life partner?", "intent": "relationship"},
    {"text": "Is there tension in my marriage this period?", "intent": "relationship"},
    # Health
    {"text": "What should I focus on for my health?", "intent": "health"},
    {"text": "How is my energy and wellness looking this month?", "intent": "health"},
    {"text": "Should I be careful about illness right now?", "intent": "health"},
    {"text": "What does my chart say about stress and sleep?", "intent": "health"},
    {"text": "Is this a good period for fitness goals?", "intent": "health"},
    {"text": "How can I improve my physical well-being?", "intent": "health"},
    # Finance
    {"text": "Is this a good time to invest money?", "intent": "finance"},
    {"text": "Should I buy property this year?", "intent": "finance"},
    {"text": "How are my finances looking in the coming months?", "intent": "finance"},
    {"text": "Will I face money problems soon?", "intent": "finance"},
    {"text": "Should I take a loan right now?", "intent": "finance"},
    {"text": "What does my chart say about wealth and savings?", "intent": "finance"},
    # General
    {"text": "Can you summarize today's guidance?", "intent": "general"},
    {"text": "What should I prioritize this week?", "intent": "general"},
    {"text": "How does this month look overall for me?", "intent": "general"},
    {"text": "Give me a general overview of my chart influences.", "intent": "general"},
    {"text": "What is the main theme for me this year?", "intent": "general"},
    {"text": "Any advice for the coming days?", "intent": "general"},
]


@dataclass
class SemanticHit:
    intent: str
    score: float
    example: str


@dataclass
class SemanticIntentResult:
    intent: str
    confidence: float
    hits: list[SemanticHit]
    method: str  # "semantic" | "unavailable"


class SemanticIntentIndex:
    """
    Thin wrapper around Qdrant for intent-example similarity search.

    Thread-safe lazy init so FastAPI workers don't all re-seed on import.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ready = False
        self._client: Any = None
        self._embedder: Any = None
        self._error: str | None = None

    @property
    def available(self) -> bool:
        self._ensure_ready()
        return self._ready

    def _ensure_ready(self) -> None:
        if self._ready or self._error is not None:
            return
        with self._lock:
            if self._ready or self._error is not None:
                return
            try:
                self._init_qdrant()
                self._seed()
                self._ready = True
                logger.info(
                    "Semantic intent index ready (collection=%s, examples=%d, mode=%s)",
                    COLLECTION_NAME,
                    len(INTENT_EXAMPLES),
                    settings.qdrant_mode,
                )
            except Exception as exc:
                self._error = str(exc)
                logger.warning(
                    "Semantic intent index unavailable (%s); keyword fallback will be used",
                    exc,
                )

    def _init_qdrant(self) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        mode = (settings.qdrant_mode or "memory").lower()
        if mode == "memory":
            self._client = QdrantClient(location=":memory:")
        elif mode == "local":
            path = settings.qdrant_path or "./data/qdrant"
            self._client = QdrantClient(path=path)
        elif mode == "remote":
            if not settings.qdrant_url:
                raise RuntimeError("MYNAKSH_QDRANT_URL required for remote mode")
            self._client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
        else:
            raise RuntimeError(f"Unknown qdrant_mode: {mode}")

        # Prefer FastEmbed (ONNX, already installed). Fall back to hashing
        # only if fastembed is broken — still lets the pipeline demonstrate
        # vector search against Qdrant.
        try:
            from fastembed import TextEmbedding

            self._embedder = TextEmbedding(model_name=settings.embedding_model)
            # Probe dimension
            probe = list(self._embedder.embed(["dimension probe"]))[0]
            dim = len(probe)
        except Exception as emb_exc:
            logger.warning("FastEmbed unavailable (%s); using deterministic hash embedder", emb_exc)
            self._embedder = None
            dim = EMBEDDING_DIM

        existing = {c.name for c in self._client.get_collections().collections}
        if COLLECTION_NAME not in existing:
            self._client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._embedder is not None:
            return [vec.tolist() for vec in self._embedder.embed(texts)]
        # Deterministic fallback embedder (not semantic, but keeps Qdrant path testable)
        vectors: list[list[float]] = []
        for t in texts:
            h = hashlib.sha256(t.lower().encode()).digest()
            # Expand hash into a pseudo-vector
            raw = (h * ((EMBEDDING_DIM // len(h)) + 1))[:EMBEDDING_DIM]
            vec = [(b / 127.5) - 1.0 for b in raw]
            # L2 normalise
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            vectors.append([x / norm for x in vec])
        return vectors

    def _seed(self) -> None:
        from qdrant_client.models import PointStruct

        # Skip re-seed if collection already has points (local/remote persistence)
        info = self._client.get_collection(COLLECTION_NAME)
        if info.points_count and info.points_count >= len(INTENT_EXAMPLES):
            logger.info("Collection already seeded with %s points", info.points_count)
            return

        texts = [ex["text"] for ex in INTENT_EXAMPLES]
        vectors = self._embed(texts)
        points = [
            PointStruct(
                id=idx,
                vector=vectors[idx],
                payload={
                    "text": INTENT_EXAMPLES[idx]["text"],
                    "intent": INTENT_EXAMPLES[idx]["intent"],
                },
            )
            for idx in range(len(INTENT_EXAMPLES))
        ]
        self._client.upsert(collection_name=COLLECTION_NAME, points=points)
        logger.info("Seeded %d intent examples into Qdrant", len(points))

    def search(self, question: str, top_k: int = 5) -> SemanticIntentResult:
        self._ensure_ready()
        if not self._ready:
            return SemanticIntentResult(
                intent="general",
                confidence=0.0,
                hits=[],
                method="unavailable",
            )

        query_vec = self._embed([question])[0]
        response = self._client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=top_k,
            with_payload=True,
        )
        hits: list[SemanticHit] = []
        score_by_intent: dict[str, float] = {}
        for pt in response.points:
            intent = (pt.payload or {}).get("intent", "general")
            example = (pt.payload or {}).get("text", "")
            score = float(pt.score or 0.0)
            hits.append(SemanticHit(intent=intent, score=score, example=example))
            score_by_intent[intent] = score_by_intent.get(intent, 0.0) + score

        if not score_by_intent:
            return SemanticIntentResult(
                intent="general",
                confidence=0.3,
                hits=[],
                method="semantic",
            )

        best_intent = max(score_by_intent, key=score_by_intent.get)  # type: ignore[arg-type]
        total = sum(score_by_intent.values()) or 1.0
        # Map weighted share into a 0.4–0.98 band
        confidence = min(0.98, 0.4 + (score_by_intent[best_intent] / total) * 0.55)
        # Boost if top hit itself is strong
        if hits and hits[0].intent == best_intent and hits[0].score >= 0.75:
            confidence = min(0.98, confidence + 0.1)

        return SemanticIntentResult(
            intent=best_intent,
            confidence=confidence,
            hits=hits,
            method="semantic",
        )


# Process-wide singleton
semantic_index = SemanticIntentIndex()


def semantic_detect_intent(question: str, top_k: int = 5) -> tuple[str, float, str]:
    """
    Public helper used by the hybrid intent detector.

    Returns (intent, confidence, reasoning).
    """
    result = semantic_index.search(question, top_k=top_k)
    if result.method == "unavailable":
        return "general", 0.0, "Semantic index unavailable"

    hit_summary = ", ".join(f"{h.intent}:{h.score:.2f}" for h in result.hits[:3])
    reasoning = (
        f"Semantic top hits [{hit_summary}] → intent={result.intent} "
        f"(conf={result.confidence:.2f})"
    )
    logger.info(reasoning)
    return result.intent, result.confidence, reasoning
