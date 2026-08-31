"""
MyNaksh Personalized AI Context Engine

POST /auth/register             → register new user
POST /auth/login                → login and receive JWT
POST /personalize              → grounded AI answer + confidence + sources
POST /debug/personalization    → personalization decisions without LLM call
GET  /health                   → liveness
WS   /ws/chat                  → continuous chat session (auth via ?token=)
GET  /                         → frontend chat UI
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings  # noqa: F401 — used by routes & lifespan
from app.logging_config import RequestLoggingMiddleware, setup_logging
from app.models.schemas import (
    DebugPersonalizationResponse,
    PersonalizeRequest,
    PersonalizeResponse,
    SemanticHitSchema,
    SemanticSearchResponse,
)
from app.routes.auth import router as auth_router
from app.services.auth import (
    UserOut,
    _decode_token,
    close_db,
    get_current_user,
    get_current_user_ws,
    init_db,
)
from app.services.llm import get_llm_provider
from app.services.personalization import build_personalization, extract_context_payload
from app.services.prompt_builder import build_messages, prompt_char_count
from app.services.upstream import fetch_all_context
from app.services.semantic_intent import semantic_index

setup_logging()
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "app" / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> Any:
    logger.info(
        "Starting %s v%s (llm_provider=%s)",
        settings.app_name,
        settings.app_version,
        settings.llm_provider,
    )
    await init_db()
    try:
        yield
    finally:
        await close_db()
        logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Intelligence layer between MyNaksh astrological backend services and the LLM. "
        "Gathers context, detects intent, selects relevant sources, personalizes tone/language, "
        "builds an optimized prompt, and returns a grounded answer."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Auth routes (no auth required)
app.include_router(auth_router)


@app.get("/auth/me")
async def auth_me(token: str) -> dict:
    try:
        user = await _decode_token(token)
    except HTTPException:
        raise HTTPException(status_code=401, detail="Invalid or expired token") from None
    return {"id": user.id, "email": user.email, "name": user.name}


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
        "llm_provider": settings.llm_provider,
    }


@app.get("/")
async def serve_frontend() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    await websocket.accept()

    token = websocket.query_params.get("token", "")
    if not token:
        await websocket.send_json({"type": "error", "message": "Authentication token required."})
        await websocket.close()
        return

    try:
        current_user = await get_current_user_ws(token)
    except HTTPException:
        await websocket.send_json({"type": "error", "message": "Invalid or expired token."})
        await websocket.close()
        return

    user_id = current_user.id
    conversation_history: list[dict[str, Any]] = []

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON payload."})
                continue

            if payload.get("action") == "end":
                await websocket.send_json({"type": "info", "message": "Chat ended by user."})
                await websocket.close()
                return

            question = payload.get("question", "").strip()

            if not question:
                await websocket.send_json({"type": "error", "message": "question is required."})
                continue

            conversation_history.append({"role": "user", "content": question})

            pipeline_start = time.perf_counter()
            logger.info("WebSocket chat userId=%s question=%r", user_id, question[:120])

            try:
                bundle = await fetch_all_context(user_id)
                if bundle.user is None and "user" in bundle.errors:
                    logger.warning("User profile unavailable; continuing with partial context")

                config = build_personalization(question, bundle)
                context_payload = extract_context_payload(config.selectedContext, bundle)

                if not context_payload:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Unable to gather sufficient astrological context. Please try again shortly.",
                    })
                    continue

                messages = build_messages(question, config, context_payload)
                p_chars = prompt_char_count(messages)
                logger.info("Prompt size=%s chars | maxWords=%s | intent=%s", p_chars, config.maxWords, config.intent)

                llm = get_llm_provider()
                result = await llm.generate(messages, config, context_payload)

                total_ms = (time.perf_counter() - pipeline_start) * 1000
                logger.info(
                    "WebSocket complete userId=%s intent=%s confidence=%s llm_ms=%.1f total_ms=%.1f sources=%s",
                    user_id,
                    config.intent,
                    result.confidence,
                    result.latency_ms,
                    total_ms,
                    result.sourcesUsed,
                )

                conversation_history.append({"role": "assistant", "content": result.answer})

                await websocket.send_json({
                    "type": "response",
                    "answer": result.answer,
                    "confidence": result.confidence,
                    "sourcesUsed": result.sourcesUsed,
                })

            except Exception as exc:
                logger.exception("WebSocket LLM generation failed: %s", exc)
                await websocket.send_json({
                    "type": "error",
                    "message": "AI generation temporarily unavailable. Please retry.",
                })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected userId=%s", user_id)


@app.get("/panchang")
async def get_panchang(
    date: str = "2026-08-13",
    lat: float = 23.02,
    lon: float = 72.57
) -> dict:
    url = f"https://hindutithi.in/api/day?date={date}&lat={lat}&lon={lon}"
    headers = {
        "Authorization": "Bearer hindutithi_live_5jv8QdNrPZ3R8LkcvsztPawaF8tOd7eqW6FGSJb7"
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            logger.error("Failed to fetch panchang data: %s", exc)
            raise HTTPException(status_code=502, detail="Failed to fetch Panchang data from external service.") from exc


@app.post("/personalize", response_model=PersonalizeResponse)
async def personalize(
    body: PersonalizeRequest,
    current_user: UserOut = Depends(get_current_user),
) -> PersonalizeResponse:
    """
    Full personalization pipeline:
      1. Concurrent upstream fetch (user, kundli, horoscope, panchang)
      2. Intent detection + config-driven source selection
      3. Optimized prompt construction
      4. LLM generation (mock or real provider)
    """
    user_id = current_user.id
    pipeline_start = time.perf_counter()
    logger.info("Personalize request userId=%s question=%r", user_id, body.question[:120])

    # 1. Gather context
    bundle = await fetch_all_context(user_id)
    if bundle.user is None and "user" in bundle.errors:
        # Soft-fail: continue with empty profile rather than hard 500
        logger.warning("User profile unavailable; continuing with partial context")

    # 2. Personalization engine
    config = build_personalization(body.question, bundle)
    context_payload = extract_context_payload(config.selectedContext, bundle)

    if not context_payload:
        logger.warning("No context materialised for userId=%s", user_id)
        raise HTTPException(
            status_code=503,
            detail="Unable to gather sufficient astrological context. Please try again shortly.",
        )

    # 3. Prompt
    messages = build_messages(body.question, config, context_payload)
    p_chars = prompt_char_count(messages)
    logger.info("Prompt size=%s chars | maxWords=%s | intent=%s", p_chars, config.maxWords, config.intent)

    # 4. LLM
    llm = get_llm_provider()
    try:
        result = await llm.generate(messages, config, context_payload)
    except Exception as exc:
        logger.exception("LLM generation failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="AI generation temporarily unavailable. Please retry.",
        ) from exc

    total_ms = (time.perf_counter() - pipeline_start) * 1000
    logger.info(
        "Personalize complete userId=%s intent=%s confidence=%s llm_ms=%.1f total_ms=%.1f sources=%s",
        user_id,
        config.intent,
        result.confidence,
        result.latency_ms,
        total_ms,
        result.sourcesUsed,
    )

    return PersonalizeResponse(
        answer=result.answer,
        confidence=result.confidence,
        sourcesUsed=result.sourcesUsed,
    )


@app.post("/debug/personalization", response_model=DebugPersonalizationResponse)
async def debug_personalization(
    body: PersonalizeRequest,
    current_user: UserOut = Depends(get_current_user),
) -> DebugPersonalizationResponse:
    """
    Inspect Personalization Engine decisions without invoking the LLM.
    Useful for validating intent detection and source selection.
    Includes Qdrant semantic nearest-neighbour hits when available.
    """
    user_id = current_user.id
    logger.info("Debug personalization userId=%s question=%r", user_id, body.question[:120])
    bundle = await fetch_all_context(user_id)
    config = build_personalization(body.question, bundle)

    semantic_hits: list[SemanticHitSchema] | None = None
    semantic_available = semantic_index.available
    if semantic_available:
        result = semantic_index.search(body.question, top_k=settings.semantic_top_k)
        semantic_hits = [
            SemanticHitSchema(intent=h.intent, score=round(h.score, 4), example=h.example)
            for h in result.hits
        ]

    return DebugPersonalizationResponse(
        intent=config.intent,
        selectedContext=config.selectedContext,
        excludedContext=config.excludedContext,
        language=config.language,
        tone=config.tone,
        maxWords=config.maxWords,
        reasoning=config.reasoning,
        intentMode=settings.intent_mode,
        semanticHits=semantic_hits,
        semanticAvailable=semantic_available,
    )


@app.post("/debug/semantic", response_model=SemanticSearchResponse)
async def debug_semantic(
    body: PersonalizeRequest,
    current_user: UserOut = Depends(get_current_user),
) -> SemanticSearchResponse:
    """
    Pure Qdrant semantic search over seeded intent examples.
    Does not run the full personalization pipeline.
    """
    result = semantic_index.search(body.question, top_k=settings.semantic_top_k)
    return SemanticSearchResponse(
        query=body.question,
        intent=result.intent,
        confidence=round(result.confidence, 4),
        method=result.method,
        hits=[
            SemanticHitSchema(intent=h.intent, score=round(h.score, 4), example=h.example)
            for h in result.hits
        ],
        available=semantic_index.available,
    )
