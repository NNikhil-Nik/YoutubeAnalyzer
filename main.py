"""
FastAPI server for YouTube Video Q&A
Run with: .venv/bin/uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ai_processor import (
    extract_video_id,
    fetch_transcript,
    load_or_build_index,
    generate_answer,
    GENERATION_MODEL,
)
import vector_store

app = FastAPI(title="YouTube Q&A API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ProcessVideoRequest(BaseModel):
    youtube_url: str

class ProcessVideoResponse(BaseModel):
    message: str
    video_id: str
    cached: bool

class AskQuestionRequest(BaseModel):
    video_id: str
    question: str

class AskQuestionResponse(BaseModel):
    answer: str
    model: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")


@app.get("/active_model")
def active_model():
    return {"model": GENERATION_MODEL}


@app.post("/process_video", response_model=ProcessVideoResponse)
def process_video(body: ProcessVideoRequest):
    video_id = extract_video_id(body.youtube_url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL.")

    cached = vector_store.is_video_cached(video_id)

    if not cached:
        try:
            transcript = fetch_transcript(video_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        try:
            load_or_build_index(transcript, video_id)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to build index: {exc}")

    return ProcessVideoResponse(
        message="Video ready." if cached else "Transcript processed and indexed.",
        video_id=video_id,
        cached=cached,
    )


@app.post("/ask_question", response_model=AskQuestionResponse)
def ask_question(body: AskQuestionRequest):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not vector_store.is_video_cached(body.video_id):
        raise HTTPException(status_code=404, detail="Video not indexed. Process the video first.")

    try:
        chunks = vector_store.retrieve_chunks(body.question, body.video_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    answer = generate_answer(body.question, chunks)
    return AskQuestionResponse(answer=answer, model=GENERATION_MODEL)
