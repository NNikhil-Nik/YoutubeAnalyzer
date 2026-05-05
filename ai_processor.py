
# Prerequisites: Python 3.10+ and Git installed.                                                                                    
                                                                                                                                  
#   ---                                                                                                                               
#   1. Clone / navigate to the project                                                                                                
#   cd /Users/nikhil.varshney/YoutubeAnalyzer                                                                                         
                                                                                                                                    
#   2. Create the virtual environment                                                                                                 
#   python3 -m venv .venv                                                                                                             
                                                                                                                                    
#   3. Activate it                                                                                                                    
#   source .venv/bin/activate                                                                                                         
                                                                                                                                    
#   4. Install dependencies                                                                                                           
#   pip install -r requirements.txt                                                                                                   
                                                                                                                                    
#   5. Add your Gemini API key to .env                                                                                                
                                                                                                                                    
#   The .env file already exists. Just make sure it contains:                                                                         
#   GEMINI_API_KEY=your_actual_key_here                                                                                               
#   Get a free key at aistudio.google.com.                                                                                            
                                                                                                                                    
#   6. Run                                                                                                                            
#   python ai_processor.py                                                                                                            
                                                                                                                                    
#   ---                                                                                                                               
#   Next time (env already set up):                           
#   cd /Users/nikhil.varshney/YoutubeAnalyzer                                                                                         
#   source .venv/bin/activate                                 
#   python ai_processor.py                                                                                                            
                                                            
# ✻ Cooked for 12s 
"""
YouTube Video Q&A — AI Processing Module
=========================================
Setup:
  1. Create a .env file in this directory with:
       GEMINI_API_KEY=your_api_key_here
  2. Install dependencies:
       pip install google-genai youtube-transcript-api langchain langchain-community chromadb python-dotenv
"""

import os
import re

from dotenv import load_dotenv
import requests
from http.cookiejar import MozillaCookieJar
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from youtube_transcript_api.proxies import WebshareProxyConfig

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings

from google import genai
from google.genai import types
import vector_store

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY not found. Add it to your .env file.")

client = genai.Client(api_key=GEMINI_API_KEY)

GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-2.5-flash")
# Gemma models don't support system_instruction — it must be prepended into the prompt
_SUPPORTS_SYSTEM_INSTRUCTION = "gemma" not in GENERATION_MODEL.lower()


class GeminiEmbeddings(Embeddings):
    """Thin LangChain-compatible wrapper around the google-genai embed_content API."""

    def __init__(self, model: str = "models/gemini-embedding-001"):
        self.model = model

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = client.models.embed_content(model=self.model, contents=texts)
        return [e.values for e in response.embeddings]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


def extract_video_id(url: str) -> str | None:
    """
    Extract YouTube video ID from both youtube.com/watch?v= and youtu.be/ formats.
    Returns None if the URL is not a recognised YouTube link.
    """
    patterns = [
        r"(?:youtube\.com/watch\?.*v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def _build_ytt_api() -> YouTubeTranscriptApi:
    """
    Build a YouTubeTranscriptApi instance with whichever auth strategy is configured:
      1. Webshare rotating proxy  (WEBSHARE_PROXY_USERNAME + WEBSHARE_PROXY_PASSWORD)
      2. Browser cookies file     (YOUTUBE_COOKIES_PATH)
      3. Plain unauthenticated    (works on most local IPs)
    """
    username = os.getenv("WEBSHARE_PROXY_USERNAME", "").strip()
    password = os.getenv("WEBSHARE_PROXY_PASSWORD", "").strip()
    cookies_path = os.getenv("YOUTUBE_COOKIES_PATH", "").strip()

    proxy_config = None
    if username and password:
        proxy_config = WebshareProxyConfig(
            proxy_username=username,
            proxy_password=password,
        )

    http_client = None
    if cookies_path and os.path.isfile(cookies_path):
        jar = MozillaCookieJar(cookies_path)
        jar.load(ignore_discard=True, ignore_expires=True)
        session = requests.Session()
        session.cookies = jar
        http_client = session

    return YouTubeTranscriptApi(proxy_config=proxy_config, http_client=http_client)


def fetch_transcript(video_id: str) -> str:
    """
    Fetch the transcript for a YouTube video.
    Returns the full transcript as a plain string, or raises a descriptive error.
    """
    try:
        snippets = _build_ytt_api().fetch(video_id)
        return " ".join(snippet.text for snippet in snippets)
    except TranscriptsDisabled:
        raise ValueError("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        raise ValueError("No transcript available for this video.")
    except Exception as exc:
        raise ValueError(f"Could not fetch transcript: {exc}")


def load_or_build_index(transcript: str, video_id: str) -> None:
    """
    If the video is not yet cached, chunk + embed + persist it.
    If it is cached, skip processing entirely.
    """
    if vector_store.is_video_cached(video_id):
        print("Cache hit — loading existing index for this video.")
        return

    print("No cache found — building index, this may take a moment...")
    # Larger chunks = fewer total embedding calls = less chance of hitting the 100 req/min free-tier limit
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    chunks = splitter.split_documents([Document(page_content=transcript)])
    try:
        vector_store.add_chunks(chunks, video_id)
    except RuntimeError as exc:
        print(f"DB write failed ({exc}). Resetting database and retrying...")
        vector_store.reset_db()
        vector_store.add_chunks(chunks, video_id)
    print("Index built and saved.")


def generate_answer(question: str, context_chunks: list[Document]) -> str:
    """
    Send the retrieved context and user question to the configured model.
    Gemma models don't support system_instruction, so the instruction is
    prepended directly into the prompt for those models.
    """
    context = "\n\n".join(doc.page_content for doc in context_chunks)

    system_instruction = (
        "You are a strict Q&A assistant. Answer questions using ONLY the transcript "
        "excerpts provided below. Do not use any external knowledge. "
        "If the answer cannot be found in the excerpts, respond exactly with: "
        "'I cannot answer this based on the provided video context.'"
    )

    if _SUPPORTS_SYSTEM_INSTRUCTION:
        prompt = f"Transcript excerpts:\n{context}\n\nQuestion: {question}"
        config = types.GenerateContentConfig(system_instruction=system_instruction)
    else:
        prompt = f"{system_instruction}\n\nTranscript excerpts:\n{context}\n\nQuestion: {question}"
        config = types.GenerateContentConfig()

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=config,
    )
    return response.text.strip()


def answer_question(youtube_url: str, question: str) -> str:
    """
    End-to-end pipeline: URL → transcript → persistent index → retrieval → answer.
    Returns a plain-string answer or a user-friendly error message.
    """
    video_id = extract_video_id(youtube_url)
    if not video_id:
        return "Error: The provided URL does not appear to be a valid YouTube link."

    try:
        transcript = fetch_transcript(video_id)
    except ValueError as exc:
        return f"Error: {exc}"

    load_or_build_index(transcript, video_id)

    try:
        chunks = vector_store.retrieve_chunks(question, video_id)
    except RuntimeError as exc:
        return f"Error retrieving context: {exc}"

    return generate_answer(question, chunks)


if __name__ == "__main__":
    print("=== YouTube Video Q&A ===\n")

    while True:
        youtube_url = input("Enter a YouTube URL: ").strip()
        video_id = extract_video_id(youtube_url)
        if not video_id:
            print("Error: Not a valid YouTube URL. Please try again.\n")
            continue

        print("Fetching transcript, please wait...")
        try:
            transcript = fetch_transcript(video_id)
        except ValueError as exc:
            print(f"Error: {exc}\n")
            continue

        load_or_build_index(transcript, video_id)

        print("\n┌─────────────────────────────────────────────┐")
        print("│  Commands: 'new' = change video  |  'exit' = quit  │")
        print("└─────────────────────────────────────────────┘\n")

        while True:
            question = input("Your question (or 'new' / 'exit'): ").strip()
            if not question:
                continue
            if question.lower() == "exit":
                print("Goodbye!")
                raise SystemExit
            if question.lower() == "new":
                print()
                break
            try:
                chunks = vector_store.retrieve_chunks(question, video_id)
            except RuntimeError as exc:
                print(f"Error: {exc}\n")
                continue
            answer = generate_answer(question, chunks)
            print(f"\nAnswer: {answer}\n")
