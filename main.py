import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from banking_agent import ingest_document
from orchestrator_agent import run_orchestrator


BASE_DIR = Path(__file__).resolve().parent
DOCUMENT_DIRECTORY = BASE_DIR / "data" / "documents"

DOCUMENT_DIRECTORY.mkdir(parents=True, exist_ok=True)

# Logging
logger = logging.getLogger(__name__)


app = FastAPI(title="Banking Agent API", version="1.0.0")

# Request Models
class ChatRequest(BaseModel):
    query: str


# Health Check
@app.get("/health")
def health_check():
    return {
        "status": "ok",
    }

@app.post("/chat")
def chat(request: ChatRequest):
    """
    Send a user question to the orchestrator.
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    logger.info("Chat request received from User.")

    try:
        answer = run_orchestrator(query)

        return {
            "answer": answer[0]["text"],
        }

    except Exception as e:
        logger.exception("Chat request failed.")
        raise HTTPException(status_code=500, detail="Unable to process the request.")

# Document Upload
@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF, DOCX or TXT document and index it into the knowledge base.
    """
    allowed_extensions = {".pdf", ".docx", ".txt"}
    filename = file.filename or ""
    extension = Path(filename).suffix.lower()
    if extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=("Only .pdf, .docx and .txt files are supported."))

    file_path = DOCUMENT_DIRECTORY / Path(filename).name

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        file_path.write_bytes(content)
        logger.info("Document uploaded: %s", filename)
        chunk_count = ingest_document(str(file_path))

        return {
            "status": "success",
            "filename": filename,
            "chunks_indexed": chunk_count,
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("Document processing failed: %s", filename)

        # Remove the file if ingestion failed.
        if file_path.exists():
            file_path.unlink()

        raise HTTPException(status_code=500, detail="Unable to process the document.")