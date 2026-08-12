# Banking Agent

A banking agent using FastAPI, LangGraph, and LangChain with database and document search tools.

## Features
- Chat endpoint for banking questions
- SQL execution tool with read-only query validation
- Document ingestion and search over PDF/DOCX/TXT files
- Orchestrator workflow that routes user queries through a banking agent
- Banking Agent can use reasoning on tools and perform actions. 

## Setup
1. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your settings:
   ```text
   MYSQL_HOST=localhost
   MYSQL_PORT=3306
   MYSQL_USER=<your_user>
   MYSQL_PASSWORD=<your_password>
   MYSQL_DATABASE=banking_db
   GOOGLE_API_KEY=<your_google_api_key>
   MODEL_NAME=gemini-3.6-flash
   EMBEDDING_MODEL=models/gemini-embedding-2
   ```

## Run
```powershell
& .\.venv\Scripts\uvicorn.exe main:app --reload
```

## API
- `GET /health` - health check
- `POST /chat` - send a JSON body with `query` to get a banking answer
- `POST /documents` - upload `.pdf`, `.docx`, or `.txt` document to index it into the knowledge base

## Notes
- SQL tool only supports read-only `SELECT` or `WITH` statements.
- Uploaded documents are stored in `data/documents`.
- Chroma vector store persists in `data/chroma`.

## Workflow

                  Orchestrator
                       │
                       ▼
                Banking Agent
                       │
              ┌────────┴────────┐
              ▼                 ▼
         execute_sql      search_knowledge_base
              │                 │
              └────────┬────────┘
                       ▼
                  Banking Agent
                       │
                enough information?
                  /           \
                No             Yes
                │               │
                ▼               ▼
             another          Final
              tool            answer