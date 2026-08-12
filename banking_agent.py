import json
import logging
import os
import re
from pathlib import Path
import time
from typing import Any

import mysql.connector
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agentstates import BankingAgentState


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "app.log"
VECTOR_DIRECTORY = BASE_DIR / "data" / "chroma"

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "banking_db")

MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3.6-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2")

MAX_ROWS = 100
MAX_ITERATIONS = 5
TOP_K = 5

# Logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# Database connection
def get_db_connection():
    """Create a new MySQL database connection."""

    return mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        connection_timeout=10,
    )

# SQL Query Validation
def validate_sql_query(query: str) -> tuple[bool, str]:
    """Validate that the SQL query is read-only."""

    query = query.strip()
    if not query:
        return False, "SQL query cannot be empty."

    query_wts = query.rstrip(";").strip()
    if ";" in query_wts:
        return False, "Multiple SQL statements are not allowed."

    if not re.match(r"^(SELECT|WITH)\b", query_wts, re.IGNORECASE):
        return False, "Only SELECT or WITH queries are allowed."

    forbidden_keywords=["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE", "REPLACE", "RENAME", "GRANT", "REVOKE", "SET", "CALL", "LOAD", "OUTFILE", "INTO OUTFILE", "INTO DUMPFILE"]
    upper_query = query_wts.upper()

    for keyword in forbidden_keywords:
        if re.search(rf"\b{re.escape(keyword)}\b", upper_query):
            return False, (f"Forbidden SQL operation detected: {keyword}")
        
    return True, ""

# Database Schema
DATABASE_SCHEMA = """
DATABASE: banking_db

TABLE: customers

customer_id       VARCHAR(20) PRIMARY KEY
country           VARCHAR(50) NOT NULL
transaction_year  SMALLINT NOT NULL

TABLE: transactions

customer_id       VARCHAR(20) PRIMARY KEY
amount_debit      DECIMAL(15,2) NOT NULL
amount_credited   DECIMAL(15,2) NOT NULL
total_amount      DECIMAL(15,2) NOT NULL
currency          VARCHAR(10) NOT NULL

RELATIONSHIP:

customers.customer_id = transactions.customer_id

CUSTOMER DATA RULES:

- India customer IDs start with IND.
- United States customer IDs start with USA.
- United Kingdom customer IDs start with GBR.
- Germany customer IDs start with DEU.
- Singapore customer IDs start with SGP.

IMPORTANT:
- country is stored in customers.
- transaction_year is stored in customers.
- financial amounts are stored in transactions.
- currency is stored in transactions.
- Use JOIN when information from both tables is required.

When the user uses an abbreviation or common name such as:
- USA → United States
- UK → United Kingdom

use the actual value stored in the country column.
Do not assume the abbreviation is stored in the database.
"""

# SQL Tool
@tool
def execute_sql(query: str) -> dict[str, Any]:
    """
    Execute a read-only SQL query against the banking database. Use this tool for structured banking data questions.
    """
    logger.info("SQL tool called.")

    is_valid, error_message = validate_sql_query(query)
    if not is_valid:
        logger.warning("SQL validation failed: %s", error_message)

        return {
            "status": "error",
            "error": error_message,
        }

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute(query)
        rows = cursor.fetchmany(MAX_ROWS)

        columns = []
        if cursor.description:
            columns = [column[0] for column in cursor.description]

        logger.info("SQL executed successfully. rows=%s", len(rows))

        return {
            "status": "success",
            "row_count": len(rows),
            "columns": columns,
            "rows": rows,
        }

    except mysql.connector.Error as exc:
        logger.exception("Database error.")

        return {
            "status": "error",
            "error": f"Database error: {str(exc)}",
        }

    except Exception as exc:
        logger.exception("Unexpected SQL error.")

        return {
            "status": "error",
            "error": f"Unexpected error: {str(exc)}",
        }

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()

# Vector Store
embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

vector_store = Chroma(
    collection_name="banking_knowledge",
    persist_directory=str(VECTOR_DIRECTORY),
    embedding_function=embeddings,
)

# Document Ingestion
def ingest_document(file_path: str) -> int:
    """
    Load and index a PDF, DOCX or TXT document.
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension == ".pdf":
        loader = PyPDFLoader(str(path))

    elif extension == ".docx":
        loader = Docx2txtLoader(str(path))

    elif extension == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError("Only .pdf, .docx and .txt files are supported.")

    logger.info("Loading document: %s", path.name)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No text could be extracted from the document.")

    vector_store.add_documents(chunks)
    logger.info("Document indexed: %s | chunks=%s", path.name, len(chunks))

    return len(chunks)

# Knowledge Base Tool
@tool
def search_knowledge_base(query: str) -> dict[str, Any]:
    """
    Search uploaded PDF, DOCX and TXT documents. Use this tool when the answer requires information from the uploaded knowledge base.
    """
    logger.info("Knowledge base search called.")

    try:
        documents = vector_store.similarity_search(query, k=TOP_K)

        results = []
        for document in documents:
            results.append(
                {
                    "content": document.page_content,
                    "metadata": document.metadata,
                }
            )
        logger.info("KB search returned %s results.", len(results))

        return {
            "status": "success",
            "results": results,
        }

    except Exception as exc:
        logger.exception("Knowledge base search failed.")

        return {
            "status": "error",
            "error": str(exc),
        }



# Banking Agent Prompt
BANKING_AGENT_SYSTEM_PROMPT = f"""
You are a Banking Data and Knowledge Agent. You answer user questions using ONLY:
1. Structured banking data through execute_sql.
2. Available document knowledge from documents through search_knowledge_base.

DATABASE SCHEMA: Strictly use available (MySQL) Database schema only
{DATABASE_SCHEMA}

TOOLS:
execute_sql
- Call for structured database questions.
Example: SQL questions related
- customer information
- country
- transaction year
- debit
- credit
- total amount
- currency
- counts

search_knowledge_base
- Call for questions requiring information from available knowledge base from documents.
Example: Knowledge-base questions related
- policies
- procedures
- explanations
- rules
- information from uploaded documents
- document-specific facts

TOOL RULES:
1. Decide which tool is required based on the user's question.
2. If SQL information is required, use execute_sql.
3. If document information is required, use search_knowledge_base.
4. Use both tools ONLY when the answer requires facts from BOTH sources.
5. You may call tools multiple times if necessary.
6. After every tool result, determine whether another tool call is required.
7. Stop calling tools when sufficient information is available.
8. Answer only after obtaining the required information.
9. Never invent information.
10. If the required information is unavailable, clearly say so.

Examples:
1.Which customers had the highest debit in 2025, and does the uploaded banking policy specify any special monitoring requirement for such customers?
Required:
a. SQL → identify customers.
b. KB → retrieve policy requirement.
Do NOT call both tools for a question that can be answered completely from one source.

SQL RULES:
- Only use tables and columns in the schema.
- Only generate SELECT or WITH queries.
- Never modify the database.
- Use JOIN when required.
- Use aggregation functions when appropriate.
- Use GROUP BY for grouped results.
- Use ORDER BY and LIMIT for ranking questions.
- country and transaction_year belong to customers.
- Financial amounts and currency belong to transactions.

CURRENCY RULE:
Do not combine different currencies into a single monetary total.
For example, do not sum amounts across countries when the currencies are different. Group by country and/or currency when appropriate.

FINAL RESPONSE:
- Answer the user's original question.
- Be concise and clear.
- Include currency when discussing monetary values.
- Do not expose internal reasoning.
- Do not expose unnecessary SQL or retrieved chunks.
- If the information is not available strictly mention I don't have information.
"""

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

tools = [execute_sql, search_knowledge_base]
llm_with_tools = llm.bind_tools(tools)

# Agent Node
def banking_agent_node(state: BankingAgentState) -> dict[str, Any]:
    """
    Ask Gemini LLM whether another tool call is required or whether the answer can be finalized.
    """
    iteration = state["iteration"] + 1
    logger.info("Banking Agent iteration=%s", iteration)

    # Safety limit.
    if iteration > MAX_ITERATIONS:
        logger.warning("Banking Agent maximum iterations reached.")

        return {
            "iteration": iteration,
            "final_result": ("Could not complete the request within the allowed number of tool calls."),
        }

    messages: list[BaseMessage] = [
        SystemMessage(
            content=BANKING_AGENT_SYSTEM_PROMPT
        ),
        *state["messages"],
    ]

    #retry mechanism for LLM invocation with exponential backoff
    for attempt in range(3):
        try:
            response = llm_with_tools.invoke(messages)
            break
        except Exception as e:
            logger.warning("Attempt %d failed: %s", attempt + 1, e)
            if attempt < 2:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
    #response = llm_with_tools.invoke(messages)

    update: dict[str, Any] = {
        "messages": [response],
        "iteration": iteration,
    }

    # No tool call means llm has produced the final answer.
    if (isinstance(response, AIMessage) and not response.tool_calls):
        update["final_result"] = response.content
        logger.info("Banking Agent produced final answer.")

    else:
        logger.info("Banking Agent requested tool call.")

    return update

# Capture Tool Result
def capture_tool_result(state: BankingAgentState) -> dict[str, Any]:
    """
    Store the latest tool result in the agent state. The full tool result remains available through messages.
    """
    if not state["messages"]:
        return {}

    last_message = state["messages"][-1]
    tool_name = getattr(last_message, "name", None)
    if not tool_name:
        return {}

    content = getattr(last_message, "content", None)
    if isinstance(content, str):
        try:
            result = json.loads(content)

        except json.JSONDecodeError:
            result = {
                "status": "success",
                "result": content,
            }

    elif isinstance(content, dict):
        result = content

    else:
        result = {
            "status": "success",
            "result": str(content),
        }

    if tool_name == "execute_sql":
        logger.info("SQL result captured in BankingAgentState.")

        return {
            "sql_result": result,
        }

    if tool_name == "search_knowledge_base":
        logger.info("KB result captured in BankingAgentState."
        )

        return {
            "kb_result": result.get("results", []),
        }

    return {}

# Route After Agent
def tool_route(state: BankingAgentState) -> str:
    """
    Decide whether the Banking Agent should continue with tools or finish with final result.
    """
    if state.get("final_result"):
        return "final"

    if state["iteration"] >= MAX_ITERATIONS:
        return "final"

    return "tools"

# Build Banking Agent
tool_node = ToolNode(tools)

def build_banking_agent():
    sub_workflow = StateGraph(BankingAgentState)

    sub_workflow.add_node("agent", banking_agent_node)
    sub_workflow.add_node("tools", tool_node)
    #sub_workflow.add_node("capture_tool_result", capture_tool_result)

    sub_workflow.set_entry_point("agent")

    sub_workflow.add_conditional_edges("agent", tool_route,
        {
            "tools": "tools",
            "final": END,
        },
    )

    #sub_workflow.add_edge("tools", "capture_tool_result")
    #sub_workflow.add_edge("capture_tool_result", "agent")
    sub_workflow.add_edge("tools", "agent")

    return sub_workflow.compile()

banking_agent = build_banking_agent()

def run_banking_agent(user_query: str) -> str:
    """
    Run the Banking Agent. The agent internally decides whether to use:
        SQL
        KB
        SQL + KB
        No tool
    """
    logger.info("Starting Banking Agent.")

    initial_state: BankingAgentState = {
        "messages": [
            HumanMessage(
                content=user_query
            )
        ],
        "user_query": user_query,
        "sql_result": None,
        "kb_result": None,
        "iteration": 0,
        "final_result": None,
    }
    #retry mechanism for LLM invocation with exponential backoff
    for attempt in range(3):
        try:
            result = banking_agent.invoke(initial_state)
            break
        except Exception as e:
            logger.warning("Attempt %d failed: %s", attempt + 1, e)
            if attempt < 2:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
    #result = banking_agent.invoke(initial_state)

    final_result = result.get("final_result")
    if not final_result:
        logger.error("Banking Agent completed without final result.")

        return ("Could not generate an answer for the request.")
    logger.info("Banking Agent completed successfully.")

    return final_result