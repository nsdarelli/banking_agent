from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class OrchestratorState(TypedDict):
    """
    State for the orchestrator.
    """
    user_query: str
    banking_result: str | None
    iteration: int
    final_answer: str | None


class BankingAgentState(TypedDict):
    """
    State for the Banking Agent.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    user_query: str
    sql_result: dict | None
    kb_result: list[dict] | None
    iteration: int
    final_result: str | None