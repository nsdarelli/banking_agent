import logging
from pathlib import Path
import time

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from banking_agent import run_banking_agent
from agentstates import OrchestratorState

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "app.log"

MAX_ITERATIONS = 3
logger = logging.getLogger(__name__)

# Orchestrator Node
def orchestrator_node(state: OrchestratorState) -> dict:
    """
    Control the high-level workflow. Assign tasks to the corresponding agents and collect results.
    """

    iteration = state["iteration"] + 1
    logger.info("Orchestrator iteration: %s", iteration)

    # Safety limit for the parent workflow.
    if iteration > MAX_ITERATIONS:
        logger.warning("Maximum orchestrator iterations reached.")

        return {
            "iteration": iteration,
            "final_answer": (state.get("banking_result") or "Unable to complete the request."),
        }

    # Banking Agent has already completed.
    if state.get("banking_result"):
        logger.info("Banking Agent result available. Moving to final response.")

        return {
            "iteration": iteration,
            "final_answer": state["banking_result"],
        }

    # Nothing to finalize yet.
    return {
        "iteration": iteration,
    }

# Route After Orchestrator
def route_agent(state: OrchestratorState) -> str:
    """
    Decide whether to call the Banking Agent or finish the workflow.
    """
    if state.get("final_answer"):
        return "final"

    if state.get("banking_result"):
        return "final"

    return "banking_agent"

# Banking Agent Node
def banking_agent_node(state: OrchestratorState) -> dict:
    """
    Execute the Banking Agent. The orchestrator only passes the user query.
    The Banking Agent internally decides whether it needs:
        - SQL
        - Knowledge Base
        - Both
        - No tool
    """
    logger.info("Calling Banking Agent.")

    try:
        result = run_banking_agent(state["user_query"])
        logger.info("Banking Agent completed successfully.")

        return {
            "banking_result": result,
        }

    except Exception:
        logger.exception("Banking Agent execution failed.")

        return {
            "banking_result": None,
            "final_answer": ("Was unable to process the request."),
        }

# Final Node
def final_node(state: OrchestratorState) -> dict:
    """
    Return the final response. The orchestrator does not generate a new response. It returns the result produced by the Banking Agent.
    """

    final_answer = (state.get("final_answer") or state.get("banking_result") or "I could not generate an answer.")
    logger.info("Orchestration completed.")

    return {
        "final_answer": final_answer,
    }

# Build Orchestrator Graph
def build_orchestrator():
    """
    Build the parent LangGraph.
    """
    workflow = StateGraph(OrchestratorState)

    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("banking_agent", banking_agent_node)
    workflow.add_node("final", final_node)

    workflow.set_entry_point("orchestrator")

    workflow.add_conditional_edges("orchestrator", route_agent,
        {
            "banking_agent": "banking_agent",
            "final": "final",
        },
    )

    workflow.add_edge("banking_agent", "orchestrator")
    workflow.add_edge("final", END)

    return workflow.compile()

orchestrator = build_orchestrator()

def run_orchestrator(user_query: str) -> str:
    """
    Run the complete orchestration workflow.
    """
    logger.info("Starting orchestration: %s", user_query)

    initial_state: OrchestratorState = {
        "user_query": user_query,
        "banking_result": None,
        "iteration": 0,
        "final_answer": None,
    }
    #Run the orchestrator with retry logic
    for attempt in range(3):
        try:
            result = orchestrator.invoke(initial_state)
            final_answer = result.get("final_answer")
            logger.info("Orchestrator returned final response.")

            return final_answer or ("Could not generate an answer.")

        except Exception:
            logger.exception("Orchestrator execution failed on attempt %s.", attempt + 1)

            if attempt < 2:
                time.sleep(2 ** attempt)  # Exponential backoff
                logger.info("Retrying orchestrator execution...")

    #result = orchestrator.invoke(initial_state)
    #final_answer = result.get("final_answer")

    logger.info("Orchestrator returned final response.")

    return final_answer or ("Could not generate an answer.")