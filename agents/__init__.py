from agents.base import BaseAgent
from agents.general import GeneralAgent
from agents.coder import CoderAgent
from agents.orchestrator import OrchestratorAgent

REGISTRY: dict[str, type[BaseAgent]] = {
    "orchestrator": OrchestratorAgent,
    "general": GeneralAgent,
    "coder": CoderAgent,
}

DEFAULT_AGENT = "orchestrator"

__all__ = ["BaseAgent", "GeneralAgent", "CoderAgent", "OrchestratorAgent", "REGISTRY", "DEFAULT_AGENT"]
