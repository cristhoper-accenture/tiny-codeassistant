from agents.base import BaseAgent
from agents.general import GeneralAgent
from agents.coder import CoderAgent
from agents.lint import LintAgent
from agents.planner import PlannerAgent
from agents.breakdown import BreakdownAgent
from agents.tester import TesterAgent
from agents.orchestrator import OrchestratorAgent

REGISTRY: dict[str, type[BaseAgent]] = {
    "orchestrator": OrchestratorAgent,
    "general": GeneralAgent,
    "coder": CoderAgent,
    "lint": LintAgent,
    "planner": PlannerAgent,
    "breakdown": BreakdownAgent,
    "tester": TesterAgent,
}

DEFAULT_AGENT = "orchestrator"

__all__ = ["BaseAgent", "GeneralAgent", "CoderAgent", "LintAgent", "PlannerAgent", "BreakdownAgent", "TesterAgent", "OrchestratorAgent", "REGISTRY", "DEFAULT_AGENT"]
