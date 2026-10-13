"""sqlpilot: a multi-agent text-to-SQL analyst built on LangGraph."""

from sqlpilot.agent import AgentConfig, AgentResult, SQLAgent
from sqlpilot.db import connect
from sqlpilot.guard import GuardConfig, GuardResult, SQLGuard, check_sql

__version__ = "0.1.1"

__all__ = [
    "AgentConfig",
    "AgentResult",
    "GuardConfig",
    "GuardResult",
    "SQLAgent",
    "SQLGuard",
    "__version__",
    "check_sql",
    "connect",
]
