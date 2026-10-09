"""Mirror sqlpilot's spans into agentlens (run / agent_step / llm / tool) for step, token and cost accounting.

    pip install agentlens   # or PYTHONPATH=path/to/agentlens/src
    python examples/agentlens_tracing.py
"""

from agentlens.exporters import JSONLExporter
from agentlens.tracer import configure

from sqlpilot.agent import AgentConfig, SQLAgent
from sqlpilot.config import demo_db_path
from sqlpilot.db import SQLiteDatabase
from sqlpilot.llm import MockLLM
from sqlpilot.tracing import Tracing

lens = configure("sqlpilot", exporters=[JSONLExporter("sqlpilot-traces.jsonl")])
agent = SQLAgent(SQLiteDatabase(demo_db_path()), MockLLM(), config=AgentConfig(), tracing=Tracing(lens))
result = agent.ask("How many genres are there?")
print(result.answer)
print("spans written to sqlpilot-traces.jsonl")
