"""
dspyer quickstart — self-correcting, schema-validated LLM output (real model).

A node must answer a question AND cite a source. If the model returns an answer
without a citation, the Pydantic validator rejects it, dspyer generates corrective
feedback, and re-queries the model until it conforms (or hits `max_retries`).

Run:
    export OPENAI_API_KEY=sk-...        # or set DSPYER_MODEL to any DSPy-supported model
    python examples/quickstart.py

No API key? See examples/run_rag_verifier.py — it runs the same idea fully offline.
"""

import os

import dspy
from pydantic import BaseModel, Field, field_validator

from dspy_transpiler.compiler import AgentTranspiler
from dspy_transpiler.graph import Graph, StatefulNode


# 1. The contract you want the LLM to honor.
class Query(BaseModel):
    query: str


class RAGResponse(BaseModel):
    answer: str = Field(description="A concise answer that references its sources")
    citations: list[str] = Field(description="Source ids backing the answer, e.g. ['doc_1']")

    @field_validator("citations")
    @classmethod
    def must_cite(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("The answer must cite at least one source.")
        return value


# 2. Turn the contract into a self-correcting, optimizable node.
synthesizer = StatefulNode(
    name="Synthesizer",
    input_model=Query,
    output_model=RAGResponse,
    instructions="Answer the user's question and cite the document sources you used.",
    max_retries=3,
)

graph = Graph()
graph.add_node(synthesizer)
graph.set_entry_point("Synthesizer")
program = AgentTranspiler.compile(graph)  # a standard, optimizable dspy.Module


def main() -> None:
    model = os.environ.get("DSPYER_MODEL", "openai/gpt-4o-mini")
    if "OPENAI_API_KEY" not in os.environ and model.startswith("openai/"):
        raise SystemExit(
            "Set OPENAI_API_KEY (or DSPYER_MODEL to another provider) to run this demo.\n"
            "For an offline version, run: python examples/run_rag_verifier.py"
        )

    dspy.configure(lm=dspy.LM(model))

    result = program(query="What open-source license is the Linux kernel released under?")

    print("Answer:   ", result.answer)
    print("Citations:", result.citations)
    print("Self-correction loops:", result["_metadata"]["refinement_steps_taken"])


if __name__ == "__main__":
    main()
