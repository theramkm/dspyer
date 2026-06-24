"""
dspyer quickstart: self-correcting, schema-validated LLM output (real model).

A node must answer a question AND cite a source. If the model returns an answer
without a citation, the Pydantic validator rejects it, dspyer generates corrective
feedback, and re-queries the model until it conforms (or hits `max_retries`).

Run (choose your model provider):
    # 1. Local Ollama (zero-setup, no API key needed):
    export DSPYER_MODEL="ollama_chat/llama3.2"

    # 2. Or Google Gemini:
    export DSPYER_MODEL="gemini/gemini-2.0-flash" GEMINI_API_KEY="your-key"

    # 3. Or OpenAI:
    export DSPYER_MODEL="openai/gpt-4o-mini" OPENAI_API_KEY="your-key"

    python examples/quickstart.py

No API key? See examples/run_rag_verifier.py: it runs the same idea fully offline.
"""

import os

import dspy
from pydantic import BaseModel, Field, field_validator

from dspyer.compiler import AgentTranspiler
from dspyer.graph import Graph, StatefulNode


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
    model = os.environ.get("DSPYER_MODEL")
    if not model:
        raise SystemExit(
            "Error: DSPYER_MODEL environment variable is not set.\n\n"
            "Pick any provider — dspyer is provider-agnostic (uses DSPy/LiteLLM under the hood):\n"
            "  # 1. Local Ollama (zero-setup, no API key needed, no signup):\n"
            '  export DSPYER_MODEL="ollama_chat/llama3.2"\n'
            "  python examples/quickstart.py\n\n"
            "  # 2. Google Gemini:\n"
            '  export DSPYER_MODEL="gemini/gemini-2.0-flash" GEMINI_API_KEY="..."\n'
            "  python examples/quickstart.py\n\n"
            "  # 3. Anthropic Claude:\n"
            '  export DSPYER_MODEL="anthropic/claude-3-5-sonnet-latest" ANTHROPIC_API_KEY="..."\n'
            "  python examples/quickstart.py\n\n"
            "  # 4. OpenAI GPT:\n"
            '  export DSPYER_MODEL="openai/gpt-4o-mini" OPENAI_API_KEY="..."\n'
            "  python examples/quickstart.py\n\n"
            "For a completely offline test run (no credentials or setup at all), try:\n"
            "  python examples/run_rag_verifier.py"
        )

    dspy.configure(lm=dspy.LM(model))

    result = program(query="What open-source license is the Linux kernel released under?")

    print("Answer:   ", result.answer)
    print("Citations:", result.citations)
    print("Self-correction loops:", result["_metadata"]["refinement_steps_taken"])


if __name__ == "__main__":
    main()
