import argparse
import os
import sys
from pydantic import BaseModel, Field, field_validator

# Allow direct script execution from subdirectories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dspy
from dspy_transpiler.graph import Graph, StatefulNode
from dspy_transpiler.compiler import AgentTranspiler

# 1. Describe the contract you want the LLM to honor
class Query(BaseModel):
    query: str

class RAGResponse(BaseModel):
    answer: str = Field(description="Answer referencing the sources")
    citations: list[str] = Field(description="Sources cited, e.g. ['doc_1']")

    @field_validator("citations")
    @classmethod
    def must_cite(cls, v):
        if not v:
            raise ValueError("Answer must cite at least one source.")
        return v

def main():
    parser = argparse.ArgumentParser(
        description="Run a real-model self-correcting quickstart demonstration."
    )
    parser.add_argument(
        "--provider",
        default=os.environ.get("DSPYER_PROVIDER", "openai"),
        help="Model provider (e.g. google, anthropic, openai, ollama).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("DSPYER_MODEL", "gpt-4o-mini"),
        help="Model name (e.g. gemini-1.5-flash, claude-3-5-sonnet, gpt-4o-mini).",
    )
    args = parser.parse_args()

    print("=========================================================================")
    print("⚡ dspyer: Real-Model Self-Correction Quickstart")
    print("=========================================================================")

    # Enforce API Keys checking
    has_keys = any(
        os.environ.get(k)
        for k in ["OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"]
    )
    if not has_keys:
        print(
            "Error: No model API keys detected in your environment variables.\n"
            "Please configure one of: OPENAI_API_KEY, GEMINI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY\n"
            "before running this real-model quickstart.",
            file=sys.stderr
        )
        sys.exit(1)

    print(f"[*] Initializing model backend: {args.provider}/{args.model}...")
    try:
        lm = dspy.LM(f"{args.provider}/{args.model}")
        dspy.configure(lm=lm)
    except Exception as e:
        print(f"Error configuring DSPy model: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Turn it into an optimizable, self-correcting node
    print("[*] Compiling self-correcting node workflow...")
    node = StatefulNode(
        "Synthesizer", Query, RAGResponse,
        instructions="Answer the query and cite sources. If you don't know, cite the placeholder source [doc_unknown].",
        max_retries=3,
    )
    graph = Graph()
    graph.add_node(node)
    graph.set_entry_point("Synthesizer")
    program = AgentTranspiler.compile(graph)

    # 3. Query the real model!
    query_text = "What license is dspyer under? (Cite the doc_license source if unknown)"
    print(f"[*] Querying real model with: '{query_text}'")
    try:
        r = program(query=query_text)
        print("\n[+] Execution completed successfully!")
        print("Answer:   ", r.answer)
        print("Citations:", r.citations)
        print("Self-correction loops run:", r["_metadata"]["refinement_steps_taken"])
    except Exception as e:
        print(f"\n[-] Execution failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
