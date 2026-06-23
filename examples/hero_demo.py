import os
import sys
import time
from typing import TypedDict

# Allow direct script execution from subdirectories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dspy
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, field_validator

from dspy_transpiler import AgentTranspiler, from_langgraph
from dspy_transpiler.graph import StatefulNode


# 1. Define State for LangGraph
class State(TypedDict):
    query: str
    cleaned_query: str
    response: str
    citations: list[str]


# 2. Define Node Functions
def clean_query(state: State):
    """Deterministic node to normalize and strip the query."""
    q = state.get("query", "")
    return {
        "query": q,
        "cleaned_query": q.strip().lower(),
        "response": "",
        "citations": [],
    }


def log_completion(state: State):
    """Deterministic logging node (runs outside of LLM routing)."""
    citations = state.get("citations", [])
    print("\n[Postprocess] Native Python node 'log_completion' ran successfully!")
    print(f"              State Citation Count: {len(citations)}")
    return {}


# 3. Define narrow models for the reasoning node
class QueryInput(BaseModel):
    cleaned_query: str


class QueryOutput(BaseModel):
    response: str = Field(description="Answer to the user query")
    citations: list[str] = Field(description="Cited documents verifying the answer")

    @field_validator("citations")
    @classmethod
    def must_contain_citations(cls, v):
        if not v or len(v) == 0:
            raise ValueError("The answer must cite at least one document.")
        return v


# 4. Construct a mock LM that fails initial run and succeeds on retry
class HeroMockLM(dspy.LM):
    def __init__(self):
        super().__init__(model="hero-mock-model")

    def forward(self, prompt=None, messages=None, **kwargs):
        # Simulate standard LLM generation latency
        time.sleep(0.6)
        prompt_str = str(prompt or messages)

        # 1. Mock response for optimization dataset
        if "what is the license" in prompt_str.lower():
            content = (
                '{"response": "dspyer uses the Apache-2.0 license.", "citations": ["doc_license"]}'
            )
        elif "what is the version" in prompt_str.lower():
            content = '{"response": "dspyer is at version 0.3.0.", "citations": ["doc_ver"]}'
        # 2. Mock response for correction retry pass
        elif "failed_output" in prompt_str or "feedback" in prompt_str:
            content = '{"response": "Paris is the capital of France.", "citations": ["doc_geo"]}'
        else:
            # First pass fails Pydantic validation (empty citations list)
            content = '{"response": "Paris is the capital of France.", "citations": []}'

        class MockChoiceMessage:
            def __init__(self, content_str: str):
                self.content = content_str
                self.role = "assistant"
                self.reasoning_content = None

        class MockChoice:
            def __init__(self, content_str: str):
                self.message = MockChoiceMessage(content_str)
                self.finish_reason = "stop"
                self.index = 0

        class MockResult:
            def __init__(self, content_str: str):
                self.choices = [MockChoice(content_str)]
                self.model = "hero-mock-model"
                self.usage = {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}

        return MockResult(content)


def main():
    print("=========================================================================")
    print("⚡ dspyer: The 60-Second Hero Demo")
    print("=========================================================================")
    time.sleep(0.8)

    # Configure Mock LM
    lm = HeroMockLM()
    dspy.configure(lm=lm)

    # 1. Build a 3-node LangGraph StateGraph
    print("\n[*] Step 1: Defining a 3-node LangGraph workflow...")
    time.sleep(1.0)
    builder = StateGraph(State)
    builder.add_node("clean_query", clean_query)
    builder.add_node("ResponseGenerator", lambda state: state)  # Stub function for reasoning
    builder.add_node("log_completion", log_completion)

    builder.add_edge(START, "clean_query")
    builder.add_edge("clean_query", "ResponseGenerator")
    builder.add_edge("ResponseGenerator", "log_completion")
    builder.add_edge("log_completion", END)

    # 2. Map only the reasoning node explicitly
    node_mappings = {
        "ResponseGenerator": StatefulNode(
            name="ResponseGenerator",
            input_model=QueryInput,
            output_model=QueryOutput,
            instructions="Provide an answer with source citations.",
            max_retries=2,
        )
    }

    # 3. Scaffold and compile using dspyer
    print("\n[*] Step 2: Scaffolding graph topology and compiling to DSPy...")
    time.sleep(1.2)
    dspyer_graph = from_langgraph(builder, node_mappings=node_mappings)
    program = AgentTranspiler.compile(dspyer_graph)

    # Verify deterministic nodes remain untouched passthroughs
    print("\n[+] Verification:")
    time.sleep(0.5)
    print(
        f"    - Node 'clean_query': Native passthrough? {dspyer_graph.nodes['clean_query'].is_passthrough}"
    )
    time.sleep(0.5)
    print(
        f"    - Node 'log_completion': Native passthrough? {dspyer_graph.nodes['log_completion'].is_passthrough}"
    )
    time.sleep(0.5)
    print(
        f"    - Node 'ResponseGenerator': Compiled to DSPy? {hasattr(program, 'predictor_ResponseGenerator')}"
    )
    time.sleep(1.2)

    # 4. Run compilation execution with automatic self-correction retry
    print("\n[*] Step 3: Running transpiled program (fails validation on first pass)...")
    time.sleep(1.0)
    res = program(query="What is the capital of France?")

    time.sleep(0.8)
    print("\n[+] Self-Correction Execution Output:")
    print(f"    - Response: {res['response']}")
    print(f"    - Citations: {res['citations']}")
    print(f"    - Refinement attempts taken: {res['_metadata']['refinement_steps_taken']}")
    time.sleep(1.5)

    # 5. Optimize prompt using DSPy BootstrapFewShot
    print("\n[*] Step 4: Optimizing node prompts using BootstrapFewShot...")
    time.sleep(1.0)
    trainset = [
        dspy.Example(
            query="What is the license?",
            response="dspyer uses the Apache-2.0 license.",
            citations=["doc_license"],
        ).with_inputs("query"),
        dspy.Example(
            query="What is the version?",
            response="dspyer is at version 0.3.0.",
            citations=["doc_ver"],
        ).with_inputs("query"),
    ]

    def simple_metric(example, pred, trace=None) -> bool:
        return len(pred.citations) > 0

    from dspy.teleprompt import BootstrapFewShot

    optimizer = BootstrapFewShot(metric=simple_metric, max_bootstrapped_demos=2)
    optimized_program = optimizer.compile(program, trainset=trainset)

    time.sleep(1.0)
    print("\n[+] Prompt Optimization Completed!")
    time.sleep(0.5)
    print(
        f"    Original instruction: '{program.predictor_ResponseGenerator.signature.instructions}'"
    )
    time.sleep(0.5)
    print(
        f"    Optimized instruction: '{optimized_program.predictor_ResponseGenerator.signature.instructions}'"
    )
    time.sleep(0.5)
    print(
        "    Deterministic nodes (clean_query, log_completion) remained native, untouched Python."
    )
    time.sleep(0.8)
    print("=========================================================================\n")


if __name__ == "__main__":
    main()
