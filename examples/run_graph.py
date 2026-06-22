import argparse
import os
import sys

# Allow direct script execution from subdirectories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dspy
from pydantic import BaseModel, Field

from dspy_transpiler.compiler import AgentTranspiler
from dspy_transpiler.graph import Graph, StatefulNode


# Define Step Schemas
class ExtractionInput(BaseModel):
    raw_text: str = Field(description="The source text to analyze")


class ExtractionOutput(BaseModel):
    user_name: str = Field(description="Name of the user extracted from text")
    rough_query: str = Field(description="User query context")


class ClassificationInput(BaseModel):
    rough_query: str = Field(description="Extracted rough query")


class ClassificationOutput(BaseModel):
    classified_intent: str = Field(description="Category intent: Support, Sales, Info")


class QuickstartMockLM(dspy.LM):
    def __init__(self):
        super().__init__(model="mock-quickstart-model")

    def forward(self, prompt=None, messages=None, **kwargs):
        prompt_str = str(prompt or messages)

        # Identify which node/stage is executing based on schema fields in prompt/messages
        if "user_name" in prompt_str or "raw_text" in prompt_str:
            # EntityExtractor node
            content = '{"user_name": "Alice", "rough_query": "buy a subscription"}'
        elif "classified_intent" in prompt_str or "rough_query" in prompt_str:
            # IntentClassifier node
            content = '{"classified_intent": "Sales"}'
        else:
            content = "{}"

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
                self.model = "mock-quickstart-model"
                self.usage = {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}

        return MockResult(content)


def main():
    parser = argparse.ArgumentParser(
        description="Run a compiled dspyer workflow using any user-specified model backend."
    )
    parser.add_argument(
        "--provider",
        default=os.environ.get("DSPYER_PROVIDER"),
        help="Model provider (e.g. google, anthropic, openai, ollama). Read from DSPYER_PROVIDER environment variable if not specified.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("DSPYER_MODEL"),
        help="Model name (e.g. gemini-3.5-flash, claude-4.8, gpt-5.5, llama3). Read from DSPYER_MODEL environment variable if not specified.",
    )
    parser.add_argument(
        "--text",
        default="Hello, my name is Alice. I would like to buy a subscription.",
        help="Text to run the extraction and classification workflow on.",
    )

    args = parser.parse_args()

    # Check if we should fallback to Mock LM (10-second zero-config quickstart)
    has_keys = any(
        os.environ.get(k)
        for k in ["OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"]
    )

    if not args.provider or not args.model or not has_keys:
        print(
            "[*] Automatically initializing zero-config QuickstartMockLM (no API keys detected or provider/model unspecified)..."
        )
        lm = QuickstartMockLM()
        dspy.configure(lm=lm)
    else:
        print(f"[*] Initializing model backend: {args.provider}/{args.model}...")
        try:
            lm = dspy.LM(f"{args.provider}/{args.model}")
            dspy.configure(lm=lm)
        except Exception as e:
            print(f"Error configuring DSPy model: {e}", file=sys.stderr)
            sys.exit(1)

    # 1. Declare Graph Nodes
    node_extraction = StatefulNode(
        name="EntityExtractor",
        input_model=ExtractionInput,
        output_model=ExtractionOutput,
        instructions="Extract the user's name and their query context from raw text.",
    )

    node_classification = StatefulNode(
        name="IntentClassifier",
        input_model=ClassificationInput,
        output_model=ClassificationOutput,
        instructions="Classify the intent of the rough query into: Support, Sales, or Info.",
    )

    # 2. Build Graph Topology
    graph = Graph()
    graph.add_node(node_extraction)
    graph.add_node(node_classification)
    graph.set_entry_point("EntityExtractor")
    graph.add_edge("EntityExtractor", "IntentClassifier")

    # 3. Transpile Graph into a Declarative DSPy Program
    print("[*] Transpiling agent graph into declarative DSPy Module...")
    program = AgentTranspiler.compile(graph)

    # 4. Execute the Program
    print(f"[*] Running agent program on input text: {repr(args.text)}")
    try:
        result = program(raw_text=args.text)
        print("\n[+] Execution completed successfully!")
        print("Final State Output:")
        import pprint

        pprint.pprint(result)
    except Exception as run_err:
        print(f"\n[-] Execution failed: {run_err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
