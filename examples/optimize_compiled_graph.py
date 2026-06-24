import json
import os
import sys
from typing import Any, TypedDict

# Allow direct script execution from subdirectories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import importlib.util

if importlib.util.find_spec("langgraph") is None:
    print(
        "Error: This example requires the 'langgraph' library.\n"
        "Please install it using: pip install 'dspyer[langgraph]'",
        file=sys.stderr,
    )
    sys.exit(1)

import dspy
from dspy.teleprompt import BootstrapFewShot
from langgraph.graph import END, START, StateGraph

from dspyer import AgentTranspiler, from_langgraph


# 1. Define State Dict for LangGraph
class AgentState(TypedDict):
    input_text: str
    cleaned_text: str
    sentiment: str


# 2. Define Node Functions (docstrings act as default prompt instructions)
def clean_node(state: AgentState):
    """Normalize whitespace, remove special characters, and lowercase the text."""
    _ = dspy  # Mark as LLM node for auto-scaffolding analysis
    text = state.get("input_text", "")
    return {
        "input_text": text,
        "cleaned_text": text.strip().lower(),
        "sentiment": "unknown",
    }


def analyze_node(state: AgentState):
    """Analyze the cleaned_text and classify sentiment polarity as: positive, negative, or neutral."""
    _ = dspy  # Mark as LLM node for auto-scaffolding analysis
    cleaned = state.get("cleaned_text", "")
    sentiment_val = "positive" if "happy" in cleaned or "great" in cleaned else "neutral"
    return {
        "input_text": state.get("input_text", ""),
        "cleaned_text": cleaned,
        "sentiment": sentiment_val,
    }


# 3. Create a local mock LLM client to support few-shot bootsrapping
class OptimizationMockLM(dspy.LM):
    def __init__(self):
        super().__init__(model="mock-optimization-model")

    def forward(self, prompt=None, messages=None, **kwargs):
        prompt_str = str(prompt or messages)

        # Basic sentiment simulations
        if "i am very happy" in prompt_str:
            content = '{"sentiment": "positive", "cleaned_text": "i am very happy", "input_text": "I am very happy."}'
        elif "this product is great" in prompt_str:
            content = '{"sentiment": "positive", "cleaned_text": "this product is great", "input_text": "This product is great!"}'
        else:
            content = '{"sentiment": "neutral", "cleaned_text": "hello", "input_text": "hello"}'

        # If use_cot is True, signature expects a rationale field
        if "Step-by-step reasoning plan" in prompt_str or "rationale" in prompt_str.lower():
            # Inject rationale in mock response
            content_dict = json.loads(content)
            content_dict["rationale"] = "The input text contains positive keywords."
            content = json.dumps(content_dict)

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
                self.model = "mock-optimization-model"
                self.usage = {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}

        return MockResult(content)


def main():
    print("[*] Configuring local OptimizationMockLM simulator...")
    lm = OptimizationMockLM()
    dspy.configure(lm=lm)

    # 4. Construct LangGraph StateGraph
    print("[*] Constructing LangGraph StateGraph...")
    builder = StateGraph(AgentState)
    builder.add_node("Cleaner", clean_node)
    builder.add_node("Analyzer", analyze_node)

    builder.add_edge(START, "Cleaner")
    builder.add_edge("Cleaner", "Analyzer")
    builder.add_edge("Analyzer", END)

    # Configure per-node settings (custom max_retries and opt-in Chain-of-Thought)
    node_configs: dict[str, dict[str, Any]] = {
        "Cleaner": {"max_retries": 1},
        "Analyzer": {
            "use_cot": True,
            "max_retries": 3,
            "refine_instructions": "Check sentiment carefully",
        },
    }

    # 5. Convert dynamically to a dspyer Graph
    print("[*] Converting LangGraph to dspyer Graph topology...")
    dspyer_graph = from_langgraph(builder, node_configs=node_configs)

    # 6. Compile to declarative DSPy Module
    print("[*] Compiling graph into declarative DSPy Module...")
    program = AgentTranspiler.compile(dspyer_graph)

    # 7. Define Training Dataset for prompt optimization
    trainset = [
        dspy.Example(
            input_text="I am very happy.", cleaned_text="i am very happy", sentiment="positive"
        ).with_inputs("input_text"),
        dspy.Example(
            input_text="This product is great!",
            cleaned_text="this product is great",
            sentiment="positive",
        ).with_inputs("input_text"),
    ]

    # 8. Define a simple evaluation metric
    def exact_match_metric(example, pred, trace=None) -> bool:
        return example.sentiment.strip().lower() == pred.sentiment.strip().lower()

    # 9. Run the prompt optimizer
    print("[*] Running BootstrapFewShot optimizer...")
    optimizer = BootstrapFewShot(metric=exact_match_metric, max_bootstrapped_demos=2)
    optimized_program = optimizer.compile(program, trainset=trainset)

    print("\n[+] Optimization completed successfully!")

    # Show initial custom instructions
    print(f"  Cleaner Instructions: '{optimized_program.predictor_Cleaner.signature.instructions}'")
    print(
        f"  Analyzer Instructions (with CoT): '{optimized_program.predictor_Analyzer.signature.instructions}'"
    )

    # Manually tweak instructions in optimized program to simulate prompt tuning
    optimized_program.predictor_Analyzer.signature.instructions = (
        "Optimized sentiment instructions."
    )

    # 10. Save the optimized configuration
    config_path = "optimized_agent_config.json"
    print(f"\n[*] Saving optimized prompt configurations to '{config_path}'...")
    optimized_program.save_config(config_path)

    # 11. Rehydrate in a fresh program
    print("\n[*] Rehydrating optimized configurations in a fresh program instance...")
    fresh_graph = from_langgraph(builder, node_configs=node_configs)
    fresh_program = AgentTranspiler.compile(fresh_graph)

    print(
        f"  Before Load - Analyzer Instructions: '{fresh_program.predictor_Analyzer.signature.instructions}'"
    )
    fresh_program.load_config(config_path)
    print(
        f"  After Load  - Analyzer Instructions: '{fresh_program.predictor_Analyzer.signature.instructions}'"
    )

    # Cleanup temporary configuration file
    if os.path.exists(config_path):
        os.remove(config_path)
        print(f"\n[+] Cleaned up '{config_path}'.")


if __name__ == "__main__":
    main()
