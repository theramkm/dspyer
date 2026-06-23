import argparse
import concurrent.futures
import os
import sys
from typing import Any, Dict, List

# Allow direct script execution from subdirectories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dspy
from pydantic import BaseModel, Field

from dspyer.compiler import AgentTranspiler
from dspyer.graph import Graph, StatefulNode
from dspyer.state import ImmutableState

# =====================================================================
# 1. Define Step Schemas
# =====================================================================


class ParserInput(BaseModel):
    raw_text: str = Field(description="The raw text to parse")


class ParserOutput(BaseModel):
    items: List[str] = Field(description="List of extracted individual customer feedback queries")
    confidence: float = Field(description="Confidence score of extraction (0.0 to 1.0)")


class SentimentInput(BaseModel):
    items: List[str] = Field(description="List of queries to analyze sentiment for")


class SentimentOutput(BaseModel):
    sentiments: List[str] = Field(
        description="List of sentiment labels matching the order of items"
    )


class TagInput(BaseModel):
    items: List[str] = Field(description="List of queries to extract tags for")


class TagOutput(BaseModel):
    tags: List[str] = Field(description="List of tags matching the items")


# =====================================================================
# 2. Mock LM for deterministic run without requiring API keys
# =====================================================================


class LoopAndParallelMockLM(dspy.LM):
    """
    Simulates:
    1. First call to EntityExtractor: Returns low confidence (0.5) to trigger the loop.
    2. Second call to EntityExtractor: Returns high confidence (0.95) and items list.
    3. SentimentAnalyzer: Returns positive/negative/neutral list.
    4. TagExtractor: Returns tag categories.
    """

    def __init__(self):
        super().__init__(model="mock-loop-parallel")
        self.extractor_calls = 0

    def forward(
        self, prompt: str | None = None, messages: List[Dict[str, Any]] | None = None, **kwargs
    ):
        from dspyer.compiler import MockCompletionResult

        prompt_str = prompt or ""
        if messages:
            import json

            prompt_str += " " + json.dumps(messages)

        is_chat = "[[ ##" in prompt_str

        if "Parse raw text feedback into a list of individual queries" in prompt_str:
            if "failed_output" in prompt_str or "error_feedback" in prompt_str:
                if is_chat:
                    content = '[[ ## items ## ]]\n["Alice is happy with product", "Bob is frustrated with shipping"]\n\n[[ ## confidence ## ]]\n0.95\n\n[[ ## completed ## ]]'
                else:
                    content = '{"items": ["Alice is happy with product", "Bob is frustrated with shipping"], "confidence": 0.95}'
                return MockCompletionResult(content, self.model)

            # Predictor step iteration call
            self.extractor_calls += 1
            if self.extractor_calls == 1:
                # Return low confidence to trigger the confidence router loop back
                if is_chat:
                    content = '[[ ## items ## ]]\n["Alice is happy"]\n\n[[ ## confidence ## ]]\n0.5\n\n[[ ## completed ## ]]'
                else:
                    content = '{"items": ["Alice is happy"], "confidence": 0.5}'
            else:
                # Return high confidence on subsequent attempts
                if is_chat:
                    content = '[[ ## items ## ]]\n["Alice is happy with product", "Bob is frustrated with shipping"]\n\n[[ ## confidence ## ]]\n0.95\n\n[[ ## completed ## ]]'
                else:
                    content = '{"items": ["Alice is happy with product", "Bob is frustrated with shipping"], "confidence": 0.95}'
        elif "Confirm the extraction steps are complete" in prompt_str:
            if is_chat:
                content = "[[ ## status ## ]]\ndone\n\n[[ ## completed ## ]]"
            else:
                content = '{"status": "done"}'
        elif "Analyze sentiment of each input query" in prompt_str:
            if is_chat:
                content = (
                    '[[ ## sentiments ## ]]\n["Positive", "Negative"]\n\n[[ ## completed ## ]]'
                )
            else:
                content = '{"sentiments": ["Positive", "Negative"]}'
        elif "Extract relevant category tags for each input query" in prompt_str:
            if is_chat:
                content = '[[ ## tags ## ]]\n["product-feedback", "shipping-issue"]\n\n[[ ## completed ## ]]'
            else:
                content = '{"tags": ["product-feedback", "shipping-issue"]}'
        else:
            content = '{"error": "Unknown step"}'

        return MockCompletionResult(content, self.model)


# =====================================================================
# 3. Main Routing & Execution Loop
# =====================================================================


class DummyOutput(BaseModel):
    status: str = Field(default="done", description="Done status")


def main():
    parser = argparse.ArgumentParser(
        description="Run a looping and parallel conflict-resolution state machine."
    )
    parser.add_argument(
        "--use-real-lm",
        action="store_true",
        help="If set, uses a real configured LM. Otherwise, uses the LoopAndParallelMockLM.",
    )
    parser.add_argument(
        "--provider", default="google", help="Real LM provider (e.g. google, openai, anthropic)."
    )
    parser.add_argument("--model", default="gemini-2.5-flash", help="Real LM model name.")
    args = parser.parse_args()

    # 1. Configure the DSPy LM environment
    if args.use_real_lm:
        print(f"[*] Configuring real LM: {args.provider}/{args.model}...")
        try:
            lm = dspy.LM(f"{args.provider}/{args.model}")
            dspy.configure(lm=lm)
        except Exception as e:
            print(f"Error configuring real LM: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("[*] Configuring local LoopAndParallelMockLM simulator...")
        lm = LoopAndParallelMockLM()
        dspy.configure(lm=lm)

    # 2. Define confidence routing function
    def confidence_router(state: Dict[str, Any]) -> str:
        confidence = state.get("confidence", 0.0)
        print(f"  [Router] Evaluating parser confidence: {confidence:.2f}")
        if confidence < 0.8:
            print(
                "  [Router] Confidence too low (< 0.8). Routing back to EntityExtractor (looping)..."
            )
            return "loop"
        print("  [Router] Confidence acceptable (>= 0.8). Routing to EndNode (exiting loop)...")
        return "proceed"

    # 3. Construct Parser Graph
    print("[*] Constructing Parser Graph...")
    parser_graph = Graph()
    node_extractor = StatefulNode(
        name="EntityExtractor",
        input_model=ParserInput,
        output_model=ParserOutput,
        instructions="Parse raw text feedback into a list of individual queries and compute extraction confidence.",
    )
    node_end = StatefulNode(
        name="EndNode",
        input_model=ParserOutput,
        output_model=DummyOutput,
        instructions="Confirm the extraction steps are complete.",
    )

    parser_graph.add_node(node_extractor)
    parser_graph.add_node(node_end)
    parser_graph.set_entry_point("EntityExtractor")
    parser_graph.add_conditional_edges(
        "EntityExtractor", confidence_router, {"loop": "EntityExtractor", "proceed": "EndNode"}
    )

    # 4. Construct Sentiment Graph
    print("[*] Constructing Sentiment Analyzer Graph...")
    sentiment_graph = Graph()
    node_sentiment = StatefulNode(
        name="SentimentAnalyzer",
        input_model=SentimentInput,
        output_model=SentimentOutput,
        instructions="Analyze sentiment of each input query.",
    )
    sentiment_graph.add_node(node_sentiment)
    sentiment_graph.set_entry_point("SentimentAnalyzer")

    # 5. Construct Tag Graph
    print("[*] Constructing Tag Extractor Graph...")
    tag_graph = Graph()
    node_tag = StatefulNode(
        name="TagExtractor",
        input_model=TagInput,
        output_model=TagOutput,
        instructions="Extract relevant category tags for each input query.",
    )
    tag_graph.add_node(node_tag)
    tag_graph.set_entry_point("TagExtractor")

    # 6. Compile Graphs into DSPy Programs
    print("[*] Compiling graphs into optimizable declarative DSPy Modules...")
    parser_program = AgentTranspiler.compile(parser_graph)
    sentiment_program = AgentTranspiler.compile(sentiment_graph)
    tag_program = AgentTranspiler.compile(tag_graph)

    # 7. Execute Parser Graph (Looping feedback example)
    raw_text = "Alice is happy with product, Bob is frustrated with shipping"
    print(f"\n[+] Starting Parser workflow on input: {repr(raw_text)}")
    parser_result = parser_program(raw_text=raw_text)
    print(f"Parser completed in {parser_result['_metadata']['step_count']} steps.")
    print(f"Extracted Items: {parser_result.get('items')}")

    # 8. Execute Sentiment & Tag Graphs concurrently
    print("\n[+] Initiating parallel analysis branches concurrently...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_sentiment = executor.submit(sentiment_program, **parser_result)
        future_tag = executor.submit(tag_program, **parser_result)

        sentiment_res = future_sentiment.result()
        tag_res = future_tag.result()

    print(f"Sentiment Analysis Branch Output: {sentiment_res.get('sentiments')}")
    print(f"Tag Extraction Branch Output: {tag_res.get('tags')}")

    # 9. Merge the divergent states
    print("\n[+] Merging parallel outcomes using ImmutableState.merge (combine_lists policy)...")
    state_s = ImmutableState(sentiment_res)
    state_t = ImmutableState(tag_res)

    merged_state = state_s.merge(state_t, policy="combine_lists")
    final_dict = merged_state.to_dict()

    # Verification: assert loop actually executed (takes at least 3 steps)
    assert parser_result["_metadata"]["step_count"] >= 3, (
        f"Validation Error: Parser graph feedback loop did not execute. Step count: {parser_result['_metadata']['step_count']}"
    )
    print("\n[+] Verification successful: feedback loop executed correctly!")

    print("\n[+] Final Reconciled State Output:")
    import pprint

    pprint.pprint(final_dict)


if __name__ == "__main__":
    main()
