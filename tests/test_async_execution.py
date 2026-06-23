import dspy
import pytest
from pydantic import BaseModel, field_validator

from dspy_transpiler import AgentTranspiler, Graph, StatefulNode, dspyer_node
from dspy_transpiler.utils import BaseStorageAdapter, get_storage_adapter, set_storage_adapter


class DummyInput(BaseModel):
    query: str


class DummyOutput(BaseModel):
    response: str
    score: int

    @field_validator("score")
    @classmethod
    def score_must_be_positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Score must be positive.")
        return v


class MockStorageAdapter(BaseStorageAdapter):
    def __init__(self):
        self.lines = []

    def append_line(self, target: str, line: str) -> None:
        self.lines.append((target, line))

    async def append_line_async(self, target: str, line: str) -> None:
        self.lines.append((target, line))


@pytest.mark.asyncio
async def test_pluggable_storage_adapter():
    mock_adapter = MockStorageAdapter()
    original_adapter = get_storage_adapter()
    try:
        set_storage_adapter(mock_adapter)
        from dspy_transpiler.utils import log_self_correction_example_async

        await log_self_correction_example_async(
            "dummy_path.jsonl",
            {"input": "test"},
            {"output": "res"},
        )
        assert len(mock_adapter.lines) == 1
        assert mock_adapter.lines[0][0] == "dummy_path.jsonl"
        assert '"input": "test"' in mock_adapter.lines[0][1]
    finally:
        set_storage_adapter(original_adapter)


@pytest.mark.asyncio
async def test_dspyer_node_decorator():
    # 1. Test decorator metadata attachments
    @dspyer_node(input_model=DummyInput, output_model=DummyOutput, is_llm=False)
    def my_node(state):
        return {"response": state["query"] + " processed", "score": 100}

    assert my_node._dspyer_input_model == DummyInput
    assert my_node._dspyer_output_model == DummyOutput
    assert my_node._dspyer_is_llm is False

    # 2. Test LangGraph conversion using decorator metadata (bypassing AST parsing)
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    class SimpleState(TypedDict):
        query: str
        response: str
        score: int

    builder = StateGraph(SimpleState)
    builder.add_node("Processor", my_node)
    builder.add_edge(START, "Processor")
    builder.add_edge("Processor", END)

    from dspy_transpiler import from_langgraph

    dspyer_graph = from_langgraph(builder)

    assert "Processor" in dspyer_graph.nodes
    node = dspyer_graph.nodes["Processor"]
    assert node.is_passthrough is True
    assert node.callable == my_node
    assert node.input_model == DummyInput
    assert node.output_model == DummyOutput


class AsyncRAGMockLM(dspy.LM):
    def __init__(self):
        super().__init__(model="async-mock-lm")
        self.count = 0

    def forward(self, prompt=None, messages=None, **kwargs):
        self.count += 1
        prompt_str = str(prompt or messages)
        is_refinement = (
            "failed_output" in prompt_str
            or "feedback" in prompt_str.lower()
            or "error" in prompt_str.lower()
        )
        is_json = "JSON" in prompt_str or "{" in prompt_str

        if is_refinement:
            if is_json:
                res = '{"response": "success", "score": 95}'
            else:
                res = "[[ ## response ## ]]\nsuccess\n\n[[ ## score ## ]]\n95"
        else:
            if is_json:
                res = '{"response": "no score", "score": -1}'
            else:
                res = "[[ ## response ## ]]\nno score\n\n[[ ## score ## ]]\n-1"
        return RAGMockResult(res)


class RAGMockResult:
    def __init__(self, content):
        self.choices = [RAGMockChoice(content)]
        self.model = "async-mock-lm"
        self.usage = {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}


class RAGMockChoice:
    def __init__(self, content):
        self.message = RAGMockMessage(content)
        self.finish_reason = "stop"
        self.index = 0


class RAGMockMessage:
    def __init__(self, content):
        self.content = content
        self.role = "assistant"
        self.reasoning_content = None


@pytest.mark.asyncio
async def test_async_forward_and_stream():
    lm = AsyncRAGMockLM()
    dspy.configure(lm=lm, cache=False)

    # Node requires score > 0
    node = StatefulNode(
        name="Evaluator",
        input_model=DummyInput,
        output_model=DummyOutput,
        instructions="Generate evaluator response and score.",
        max_retries=2,
    )

    def custom_formatter(err):
        return f"Please fix validation failure: {str(err)}"

    graph = Graph()
    graph.add_node(node)
    graph.set_entry_point("Evaluator")

    program = AgentTranspiler.compile(
        graph,
        output_model=DummyOutput,
        error_formatter=custom_formatter,
    )

    # 1. Test async forward
    result = await program.aforward(query="Hello")
    assert isinstance(result, DummyOutput)
    assert result.response == "success"
    assert result.score == 95

    # Reset LM count for stream test
    lm.count = 0

    # 2. Test async stream
    events = []
    async for ev in program.astream(query="World"):
        events.append(ev)

    # Assert sequence of events
    assert len(events) > 0
    # First event should be node_start
    assert events[0]["event"] == "node_start"
    assert events[0]["node"] == "Evaluator"

    # Should contain a validation_error event
    val_errors = [e for e in events if e["event"] == "validation_error"]
    assert len(val_errors) == 1
    assert "score" in val_errors[0]["failed_fields"]
    assert "Please fix validation" in val_errors[0]["error"]

    # End event should be finished
    assert events[-1]["event"] == "finished"
    assert isinstance(events[-1]["prediction"], DummyOutput)
    assert events[-1]["prediction"].response == "success"
