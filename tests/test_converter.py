import pytest

pytest.importorskip("langgraph")

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from dspy_transpiler.compiler import AgentTranspiler
from dspy_transpiler.converter import from_langgraph
from dspy_transpiler.graph import StatefulNode


# Define state schema for testing
class TestStateDict(TypedDict):
    input_text: str
    processed_count: int
    decision: str


# Node functions for LangGraph
def node_a(state: TestStateDict):
    """Instructions for node A: increment count."""
    return {
        "input_text": state.get("input_text", ""),
        "processed_count": state.get("processed_count", 0) + 1,
        "decision": "proceed",
    }


def node_b(state: TestStateDict):
    """Instructions for node B: evaluate decision."""
    decision_val = "loop" if state.get("processed_count", 0) < 2 else "end"
    return {
        "input_text": state.get("input_text", ""),
        "processed_count": state.get("processed_count", 0),
        "decision": decision_val,
    }


def router(state: TestStateDict) -> str:
    """Decide next node."""
    return state.get("decision", "end")


def test_from_langgraph_conversion():
    # 1. Construct LangGraph StateGraph
    builder = StateGraph(TestStateDict)
    builder.add_node("NodeA", node_a)
    builder.add_node("NodeB", node_b)

    builder.add_edge(START, "NodeA")
    builder.add_edge("NodeA", "NodeB")
    builder.add_conditional_edges("NodeB", router, {"loop": "NodeA", "end": END})

    # 2. Convert to dspyer Graph
    dspyer_graph = from_langgraph(builder)

    # 3. Assertions on graph topology
    assert dspyer_graph.entry_point == "NodeA"
    assert "NodeA" in dspyer_graph.nodes
    assert "NodeB" in dspyer_graph.nodes

    # Verify auto-generated StatefulNodes
    node_a_wrapper = dspyer_graph.nodes["NodeA"]
    assert node_a_wrapper.name == "NodeA"
    assert node_a_wrapper.instructions == "Instructions for node A: increment count."
    assert issubclass(node_a_wrapper.input_model, BaseModel)
    assert issubclass(node_a_wrapper.output_model, BaseModel)

    # Verify static edges
    assert dspyer_graph.edges["NodeA"] == "NodeB"

    # Verify conditional edges
    assert "NodeB" in dspyer_graph.conditional_edges
    router_func, path_map = dspyer_graph.conditional_edges["NodeB"]
    assert router_func == router
    assert path_map == {"loop": "NodeA", "end": "__end__"}


def test_from_langgraph_custom_mappings():
    # 1. Define custom mapped node
    class CustomInput(BaseModel):
        input_text: str = Field(description="Custom field")

    class CustomOutput(BaseModel):
        processed_count: int = Field(description="Custom count")
        decision: str = Field(description="Custom decision")

    custom_node_a = StatefulNode(
        name="NodeA",
        input_model=CustomInput,
        output_model=CustomOutput,
        instructions="Overridden custom instructions.",
    )

    builder = StateGraph(TestStateDict)
    builder.add_node("NodeA", node_a)
    builder.add_edge(START, "NodeA")
    builder.add_edge("NodeA", END)

    # Convert with mappings
    dspyer_graph = from_langgraph(builder, node_mappings={"NodeA": custom_node_a})

    # Verify the override took place
    assert dspyer_graph.nodes["NodeA"] is custom_node_a
    assert dspyer_graph.nodes["NodeA"].instructions == "Overridden custom instructions."
    assert dspyer_graph.nodes["NodeA"].input_model is CustomInput


def test_from_langgraph_execution(monkeypatch):
    # 1. Build and compile LangGraph
    builder = StateGraph(TestStateDict)
    builder.add_node("NodeA", node_a)
    builder.add_node("NodeB", node_b)

    builder.add_edge(START, "NodeA")
    builder.add_edge("NodeA", "NodeB")
    builder.add_conditional_edges("NodeB", router, {"loop": "NodeA", "end": END})

    compiled_lg = builder.compile()

    # 2. Convert to dspyer graph and compile program
    dspyer_graph = from_langgraph(compiled_lg)
    program = AgentTranspiler.compile(dspyer_graph)

    # Mock predictions to replicate standard LangGraph functions
    class MockNodeAOutput:
        input_text = "test"
        processed_count = 1
        decision = "proceed"

    class MockNodeBOutput:
        input_text = "test"
        processed_count = 1
        decision = "loop"

    # We need to trace executions of predictors
    node_a_calls = 0
    node_b_calls = 0

    def mock_predict_node_a(**kwargs):
        nonlocal node_a_calls
        node_a_calls += 1
        # Replicates first step output, then second step output
        res = MockNodeAOutput()
        res.processed_count = node_a_calls
        return res

    def mock_predict_node_b(**kwargs):
        nonlocal node_b_calls
        node_b_calls += 1
        res = MockNodeBOutput()
        res.processed_count = kwargs.get("processed_count", 0)
        res.decision = "loop" if res.processed_count < 2 else "end"
        return res

    monkeypatch.setattr(program, "predictor_NodeA", mock_predict_node_a)
    monkeypatch.setattr(program, "predictor_NodeB", mock_predict_node_b)

    # Execute the transpiled program
    # NodeA (count=1) -> NodeB (decision=loop) -> NodeA (count=2) -> NodeB (decision=end) -> END
    result = program(input_text="test", processed_count=0, decision="proceed", max_steps=10)

    assert node_a_calls == 2
    assert node_b_calls == 2
    assert result["processed_count"] == 2
    assert result["decision"] == "end"
    assert result["_metadata"]["step_count"] == 4


def test_from_langgraph_node_configs_overrides():
    builder = StateGraph(TestStateDict)
    builder.add_node("NodeA", node_a)
    builder.add_node("NodeB", node_b)
    builder.add_edge(START, "NodeA")
    builder.add_edge("NodeA", "NodeB")
    builder.add_edge("NodeB", END)

    node_configs: dict[str, dict[str, Any]] = {
        "NodeA": {"max_retries": 5, "refine_instructions": "Verify A output strictly"},
        "NodeB": {"max_retries": 0},
    }

    dspyer_graph = from_langgraph(builder, node_configs=node_configs)

    assert dspyer_graph.nodes["NodeA"].max_retries == 5
    assert dspyer_graph.nodes["NodeA"].refine_instructions == "Verify A output strictly"
    assert dspyer_graph.nodes["NodeB"].max_retries == 0
    assert dspyer_graph.nodes["NodeB"].refine_instructions is None


def test_from_langgraph_auto_scaffold_execution(monkeypatch):
    builder = StateGraph(TestStateDict)
    builder.add_node("NodeA", node_a)
    builder.add_edge(START, "NodeA")
    builder.add_edge("NodeA", END)

    # Use a warnings context manager to catch the warning
    with pytest.warns(UserWarning, match="Auto-generating StatefulNode 'NodeA'"):
        dspyer_graph = from_langgraph(builder)

    program = AgentTranspiler.compile(dspyer_graph)
    assert dspyer_graph.nodes["NodeA"]._is_autogenerated is True

    class MockOutput:
        processed_count = 5

    def mock_predict(**kwargs):
        assert "processed_count" in kwargs
        assert "decision" in kwargs
        res = MockOutput()
        return res

    monkeypatch.setattr(program, "predictor_NodeA", mock_predict)

    # Execute without passing other fields in State
    result = program(input_text="only_this")
    assert result["processed_count"] == 5
    assert result["input_text"] == "only_this"
