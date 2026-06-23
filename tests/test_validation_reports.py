import os

import dspy
import pytest
from pydantic import BaseModel, Field, ValidationError

from dspyer.compiler import AgentTranspiler
from dspyer.decorator import self_correcting
from dspyer.graph import Graph, StatefulNode
from dspyer.utils import generate_validation_report


class SimpleOutput(BaseModel):
    answer: str
    confidence: float = Field(gt=0.5)


def test_decorator_validation_logging_success(tmp_path, monkeypatch):
    log_file = os.path.join(tmp_path, "validation.jsonl")

    class MockInvalidPrediction:
        def __init__(self):
            self.answer = "Paris"
            self.confidence = 0.2

        def items(self):
            return [("answer", self.answer), ("confidence", self.confidence)]

        def __getitem__(self, key):
            return getattr(self, key)

    class MockValidPrediction:
        def __init__(self):
            self.answer = "Paris"
            self.confidence = 0.9

        def items(self):
            return [("answer", self.answer), ("confidence", self.confidence)]

        def __getitem__(self, key):
            return getattr(self, key)

    def mock_predict_dispatch(self_inst, *args, **kwargs):
        if "Refine" in getattr(self_inst.signature, "__name__", ""):
            return MockValidPrediction()
        else:
            return MockInvalidPrediction()

    monkeypatch.setattr(dspy.Predict, "forward", mock_predict_dispatch)

    @self_correcting(max_retries=3, validation_log_path=log_file)
    def my_step(question: str) -> SimpleOutput:
        """Extract capital."""
        raise NotImplementedError()

    res = my_step(question="What is capital of France?")
    assert res.answer == "Paris"

    report = generate_validation_report(log_file)
    assert "Node: SimpleOutput" in report
    assert "Retry Rate: 100.0%" in report
    assert "confidence: 1 errors" in report


def test_decorator_validation_logging_failure(tmp_path, monkeypatch):
    log_file = os.path.join(tmp_path, "validation.jsonl")

    class MockInvalidPrediction:
        def __init__(self):
            self.answer = "Berlin"
            self.confidence = 0.2

        def items(self):
            return [("answer", self.answer), ("confidence", self.confidence)]

        def __getitem__(self, key):
            return getattr(self, key)

    def mock_predict_dispatch(self_inst, *args, **kwargs):
        return MockInvalidPrediction()

    monkeypatch.setattr(dspy.Predict, "forward", mock_predict_dispatch)

    @self_correcting(max_retries=2, validation_log_path=log_file)
    def my_step(question: str) -> SimpleOutput:
        """Extract capital."""
        raise NotImplementedError()

    with pytest.raises(ValidationError):
        my_step(question="What is capital of Germany?")

    report = generate_validation_report(log_file)
    assert "Node: SimpleOutput" in report
    assert "Successful Runs: 0" in report
    assert "Failed Runs: 1 (100.0%)" in report
    assert "confidence: 2 errors" in report


def test_graph_validation_report(tmp_path, monkeypatch):
    log_file = os.path.join(tmp_path, "validation.jsonl")

    class InputSchema(BaseModel):
        val: str

    class OutputSchema(BaseModel):
        res: str
        score: float = Field(gt=0.5)

    class MockInvalidPrediction:
        def __init__(self):
            self.res = "Yes"
            self.score = 0.1

        def items(self):
            return [("res", self.res), ("score", self.score)]

        def __getitem__(self, key):
            return getattr(self, key)

    class MockValidPrediction:
        def __init__(self):
            self.res = "Yes"
            self.score = 0.95

        def items(self):
            return [("res", self.res), ("score", self.score)]

        def __getitem__(self, key):
            return getattr(self, key)

    calls = 0

    def mock_predict_dispatch(self_inst, *args, **kwargs):
        nonlocal calls
        calls += 1
        if "Refine" in getattr(self_inst.signature, "__name__", ""):
            return MockValidPrediction()
        else:
            return MockInvalidPrediction()

    monkeypatch.setattr(dspy.Predict, "forward", mock_predict_dispatch)

    node = StatefulNode(
        name="PredictNode",
        input_model=InputSchema,
        output_model=OutputSchema,
        instructions="Extract details",
        max_retries=2,
    )
    graph = Graph()
    graph.add_node(node)
    graph.set_entry_point("PredictNode")

    program = AgentTranspiler.compile(graph, validation_log_path=log_file)

    program(val="Hello")

    report = generate_validation_report(log_file)
    assert "Node: PredictNode" in report
    assert "Retry Rate: 100.0%" in report
    assert "score: 1 errors" in report
