import json
import os

import dspy
from pydantic import BaseModel, Field

from dspyer.compiler import AgentTranspiler
from dspyer.decorator import self_correcting
from dspyer.graph import Graph, StatefulNode
from dspyer.utils import load_logged_dataset


class SimpleOutput(BaseModel):
    answer: str
    confidence: float = Field(gt=0.5)


def test_decorator_retry_logging(tmp_path, monkeypatch):
    log_file = os.path.join(tmp_path, "dataset.jsonl")

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

    predict_calls = 0

    def mock_predict_dispatch(self_inst, *args, **kwargs):
        nonlocal predict_calls
        if "Refine" in getattr(self_inst.signature, "__name__", ""):
            return MockValidPrediction()
        else:
            predict_calls += 1
            return MockInvalidPrediction()

    monkeypatch.setattr(dspy.Predict, "forward", mock_predict_dispatch)

    @self_correcting(max_retries=3, dataset_log_path=log_file)
    def my_step(question: str) -> SimpleOutput:
        """Extract capital and confidence."""
        raise NotImplementedError()

    res = my_step(question="What is the capital of France?")
    assert res.answer == "Paris"
    assert res.confidence == 0.9

    # Verify log file exists and has 1 entry
    assert os.path.exists(log_file)
    with open(log_file, "r") as f:
        lines = f.readlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["question"] == "What is the capital of France?"
    assert data["answer"] == "Paris"
    assert data["confidence"] == 0.9


def test_decorator_no_logging_on_first_success(tmp_path, monkeypatch):
    log_file = os.path.join(tmp_path, "dataset.jsonl")

    class MockValidPrediction:
        def __init__(self):
            self.answer = "Rome"
            self.confidence = 0.85

        def items(self):
            return [("answer", self.answer), ("confidence", self.confidence)]

        def __getitem__(self, key):
            return getattr(self, key)

    def mock_predict_dispatch(self_inst, *args, **kwargs):
        return MockValidPrediction()

    monkeypatch.setattr(dspy.Predict, "forward", mock_predict_dispatch)

    @self_correcting(max_retries=3, dataset_log_path=log_file)
    def my_step(question: str) -> SimpleOutput:
        """Extract capital and confidence."""
        raise NotImplementedError()

    res = my_step(question="What is the capital of Italy?")
    assert res.answer == "Rome"
    assert not os.path.exists(log_file)


def test_pii_redaction_and_skip(tmp_path, monkeypatch):
    log_file = os.path.join(tmp_path, "dataset.jsonl")

    class MockInvalidPrediction:
        def __init__(self):
            self.answer = "Berlin"
            self.confidence = 0.2

        def items(self):
            return [("answer", self.answer), ("confidence", self.confidence)]

        def __getitem__(self, key):
            return getattr(self, key)

    class MockValidPrediction:
        def __init__(self):
            self.answer = "Berlin"
            self.confidence = 0.7

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

    # 1. Test Redaction
    def redact_name(example):
        example["question"] = example["question"].replace("Alice", "[REDACTED]")
        return example

    @self_correcting(max_retries=3, dataset_log_path=log_file, redact_hook=redact_name)
    def step_redact(question: str) -> SimpleOutput:
        """Step."""
        raise NotImplementedError()

    step_redact(question="Alice asks: What is the capital of Germany?")

    assert os.path.exists(log_file)
    with open(log_file, "r") as f:
        lines = f.readlines()
    data = json.loads(lines[0])
    assert data["question"] == "[REDACTED] asks: What is the capital of Germany?"

    os.remove(log_file)

    # 2. Test Skip Logging (return None)
    def skip_sensitive(example):
        if "secret" in example["question"]:
            return None
        return example

    @self_correcting(max_retries=3, dataset_log_path=log_file, redact_hook=skip_sensitive)
    def step_skip(question: str) -> SimpleOutput:
        """Step."""
        raise NotImplementedError()

    step_skip(question="A secret query about Germany")
    assert not os.path.exists(log_file)


def test_graph_compiler_dataset_logging(tmp_path, monkeypatch):
    log_file = os.path.join(tmp_path, "dataset.jsonl")

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

    def mock_predict_dispatch(self_inst, *args, **kwargs):
        if "Refine" in getattr(self_inst.signature, "__name__", ""):
            return MockValidPrediction()
        else:
            return MockInvalidPrediction()

    monkeypatch.setattr(dspy.Predict, "forward", mock_predict_dispatch)

    node = StatefulNode(
        name="PredictNode",
        input_model=InputSchema,
        output_model=OutputSchema,
        instructions="Extract val and confidence.",
        max_retries=2,
    )
    graph = Graph()
    graph.add_node(node)
    graph.set_entry_point("PredictNode")

    program = AgentTranspiler.compile(graph, dataset_log_path=log_file)

    res = program(val="Input string")
    assert res.res == "Yes"
    assert res.score == 0.95

    # Verify log entry
    assert os.path.exists(log_file)
    with open(log_file, "r") as f:
        lines = f.readlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["val"] == "Input string"
    assert data["res"] == "Yes"
    assert data["score"] == 0.95


def test_dataset_loader(tmp_path):
    log_file = os.path.join(tmp_path, "dataset.jsonl")
    entries = [
        {"question": "What is 2+2?", "answer": "4", "confidence": 1.0},
        {"question": "What is 3+3?", "answer": "6", "confidence": 1.0},
    ]
    with open(log_file, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    examples = load_logged_dataset(log_file, input_keys=["question"])
    assert len(examples) == 2
    assert isinstance(examples[0], dspy.Example)
    assert examples[0].question == "What is 2+2?"
    assert examples[0].answer == "4"
    assert examples[0]._input_keys == {"question"}
