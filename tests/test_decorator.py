import os

import dspy
import pytest
from pydantic import BaseModel, Field, ValidationError

from dspyer.compiler import AgentTranspiler
from dspyer.decorator import parse_and_validate, self_correcting
from dspyer.graph import Graph, StatefulNode


class OutputSchema(BaseModel):
    answer: str
    confidence: float = Field(gt=0.5)


class TestSignature(dspy.Signature):
    question = dspy.InputField()
    output = dspy.OutputField()


def test_parse_and_validate_success():
    class SimplePrediction:
        def __init__(self, output):
            self.output = output

        def items(self):
            return [("output", self.output)]

    pred = SimplePrediction('{"answer": "Paris", "confidence": 0.8}')
    parsed, raw = parse_and_validate(pred, OutputSchema)
    assert parsed.answer == "Paris"  # type: ignore[attr-defined]
    assert parsed.confidence == 0.8  # type: ignore[attr-defined]
    assert raw["output"] == '{"answer": "Paris", "confidence": 0.8}'


def test_parse_and_validate_failure():
    class SimplePrediction:
        def __init__(self, output):
            self.output = output

        def items(self):
            return [("output", self.output)]

    pred = SimplePrediction('{"answer": "Paris", "confidence": 0.2}')
    with pytest.raises(ValidationError):
        parse_and_validate(pred, OutputSchema)


def test_predictor_decorator_retry_and_success(monkeypatch):

    class MockInvalidPrediction:
        def __init__(self):
            self.output = '{"answer": "Berlin", "confidence": 0.3}'

        def items(self):
            return [("output", self.output)]

        def __getitem__(self, key):
            return self.output

    class MockValidPrediction:
        def __init__(self):
            self.output = '{"answer": "Berlin", "confidence": 0.95}'

        def items(self):
            return [("output", self.output)]

        def __getitem__(self, key):
            return self.output

    predict_calls = 0
    refine_calls = 0

    def mock_orig_forward(*args, **kwargs):
        nonlocal predict_calls
        predict_calls += 1
        return MockInvalidPrediction()

    def mock_refiner_forward(*args, **kwargs):
        nonlocal refine_calls
        refine_calls += 1
        return MockValidPrediction()

    # Apply mock on the original predictor's forward (which wrapped saved)
    # The wrapped decorator replaces `predictor.forward` with `new_forward`.
    # Let's set the mock before wrapping to make it cleaner, or mock wrapped._refiner_OutputSchema
    # Actually, we can patch `predictor.forward` with a mock first, then decorate:
    predictor_base = dspy.Predict(TestSignature)
    monkeypatch.setattr(predictor_base, "forward", mock_orig_forward)

    wrapped_predictor = self_correcting(schema=OutputSchema, max_retries=2)(predictor_base)

    # Bind mock refiner
    class DummyRefine(dspy.Signature):
        question = dspy.InputField()
        failed_output = dspy.InputField()
        error_feedback = dspy.InputField()
        output = dspy.OutputField()

    refiner = dspy.Predict(DummyRefine)
    monkeypatch.setattr(refiner, "forward", mock_refiner_forward)
    monkeypatch.setattr(refiner, "__call__", mock_refiner_forward)
    setattr(predictor_base, "_refiner_OutputSchema", refiner)

    res = wrapped_predictor(question="What is the capital of Germany?")
    assert predict_calls == 1
    assert refine_calls == 1
    assert "confidence" in res.output


def test_module_decorator_retry_and_success(monkeypatch):
    class MockInvalidPrediction:
        def __init__(self):
            self.output = '{"answer": "Rome", "confidence": 0.4}'

        def items(self):
            return [("output", self.output)]

        def __getitem__(self, key):
            return self.output

    class MockValidPrediction:
        def __init__(self):
            self.output = '{"answer": "Rome", "confidence": 0.85}'

        def items(self):
            return [("output", self.output)]

        def __getitem__(self, key):
            return self.output

    predict_calls = 0
    refine_calls = 0

    def mock_predict_dispatch(self_inst, *args, **kwargs):
        nonlocal predict_calls, refine_calls
        # If it is the refiner signature
        if "Refine" in self_inst.signature.__name__:
            refine_calls += 1
            return MockValidPrediction()
        else:
            predict_calls += 1
            return MockInvalidPrediction()

    monkeypatch.setattr(dspy.Predict, "forward", mock_predict_dispatch)

    @self_correcting(schema=OutputSchema, max_retries=2)
    class TestModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate = dspy.Predict(TestSignature)

        def forward(self, question):
            return self.generate(question=question)

    solver = TestModule()

    # Verify nested predictor was wrapped
    assert getattr(solver.generate, "_wrapped_self_correcting", False) is True

    res = solver(question="What is the capital of Italy?")

    assert predict_calls == 1
    assert refine_calls == 1
    assert "confidence" in res.output


def test_module_decorator_custom_validator(monkeypatch):
    # Custom validator check
    def my_custom_validator(x: OutputSchema) -> bool:
        return "London" in x.answer

    class MockInvalidPrediction:
        def __init__(self):
            self.output = (
                '{"answer": "Paris", "confidence": 0.99}'  # Fails custom validator (no London)
            )

        def items(self):
            return [("output", self.output)]

        def __getitem__(self, key):
            return self.output

    class MockValidPrediction:
        def __init__(self):
            self.output = '{"answer": "London", "confidence": 0.99}'  # Passes both Pydantic and custom validator

        def items(self):
            return [("output", self.output)]

        def __getitem__(self, key):
            return self.output

    predict_calls = 0
    refine_calls = 0

    def mock_predict_dispatch(self_inst, *args, **kwargs):
        nonlocal predict_calls, refine_calls
        if "Refine" in self_inst.signature.__name__:
            refine_calls += 1
            return MockValidPrediction()
        else:
            predict_calls += 1
            return MockInvalidPrediction()

    monkeypatch.setattr(dspy.Predict, "forward", mock_predict_dispatch)

    @self_correcting(schema=OutputSchema, max_retries=2, validator=my_custom_validator)
    class CustomModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate = dspy.Predict(TestSignature)

        def forward(self, question):
            return self.generate(question=question)

    solver = CustomModule()

    res = solver(question="What is the capital of UK?")

    assert predict_calls == 1
    assert refine_calls == 1
    assert "London" in res.output


def test_prompt_configs_serialization(tmp_path):
    class DummyInput(BaseModel):
        val: str

    class DummyOutput(BaseModel):
        res: str

    node = StatefulNode(
        name="NodeX", input_model=DummyInput, output_model=DummyOutput, instructions="Extract X"
    )
    graph = Graph()
    graph.add_node(node)
    graph.set_entry_point("NodeX")

    program = AgentTranspiler.compile(graph)

    # Check save_prompts
    config_path = os.path.join(tmp_path, "prompts.json")
    program.save_prompts(config_path)

    assert os.path.exists(config_path)

    # Check load_prompts
    program.load_prompts(config_path)


def test_function_decorator_success(monkeypatch):
    class SimpleOutput(BaseModel):
        answer: str
        confidence: float

    class MockPrediction:
        def __init__(self):
            self.answer = "Berlin"
            self.confidence = 0.9

        def items(self):
            return [("answer", self.answer), ("confidence", self.confidence)]

        def __getitem__(self, key):
            return getattr(self, key)

    called = 0

    def mock_predict_forward(*args, **kwargs):
        nonlocal called
        called += 1
        return MockPrediction()

    monkeypatch.setattr(dspy.Predict, "forward", mock_predict_forward)

    @self_correcting(max_retries=2)
    def my_llm_step(question: str) -> SimpleOutput:
        """Extract capital and confidence."""
        raise NotImplementedError()

    res = my_llm_step(question="What is the capital of Germany?")
    assert called == 1
    assert isinstance(res, SimpleOutput)
    assert res.answer == "Berlin"
    assert res.confidence == 0.9


def test_function_decorator_retry_and_success(monkeypatch):
    class SimpleOutput(BaseModel):
        answer: str
        confidence: float = Field(gt=0.5)

    class MockInvalidPrediction:
        def __init__(self):
            self.answer = "Berlin"
            self.confidence = 0.2  # Fails gt=0.5 validation

        def items(self):
            return [("answer", self.answer), ("confidence", self.confidence)]

        def __getitem__(self, key):
            return getattr(self, key)

    class MockValidPrediction:
        def __init__(self):
            self.answer = "Berlin"
            self.confidence = 0.8  # Passes validation

        def items(self):
            return [("answer", self.answer), ("confidence", self.confidence)]

        def __getitem__(self, key):
            return getattr(self, key)

    predict_calls = 0
    refine_calls = 0

    def mock_predict_dispatch(self_inst, *args, **kwargs):
        nonlocal predict_calls, refine_calls
        if "Refine" in getattr(self_inst.signature, "__name__", ""):
            refine_calls += 1
            return MockValidPrediction()
        else:
            predict_calls += 1
            return MockInvalidPrediction()

    monkeypatch.setattr(dspy.Predict, "forward", mock_predict_dispatch)

    @self_correcting(max_retries=3)
    def my_retry_step(question: str) -> SimpleOutput:
        """Identify capital and verify confidence."""
        raise NotImplementedError()

    res = my_retry_step(question="What is the capital of Germany?")
    assert predict_calls == 1
    assert refine_calls == 1
    assert isinstance(res, SimpleOutput)
    assert res.answer == "Berlin"
    assert res.confidence == 0.8


def test_function_decorator_errors():
    # Missing return annotation
    with pytest.raises(TypeError, match="must have a return type annotation"):

        @self_correcting(max_retries=2)
        def step_no_return(x: str):
            pass

    # Return annotation is not a BaseModel
    with pytest.raises(TypeError, match="must have a return type annotation"):

        @self_correcting(max_retries=2)
        def step_invalid_return(x: str) -> str:
            raise NotImplementedError()
