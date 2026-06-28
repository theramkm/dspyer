import asyncio
import sys

import dspy
import pytest
from pydantic import BaseModel, Field

from dspyer.decorator import self_correcting, wrap_predictor
from dspyer.trace import (
    Attempt,
    SelfCorrectionTrace,
    ValidationErrorDetail,
    _active_trace,
    execute_on_trace,
    get_trace,
    should_print_trace,
)


class SampleOutput(BaseModel):
    name: str
    value: float = Field(gt=10.0)


def test_validation_error_detail():
    err = ValidationErrorDetail(
        loc=["a", "b"], msg="required", type_str="value_error", input_val="bad"
    )
    d = err.to_dict()
    assert d["loc"] == ["a", "b"]
    assert d["msg"] == "required"
    assert d["type"] == "value_error"
    assert d["input"] == "bad"


def test_attempt_to_dict():
    att = Attempt(number=1, node_name="test_node")
    att.success = False
    att.duration_s = 0.5
    att.error_feedback = "Please fix it"
    att.validation_errors.append(
        ValidationErrorDetail(loc=["x"], msg="invalid", type_str="type_error", input_val=3)
    )
    att.outputs = {"res": "ok"}

    d = att.to_dict()
    assert d["number"] == 1
    assert d["node_name"] == "test_node"
    assert d["success"] is False
    assert d["duration_s"] == 0.5
    assert d["error_feedback"] == "Please fix it"
    assert len(d["validation_errors"]) == 1
    assert d["outputs"] == {"res": "ok"}


def test_trace_properties():
    trace = SelfCorrectionTrace(name="test_flow")
    assert trace.corrected is False
    assert trace.failed is False
    assert trace.retries == 0
    assert trace.failed_fields == []

    # Attempt 1: Failed
    att1 = Attempt(1, "node1")
    att1.validation_errors.append(
        ValidationErrorDetail(loc=["val"], msg="too small", type_str="value_error", input_val=5.0)
    )
    trace.attempts.append(att1)

    assert trace.corrected is False
    assert trace.failed is True
    assert trace.retries == 0
    assert trace.failed_fields == ["val"]

    # Attempt 2: Success
    att2 = Attempt(2, "node1")
    att2.success = True
    trace.attempts.append(att2)

    assert trace.corrected is True
    assert trace.failed is False
    assert trace.retries == 1
    assert trace.failed_fields == ["val"]


def test_pretty_string_truncation():
    trace = SelfCorrectionTrace(name="truncation_test")
    att = Attempt(1)
    att.success = True
    att.outputs = {
        "short": "hello",
        "long": "a" * 100,
        "binary_like": [1] * 20,
    }
    trace.attempts.append(att)
    s = trace.pretty_string()

    # Check that the long string is truncated
    assert "aaaa..." in s
    assert "[20 items]" in s
    assert "short = 'hello'" in s


def test_should_print_trace(monkeypatch):
    trace = SelfCorrectionTrace(name="test")
    # Unset
    monkeypatch.delenv("DSPYER_TRACE", raising=False)
    assert should_print_trace(trace) is False

    # Disabled / off
    monkeypatch.setenv("DSPYER_TRACE", "0")
    assert should_print_trace(trace) is False
    monkeypatch.setenv("DSPYER_TRACE", "false")
    assert should_print_trace(trace) is False

    # Active for failure / corrected only
    monkeypatch.setenv("DSPYER_TRACE", "1")
    assert should_print_trace(trace) is False  # Passed on attempt 1 with no failures

    # Failed
    trace.attempts.append(Attempt(1))  # Failed by default
    assert should_print_trace(trace) is True

    # Reset attempts, test all env
    trace.attempts = []
    monkeypatch.setenv("DSPYER_TRACE", "all")
    assert should_print_trace(trace) is True


def test_execute_on_trace():
    called = []

    def cb(t):
        called.append(t.name)

    trace = SelfCorrectionTrace("cb_test")
    execute_on_trace(cb, trace)
    assert called == ["cb_test"]

    # Callback throws exception: should be caught and swallowed
    def bad_cb(t):
        raise ValueError("Simulated handler crash")

    # This should not throw
    execute_on_trace(bad_cb, trace)


def test_get_trace():
    class Dummy:
        pass

    d = Dummy()
    assert get_trace(d) is None
    object.__setattr__(d, "dspyer_trace", "trace-object")
    assert get_trace(d) == "trace-object"


def test_decoupled_trace_silent(monkeypatch):
    monkeypatch.delenv("DSPYER_TRACE", raising=False)

    # Decorator with trace=True should not print to stderr
    # Let's verify by capturing stderr
    import io

    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:

        @self_correcting(schema=SampleOutput, trace=True)
        def process(data: str) -> SampleOutput:
            return SampleOutput(name="test", value=15.0)

        # Mock wrapped_predictor call
        # Create a dummy predictor that returns output
        class DummyPrediction:
            name = "test"
            value = 15.0

            def items(self):
                return [("name", "test"), ("value", 15.0)]

        # Patch wrapped_predictor in decorators module or process closure
        # Actually, let's just test wrap_predictor directly
        sig = dspy.make_signature("query -> output", instructions="test")
        pred = dspy.Predict(sig)

        # Patch call
        monkeypatch.setattr(pred, "forward", lambda *args, **kwargs: DummyPrediction())
        monkeypatch.setattr(pred, "__call__", lambda *args, **kwargs: DummyPrediction())

        wrapped = wrap_predictor(pred, SampleOutput, 2, None, trace=True)
        res = wrapped(query="hello")

        # The result must have trace attached
        assert res.dspyer_trace is not None
        assert isinstance(res.dspyer_trace, SelfCorrectionTrace)

        stderr_val = sys.stderr.getvalue()
        assert "dspyer" not in stderr_val
    finally:
        sys.stderr = old_stderr


def test_nesting_traces():
    # Outer trace
    outer_trace = SelfCorrectionTrace("outer")
    token = _active_trace.set(outer_trace)
    try:
        # Inner decorator/predictor should not create new trace, but reuse the outer one
        # Let's verify by calling a wrapped predictor
        class DummyPrediction:
            name = "inner"
            value = 12.0

            def items(self):
                return [("name", "inner"), ("value", 12.0)]

        sig = dspy.make_signature("query -> output", instructions="test")
        pred = dspy.Predict(sig)
        pred.forward = lambda *args, **kwargs: DummyPrediction()
        pred.__call__ = lambda *args, **kwargs: DummyPrediction()

        wrapped = wrap_predictor(pred, SampleOutput, 2, None, trace=True)
        res = wrapped(query="hello")

        # Result does NOT get a unique dspyer_trace since it didn't create one
        # But wait, does nested call attach the trace anyway?
        # Yes, it attaches the shared trace (outer_trace) to result!
        assert res.dspyer_trace is outer_trace
        assert len(outer_trace.attempts) == 1
        assert outer_trace.attempts[0].node_name == "SampleOutput"
    finally:
        _active_trace.reset(token)


@pytest.mark.asyncio
async def test_async_self_correction_trace_population(monkeypatch):
    # Tests that async wrapping propagates ContextVar and attempts correctly
    # inside asyncio.to_thread runs.
    class MockInvalidPrediction:
        name = "invalid"
        value = 2.0  # Fails gt=10.0 validation

        def items(self):
            return [("name", "invalid"), ("value", 2.0)]

        def __getitem__(self, key):
            return 2.0 if key == "value" else "invalid"

    class MockValidPrediction:
        name = "valid"
        value = 15.0

        def items(self):
            return [("name", "valid"), ("value", 15.0)]

        def __getitem__(self, key):
            return 15.0 if key == "value" else "valid"

    # We want a function that will fail on first attempt, and succeed on retry
    predict_calls = 0

    def mock_wrapped_predictor(*args, **kwargs):
        nonlocal predict_calls
        predict_calls += 1
        # Also simulate dspyer_trace on the intermediate prediction
        pred = MockValidPrediction() if predict_calls > 1 else MockInvalidPrediction()
        # Mock dspyer_trace attachment inside wrap_predictor
        return pred

    # Let's mock a signature and predictor
    sig = dspy.make_signature("query -> output", instructions="test")
    pred = dspy.Predict(sig)

    # Mocking call to return predictions
    monkeypatch.setattr(pred, "forward", mock_wrapped_predictor)
    monkeypatch.setattr(pred, "__call__", mock_wrapped_predictor)

    # Let's bind mock refiner
    class DummyRefine(dspy.Signature):
        query = dspy.InputField()
        failed_output = dspy.InputField()
        error_feedback = dspy.InputField()
        output = dspy.OutputField()

    refiner = dspy.Predict(DummyRefine)
    monkeypatch.setattr(refiner, "forward", lambda *args, **kwargs: MockValidPrediction())
    monkeypatch.setattr(refiner, "__call__", lambda *args, **kwargs: MockValidPrediction())
    setattr(pred, "_refiner_SampleOutput", refiner)

    # Decorate async function
    @self_correcting(trace=True)
    async def solve(query: str) -> SampleOutput:
        return SampleOutput(name="valid", value=15.0)

    # Manually patch wrapped_predictor inside decorator's function scope
    # Wait, solve is wrapped. We can construct a decorated async function
    # by directly wrapping a custom mock
    wrapped = wrap_predictor(pred, SampleOutput, 2, None, trace=True)

    # Call wrapped directly asynchronously in a thread
    res = await asyncio.to_thread(wrapped, query="run")

    # Verify trace population
    trace = get_trace(res)
    assert trace is not None
    assert trace.corrected is True
    assert trace.retries == 1
    assert len(trace.attempts) == 2
    assert trace.attempts[0].success is False
    assert trace.attempts[1].success is True
    assert len(trace.attempts[0].validation_errors) == 1


def test_unwritable_trace_target():
    from dspyer.trace import get_trace, register_trace

    # Built-in type (e.g. dict) is un-writable (raising TypeError on object.__setattr__)
    target_dict = {"data": "test"}
    trace = SelfCorrectionTrace(name="unwritable_target")

    # This should not raise and should successfully register in fallback WeakKeyDictionary
    register_trace(target_dict, trace)

    # Note: dict keys are unhashable, so register_trace will fail to add to weakref registry too,
    # but it must fail gracefully without crashing!
    assert get_trace(target_dict) is None

    # Custom extensible class
    class HashableCustomTarget:
        pass

    target_extensible = HashableCustomTarget()
    register_trace(target_extensible, trace)
    assert get_trace(target_extensible) is trace
