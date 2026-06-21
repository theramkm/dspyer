import importlib.util
import logging
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger("dspyer.telemetry")

# Check if OpenTelemetry is available in the environment
HAS_OTEL = importlib.util.find_spec("opentelemetry") is not None

if HAS_OTEL:
    try:
        from opentelemetry import trace

        otel_tracer = trace.get_tracer("dspyer")
    except ImportError:
        HAS_OTEL = False
        otel_tracer = None
else:
    otel_tracer = None


class TelemetrySpan:
    """
    Abstractions over OTel Spans and log entities.
    """

    def __init__(self, name: str, trace_id: str, span_id: str):
        self.name = name
        self.trace_id = trace_id
        self.span_id = span_id
        self.start_time = time.time()
        self.otel_span = None

    def set_status(self, code: str, message: Optional[str] = None) -> None:
        if self.otel_span and HAS_OTEL:
            from opentelemetry.trace import Status, StatusCode

            status_code = StatusCode.ERROR if code == "ERROR" else StatusCode.OK
            self.otel_span.set_status(Status(status_code, description=message))

    def set_attribute(self, key: str, value: Any) -> None:
        if self.otel_span and HAS_OTEL:
            self.otel_span.set_attribute(key, str(value))


@contextmanager
def trace_span(
    name: str, inputs: Dict[str, Any], parent: Optional[TelemetrySpan] = None
) -> Generator[TelemetrySpan, None, None]:
    """
    Context manager to wrap node execution. Dispatches to OTel if installed,
    or falls back to structured logging.
    """
    trace_id = parent.trace_id if parent else str(uuid.uuid4())
    span_id = str(uuid.uuid4())
    span = TelemetrySpan(name, trace_id, span_id)

    if HAS_OTEL and otel_tracer:
        from opentelemetry.trace import Status, StatusCode

        ctx = None
        if parent and parent.otel_span:
            from opentelemetry.trace import set_span_in_context

            ctx = set_span_in_context(parent.otel_span)

        # Start OTel span
        otel_s = otel_tracer.start_span(name, context=ctx)
        span.otel_span = otel_s

        # Log input attributes
        for k, v in inputs.items():
            otel_s.set_attribute(f"input.{k}", str(v))

        try:
            yield span
        except Exception as e:
            otel_s.set_status(Status(StatusCode.ERROR, description=str(e)))
            otel_s.record_exception(e)
            raise
        finally:
            otel_s.end()
    else:
        # Structured log fallback
        logger.info(
            f"[SPAN START] TraceID: {trace_id} | SpanID: {span_id} | Node: {name} | Inputs: {inputs}"
        )
        try:
            yield span
            logger.info(
                f"[SPAN SUCCESS] TraceID: {trace_id} | SpanID: {span_id} | Duration: {time.time() - span.start_time:.4f}s"
            )
        except Exception as e:
            logger.error(
                f"[SPAN ERROR] TraceID: {trace_id} | SpanID: {span_id} | Error: {str(e)} | Duration: {time.time() - span.start_time:.4f}s"
            )
            raise
