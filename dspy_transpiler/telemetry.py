import importlib.util
import logging
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger("dspyer.telemetry")

# Check if OpenTelemetry is available in the environment
HAS_OTEL = importlib.util.find_spec("opentelemetry") is not None

otel_tracer: Optional[Any] = None
if HAS_OTEL:
    try:
        from opentelemetry import trace

        otel_tracer = trace.get_tracer("dspyer")
    except ImportError:
        HAS_OTEL = False
        otel_tracer = None
else:
    otel_tracer = None


_current_span: ContextVar[Optional["TelemetrySpan"]] = ContextVar("current_span", default=None)


def get_current_span() -> Optional[Any]:
    """Retrieve the current active TelemetrySpan or standard OpenTelemetry span."""
    span = _current_span.get()
    if span is not None:
        return span
    if HAS_OTEL:
        try:
            from opentelemetry import trace

            otel_span = trace.get_current_span()
            if otel_span and otel_span.is_recording():
                return otel_span
        except Exception:
            pass
    return None


class TelemetrySpan:
    """
    Abstractions over OTel Spans and log entities.
    """

    def __init__(self, name: str, trace_id: str, span_id: str):
        self.name = name
        self.trace_id = trace_id
        self.span_id = span_id
        self.start_time = time.time()
        self.otel_span: Optional[Any] = None

    def set_status(self, code: str, message: Optional[str] = None) -> None:
        if self.otel_span and HAS_OTEL:
            from opentelemetry.trace import Status, StatusCode

            status_code = StatusCode.ERROR if code == "ERROR" else StatusCode.OK
            self.otel_span.set_status(Status(status_code, description=message))

    def set_attribute(self, key: str, value: Any) -> None:
        if self.otel_span and HAS_OTEL:
            self.otel_span.set_attribute(key, str(value))

    def record_validation_error(self, err: Exception) -> None:
        """
        Record detailed validation error details onto the tracing span attributes.
        """
        self.set_attribute("validation.failed", True)
        if self.otel_span and HAS_OTEL:
            try:
                self.otel_span.record_exception(err)
            except Exception:
                pass

        if hasattr(err, "errors") and callable(err.errors):
            try:
                err_list = err.errors()
                self.set_attribute("validation.error.count", len(err_list))
                import json

                self.set_attribute("validation.errors_json", json.dumps(err_list))
                for i, error in enumerate(err_list):
                    loc_str = ".".join(str(x) for x in error.get("loc", []))
                    self.set_attribute(f"validation.error.{i}.field", loc_str)
                    self.set_attribute(f"validation.error.{i}.message", error.get("msg", ""))
                    self.set_attribute(f"validation.error.{i}.type", error.get("type", ""))
                    self.set_attribute(f"validation.error.{i}.input", str(error.get("input", "")))
            except Exception as parse_err:
                self.set_attribute("validation.error.parse_failure", str(parse_err))
        else:
            self.set_attribute("validation.error.message", str(err))

        if not HAS_OTEL:
            logger.warning(f"[VALIDATION FAILED] Span: {self.name} | Error: {str(err)}")


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

    token = _current_span.set(span)
    try:
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
            logger.info(f"[SPAN START] TraceID: {trace_id} | SpanID: {span_id} | Node: {name}")
            logger.debug(f"[SPAN DATA] TraceID: {trace_id} | SpanID: {span_id} | Inputs: {inputs}")
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
    finally:
        _current_span.reset(token)
