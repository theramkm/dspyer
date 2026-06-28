import logging
import os
import sys
import time
import weakref
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

logger = logging.getLogger("dspyer.trace")

# Global context variable to hold the active trace
_active_trace: ContextVar[Optional["SelfCorrectionTrace"]] = ContextVar(
    "active_trace", default=None
)


class ValidationErrorDetail:
    """Captures structured Pydantic validation error details."""

    def __init__(self, loc: List[str], msg: str, type_str: str, input_val: Any):
        self.loc = loc
        self.msg = msg
        self.type = type_str
        self.input = input_val

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loc": self.loc,
            "msg": self.msg,
            "type": self.type,
            "input": self.input,
        }


class Attempt:
    """Captures structured iteration details of a single correction attempt."""

    def __init__(self, number: int, node_name: Optional[str] = None):
        self.number = number
        self.node_name = node_name
        self.success = False
        self.duration_s = 0.0
        self.error_feedback: Optional[str] = None
        self.validation_errors: List[ValidationErrorDetail] = []
        self.outputs: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "number": self.number,
            "node_name": self.node_name,
            "success": self.success,
            "duration_s": round(self.duration_s, 4),
            "error_feedback": self.error_feedback,
            "validation_errors": [err.to_dict() for err in self.validation_errors],
            "outputs": self.outputs,
        }


class SelfCorrectionTrace:
    """Stores the complete hierarchical self-correction loop execution history."""

    def __init__(self, name: str):
        self.name = name
        self.attempts: List[Attempt] = []
        self.start_time = time.time()
        self.duration_s = 0.0

    @property
    def corrected(self) -> bool:
        """Returns True if the run initially failed validation but succeeded on a later retry."""
        if not self.attempts:
            return False
        has_failure = any(not a.success for a in self.attempts)
        has_success = any(a.success for a in self.attempts)
        return has_failure and has_success

    @property
    def failed(self) -> bool:
        """Returns True if all retry attempts failed validation."""
        if not self.attempts:
            return False
        return all(not a.success for a in self.attempts)

    @property
    def retries(self) -> int:
        """Returns the total number of retries executed."""
        return max(0, len(self.attempts) - 1)

    @property
    def failed_fields(self) -> List[str]:
        """Returns a list of unique field names that failed validation across all attempts."""
        fields = []
        for a in self.attempts:
            for err in a.validation_errors:
                loc_str = ".".join(str(x) for x in err.loc)
                if loc_str not in fields:
                    fields.append(loc_str)
        return fields

    def as_dict(self) -> Dict[str, Any]:
        """Returns a JSON-serializable dictionary representation of the trace."""
        return {
            "name": self.name,
            "attempts": [a.to_dict() for a in self.attempts],
            "corrected": self.corrected,
            "failed": self.failed,
            "retries": self.retries,
            "duration_s": round(self.duration_s, 4),
            "failed_fields": self.failed_fields,
        }

    def print(self) -> None:
        """Renders and prints the pretty-formatted trace representation to sys.stderr."""
        sys.stderr.write(self.pretty_string() + "\n")

    def pretty_string(self) -> str:
        """Generates a clean, tabular, human-readable terminal representation of the trace."""
        if self.failed:
            status = "✗ FAILED (retries exhausted)"
        elif self.corrected:
            status = "✓ corrected"
        else:
            status = "✓ passed"

        header_fields = (
            f" [failed fields: {', '.join(self.failed_fields)}]" if self.failed_fields else ""
        )
        header = f"dspyer · {self.name} · {len(self.attempts)} attempt{'s' if len(self.attempts) != 1 else ''} · {self.duration_s:.2f}s · {status}{header_fields}"
        divider = "─" * max(65, len(header))

        lines = [header, divider]

        def format_val(v: Any) -> str:
            if isinstance(v, list):
                count = len(v)
                s = f"[{count} item{'s' if count != 1 else ''}]"
            elif isinstance(v, dict):
                count = len(v)
                s = f"{{{count} key{'s' if count != 1 else ''}}}"
            else:
                s = repr(v)
            if len(s) > 80:
                return s[:77] + "..."
            return s

        for a in self.attempts:
            a_status = "✓ passed" if a.success else "✗ validation failed"
            prefix = f" [{a.node_name}]" if a.node_name else ""
            lines.append(f"attempt {a.number}{prefix}  {a_status} ({a.duration_s:.2f}s)")

            if not a.success:
                for err in a.validation_errors:
                    loc_str = ".".join(str(x) for x in err.loc)
                    got_val = format_val(err.input) if err.input is not None else "<missing>"
                    lines.append(f"   {loc_str:<15} {err.msg:<30} got: {got_val}")
                if a.error_feedback:
                    # Indent multiline feedback nicely
                    feedback_indented = a.error_feedback.replace("\n", "\n   ")
                    lines.append(f'   feedback sent → "{feedback_indented}"')
            else:
                out_parts = []
                for k, v in a.outputs.items():
                    out_parts.append(f"{k} = {format_val(v)}")
                if out_parts:
                    lines.append("   " + "   ".join(out_parts))

        lines.append(divider)
        return "\n".join(lines)


_trace_registry: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def register_trace(target: Any, trace: "SelfCorrectionTrace") -> None:
    """
    Registers the trace onto a target result or exception.
    Tries object.__setattr__ first and falls back to a global weak-key registry.
    """
    if target is None:
        return
    try:
        object.__setattr__(target, "dspyer_trace", trace)
    except Exception:
        pass
    try:
        _trace_registry[target] = trace
    except Exception:
        pass


def get_trace(target: Any) -> Optional["SelfCorrectionTrace"]:
    """
    Uniformly retrieves the SelfCorrectionTrace from a result or a caught exception.
    """
    if target is None:
        return None
    val = getattr(target, "dspyer_trace", None)
    if val is not None:
        return val
    try:
        return _trace_registry.get(target)
    except Exception:
        pass
    return None


def should_print_trace(trace: SelfCorrectionTrace) -> bool:
    """
    Centralized parsing of DSPYER_TRACE env var to filter what traces to output.
    Returns True if the trace matches the filtering criteria.
    """
    val = os.environ.get("DSPYER_TRACE", "").strip().lower()
    if val in ("all",):
        return True
    if val in ("1", "true"):
        return trace.corrected or trace.failed
    return False


def execute_on_trace(on_trace: Optional[Any], trace: SelfCorrectionTrace) -> None:
    """Safely executes the user callback, preventing callback errors from disrupting execution."""
    if on_trace is None:
        return
    try:
        on_trace(trace)
    except Exception as e:
        logger.warning(f"User-provided on_trace callback raised an exception: {e}", exc_info=True)


def record_attempt(
    trace: Optional[SelfCorrectionTrace],
    attempt_num: int,
    success: bool,
    duration_s: float,
    outputs: Dict[str, Any],
    validation_err: Optional[Any] = None,
    error_feedback: Optional[str] = None,
    node_name: Optional[str] = None,
) -> None:
    """Utility helper to record a single attempt iteration onto the active trace."""
    if trace is None:
        return

    # Check if this attempt number under the same node_name is already recorded
    existing = [a for a in trace.attempts if a.number == attempt_num and a.node_name == node_name]
    if existing:
        return

    attempt = Attempt(attempt_num, node_name=node_name)
    attempt.success = success
    attempt.duration_s = duration_s
    attempt.error_feedback = error_feedback
    attempt.outputs = dict(outputs)

    if validation_err is not None:
        try:
            # Handle Pydantic ValidationError errors list
            if hasattr(validation_err, "errors") and callable(validation_err.errors):
                for err in validation_err.errors():
                    loc = [str(x) for x in err.get("loc", [])]
                    msg = err.get("msg", "")
                    type_str = err.get("type", "")
                    input_val = err.get("input", None)
                    attempt.validation_errors.append(
                        ValidationErrorDetail(loc, msg, type_str, input_val)
                    )
            else:
                attempt.validation_errors.append(
                    ValidationErrorDetail(["unknown"], str(validation_err), "unknown", None)
                )
        except Exception:
            attempt.validation_errors.append(
                ValidationErrorDetail(["unknown"], str(validation_err), "unknown", None)
            )

    trace.attempts.append(attempt)
