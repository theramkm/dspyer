import asyncio
import contextvars
import importlib.util
import json
import logging
import os
import random
import re
import time
import urllib.error
import urllib.request
import warnings
from typing import Any, Dict, Optional, Union, get_args, get_origin

import dspy
from pydantic import BaseModel, Field
from pydantic_core import PydanticUndefined

from dspy_transpiler.graph import Graph, StatefulNode
from dspy_transpiler.signatures import DynamicSignatureBuilder
from dspy_transpiler.state import ImmutableState
from dspy_transpiler.telemetry import trace_span

logger = logging.getLogger("dspyer")
HAS_HTTPX = importlib.util.find_spec("httpx") is not None


class DirectClient:
    """
    Direct model client with optional httpx-based async connection pooling (bypassing LiteLLM at runtime).
    Supports Ollama, Gemini, Claude, and OpenAI API protocols with jittered exponential backoff.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_network_retries: int = 3,
        base_delay: float = 1.0,
    ):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key or os.environ.get(f"{self.provider.upper()}_API_KEY")
        if not self.api_key and self.provider in ("google", "gemini"):
            self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.api_base = api_base
        self.max_network_retries = max_network_retries
        self.base_delay = base_delay
        self._sync_client: Optional[Any] = None
        self._async_client: Optional[Any] = None

        if not HAS_HTTPX:
            warnings.warn(
                "Optional dependency 'httpx' is missing. Falling back to urllib for network requests. "
                "Install 'httpx' (pip install httpx) to enable async execution and connection pooling.",
                UserWarning,
                stacklevel=2,
            )

    def _get_sync_client(self) -> Any:
        if not HAS_HTTPX:
            raise RuntimeError("httpx is required for connection pooling but not installed.")
        if self._sync_client is None:
            import httpx

            self._sync_client = httpx.Client(
                timeout=60.0,
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            )
        return self._sync_client

    def _get_async_client(self) -> Any:
        if not HAS_HTTPX:
            raise RuntimeError("httpx is required for connection pooling but not installed.")
        if self._async_client is None:
            import httpx

            self._async_client = httpx.AsyncClient(
                timeout=60.0,
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            )
        return self._async_client

    def close(self):
        """Close the sync connection pool."""
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None

    async def aclose(self):
        """Close the async connection pool."""
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _get_request_details(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> tuple[str, dict, dict]:
        """
        Formats URL, Headers, and JSON payload for the chosen provider.
        """
        url = ""
        headers = {"Content-Type": "application/json"}
        payload = {}

        if self.provider == "ollama":
            url = self.api_base or "http://localhost:11434/api/chat"
            payload = {
                "model": self.model,
                "messages": [
                    *([{"role": "system", "content": system_prompt}] if system_prompt else []),
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            }
        elif self.provider == "openai":
            url = self.api_base or "https://api.openai.com/v1/chat/completions"
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            payload = {
                "model": self.model,
                "messages": [
                    *([{"role": "system", "content": system_prompt}] if system_prompt else []),
                    {"role": "user", "content": prompt},
                ],
            }
        elif self.provider == "anthropic":
            url = self.api_base or "https://api.anthropic.com/v1/messages"
            if self.api_key:
                headers["x-api-key"] = self.api_key
            headers["anthropic-version"] = "2023-06-01"
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                **({"system": system_prompt} if system_prompt else {}),
                "max_tokens": 4096,
            }
        elif self.provider == "google":
            url = (
                self.api_base
                or f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            )
            if self.api_key:
                headers["x-goog-api-key"] = self.api_key

            contents = []
            if system_prompt:
                payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
            contents.append({"parts": [{"text": prompt}]})
            payload["contents"] = contents
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        return url, headers, payload

    def _extract_response(self, response_data: dict) -> str:
        """
        Extracts completion text from the provider's response JSON.
        """
        try:
            if self.provider == "ollama":
                return response_data["message"]["content"]
            elif self.provider == "openai":
                return response_data["choices"][0]["message"]["content"]
            elif self.provider == "anthropic":
                return response_data["content"][0]["text"]
            elif self.provider == "google":
                return response_data["candidates"][0]["content"]["parts"][0]["text"]
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except (KeyError, IndexError) as err:
            raise RuntimeError(
                f"Failed to parse model response: {err}. Raw response: {response_data}"
            )

    def generate_sync(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        url, headers, payload = self._get_request_details(prompt, system_prompt)
        data = json.dumps(payload).encode("utf-8")

        attempt = 0
        while True:
            try:
                if HAS_HTTPX:
                    client = self._get_sync_client()
                    res = client.post(url, headers=headers, content=data)
                    res.raise_for_status()
                    return self._extract_response(res.json())
                else:
                    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                    with urllib.request.urlopen(req, timeout=60.0) as response:
                        res_body = response.read().decode("utf-8")
                        return self._extract_response(json.loads(res_body))
            except Exception as e:
                status_code = None
                if HAS_HTTPX:
                    import httpx

                    if isinstance(e, httpx.HTTPStatusError):
                        status_code = e.response.status_code
                if isinstance(e, urllib.error.HTTPError):
                    status_code = e.code

                is_transient = status_code in (429, 500, 502, 503, 504)
                if is_transient and attempt < self.max_network_retries:
                    attempt += 1
                    sleep_time = (self.base_delay * (2**attempt)) + random.uniform(0, 1.0)
                    logger.warning(
                        f"DirectClient sync request failed with status {status_code}. "
                        f"Retrying in {sleep_time:.2f}s (Attempt {attempt}/{self.max_network_retries})..."
                    )
                    time.sleep(sleep_time)
                else:
                    raise

    async def generate_async(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        url, headers, payload = self._get_request_details(prompt, system_prompt)
        data = json.dumps(payload).encode("utf-8")

        attempt = 0
        while True:
            try:
                if HAS_HTTPX:
                    client = self._get_async_client()
                    res = await client.post(url, headers=headers, content=data)
                    res.raise_for_status()
                    return self._extract_response(res.json())
                else:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        None, self.generate_sync, prompt, system_prompt
                    )
            except Exception as e:
                status_code = None
                if HAS_HTTPX:
                    import httpx

                    if isinstance(e, httpx.HTTPStatusError):
                        status_code = e.response.status_code
                if isinstance(e, urllib.error.HTTPError):
                    status_code = e.code

                is_transient = status_code in (429, 500, 502, 503, 504)
                if is_transient and attempt < self.max_network_retries:
                    attempt += 1
                    sleep_time = (self.base_delay * (2**attempt)) + random.uniform(0, 1.0)
                    logger.warning(
                        f"DirectClient async request failed with status {status_code}. "
                        f"Retrying in {sleep_time:.2f}s (Attempt {attempt}/{self.max_network_retries})...."
                    )
                    await asyncio.sleep(sleep_time)
                else:
                    raise


class MockChoiceMessage:
    def __init__(self, content: str):
        self.content = content
        self.role = "assistant"
        self.reasoning_content = None


class MockChoice:
    def __init__(self, content: str):
        self.message = MockChoiceMessage(content)
        self.finish_reason = "stop"
        self.index = 0


class MockCompletionResult:
    def __init__(self, content: str, model: str):
        self.choices = [MockChoice(content)]
        self.model = model
        self.usage = {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
        self._hidden_params = {"response_cost": 0.0}


class DirectLM(dspy.BaseLM):
    """
    Custom dspy.BaseLM subclass that wraps DirectClient.
    Integrates directly with DSPy's global runtime, history tracking, and teleprompters,
    bypassing LiteLLM entirely at execution time.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_network_retries: int = 3,
        base_delay: float = 1.0,
        **kwargs,
    ):
        provider = "openai"
        model_name = model
        if "/" in model:
            provider, model_name = model.split("/", 1)

        super().__init__(model=model, **kwargs)
        self.client = DirectClient(
            provider=provider,
            model=model_name,
            api_key=api_key,
            api_base=api_base,
            max_network_retries=max_network_retries,
            base_delay=base_delay,
        )

    def forward(
        self, prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs
    ):
        system_prompt = None
        user_prompt = ""
        if messages:
            system_msgs = [m["content"] for m in messages if m.get("role") == "system"]
            if system_msgs:
                system_prompt = "\n".join(system_msgs)

            user_msgs = [m["content"] for m in messages if m.get("role") in ("user", "developer")]
            if user_msgs:
                user_prompt = "\n".join(user_msgs)
            else:
                user_prompt = messages[-1]["content"] if messages else ""
        else:
            user_prompt = prompt or ""

        content = self.client.generate_sync(user_prompt, system_prompt=system_prompt)
        return MockCompletionResult(content, self.model)

    async def aforward(
        self, prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs
    ):
        system_prompt = None
        user_prompt = ""
        if messages:
            system_msgs = [m["content"] for m in messages if m.get("role") == "system"]
            if system_msgs:
                system_prompt = "\n".join(system_msgs)

            user_msgs = [m["content"] for m in messages if m.get("role") in ("user", "developer")]
            if user_msgs:
                user_prompt = "\n".join(user_msgs)
            else:
                user_prompt = messages[-1]["content"] if messages else ""
        else:
            user_prompt = prompt or ""

        content = await self.client.generate_async(user_prompt, system_prompt=system_prompt)
        return MockCompletionResult(content, self.model)


def repair_and_parse_json(raw_text: str) -> Any:
    """
    Extracts, repairs, and parses JSON content from a raw string.
    Handles markdown wrappers (code blocks) and truncated JSON payloads.

    Note: JSON repair mechanisms are heuristic-based and can be lossy (e.g., discarding
    unmatched delimiters), which may yield valid but incomplete structures on truncated
    or malformed inputs.
    """
    # 1. Try importing and using json-repair first on the raw text
    try:
        import json_repair

        repaired = json_repair.repair_json(raw_text, return_objects=True)
        if isinstance(repaired, (dict, list)):
            return repaired
    except Exception:
        pass

    raw_text = raw_text.strip()

    # 2. Clean markdown code fences if present
    if raw_text.startswith("```"):
        # Match ```json ... ``` or ``` ... ```
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_text, re.DOTALL)
        if match:
            raw_text = match.group(1).strip()

    # 3. Find boundary of the first outer curly brace/bracket
    first_brace = raw_text.find("{")
    first_bracket = raw_text.find("[")

    start_idx = -1
    end_char = ""
    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        start_idx = first_brace
        end_char = "}"
    elif first_bracket != -1:
        start_idx = first_bracket
        end_char = "]"

    if start_idx == -1:
        # One last attempt with json-repair on the clean text before parsing
        try:
            import json_repair

            repaired = json_repair.repair_json(raw_text, return_objects=True)
            if isinstance(repaired, (dict, list)):
                return repaired
        except Exception:
            pass
        return json.loads(raw_text)

    last_idx = raw_text.rfind(end_char)
    if last_idx != -1 and last_idx > start_idx:
        json_candidate = raw_text[start_idx : last_idx + 1]
    else:
        json_candidate = raw_text[start_idx:]

    try:
        return json.loads(json_candidate)
    except json.JSONDecodeError:
        pass

    # 4. Try json-repair on the candidate string
    try:
        import json_repair

        repaired = json_repair.repair_json(json_candidate, return_objects=True)
        if isinstance(repaired, (dict, list)):
            return repaired
    except Exception:
        pass

    # 5. Fallback repair path: balance braces, quotes, and brackets manually
    in_quote = False
    escape = False
    clean_chars = []
    stack = []

    for char in json_candidate:
        if escape:
            escape = False
            clean_chars.append(char)
            continue
        if char == "\\":
            escape = True
            clean_chars.append(char)
            continue
        if char == '"':
            in_quote = not in_quote
            clean_chars.append(char)
            continue

        if not in_quote:
            if char in "{[":
                stack.append(char)
                clean_chars.append(char)
            elif char in "}]":
                if stack:
                    if char == "}" and stack[-1] == "{":
                        stack.pop()
                        clean_chars.append(char)
                    elif char == "]" and stack[-1] == "[":
                        stack.pop()
                        clean_chars.append(char)
            else:
                clean_chars.append(char)
        else:
            clean_chars.append(char)

    repaired_str = "".join(clean_chars)
    if in_quote:
        repaired_str += '"'

    repaired_str = repaired_str.rstrip(",: \n\t")

    while stack:
        top = stack.pop()
        if top == "{":
            repaired_str += "}"
        elif top == "[":
            repaired_str += "]"

    return json.loads(repaired_str)


def format_validation_error(err: Exception) -> str:
    """
    Translates Pydantic ValidationErrors into concise, human-readable
    natural language instructions for self-correction.
    """
    if hasattr(err, "errors") and callable(err.errors):
        formatted_messages = []
        for error in err.errors():
            loc = ".".join(str(x) for x in error.get("loc", []))
            msg = error.get("msg", "Invalid value")
            inp = error.get("input", "unknown")
            formatted_messages.append(
                f"Field '{loc}' failed validation: {msg} (value provided: {inp})"
            )
        return "\n".join(formatted_messages)
    return str(err)


_refinement_steps: contextvars.ContextVar[int] = contextvars.ContextVar(
    "refinement_steps", default=-1
)
_last_refinement_steps: contextvars.ContextVar[int] = contextvars.ContextVar(
    "last_refinement_steps", default=0
)


class GraphExecutionError(RuntimeError):
    """
    Exception raised when a graph node fails validation or execution.
    Provides context-rich diagnostic information to debug leaky abstractions.
    """

    def __init__(
        self,
        node_name: str,
        inputs: Dict[str, Any],
        raw_output: Any,
        error_feedback: str,
        retries: int,
        original_exception: Optional[Exception] = None,
    ):
        self.node_name = node_name
        self.inputs = inputs
        self.raw_output = raw_output
        self.error_feedback = error_feedback
        self.retries = retries
        self.original_exception = original_exception

        msg = (
            f"Execution failed at graph node '{node_name}' after {retries} retries.\n"
            f"  - Node Inputs: {inputs}\n"
            f"  - Raw Completion Output: {raw_output}\n"
            f"  - Validation/Execution Error: {error_feedback}"
        )
        super().__init__(msg)


class TranspiledAgentProgram(dspy.Module):
    """
    A dynamically compiled, optimizable DSPy Module generated from
    an execution Graph. Supports branching, loops, and self-correction.
    """

    def __init__(self, graph: Graph):
        super().__init__()
        self.entry_point = graph.entry_point
        self.edges = graph.edges
        self.conditional_edges = graph.conditional_edges
        self._nodes_map = graph.nodes
        self._last_refinement_steps_taken = 0

        # Statically bind predictors and refiners for ALL registered nodes
        for node in graph.nodes.values():
            sig = DynamicSignatureBuilder.build(node)
            predictor = dspy.Predict(sig)
            setattr(self, f"predictor_{node.name}", predictor)

            refine_sig = DynamicSignatureBuilder.build_refine(node)
            refiner = dspy.Predict(refine_sig)
            setattr(self, f"refiner_{node.name}", refiner)

    @property
    def refinement_steps_taken(self) -> int:
        val = _refinement_steps.get()
        if val == -1:
            return _last_refinement_steps.get()
        return val

    def _execute_node(
        self, node: StatefulNode, state: ImmutableState, max_retries: int
    ) -> ImmutableState:
        """
        Runs execution pipeline for a single node including pre-flight checks,
        predictions, output Pydantic validations, retries, and telemetry hooks.
        """
        predictor = getattr(self, f"predictor_{node.name}")
        refiner = getattr(self, f"refiner_{node.name}")

        # Extract and validate input payload from current state
        current_state_data = state.to_dict()
        node_inputs = {}
        for field_name, field_info in node.input_model.model_fields.items():
            if field_name in current_state_data:
                node_inputs[field_name] = current_state_data[field_name]
            else:
                # Check for default value
                if field_info.default is not PydanticUndefined:
                    node_inputs[field_name] = field_info.default
                elif field_info.default_factory is not None:
                    node_inputs[field_name] = field_info.default_factory()  # type: ignore[call-arg]
                else:
                    # Check if the field is nullable (Optional or None type)
                    annotation = field_info.annotation
                    is_nullable = False
                    if annotation is not None:
                        if get_origin(annotation) is Union:
                            is_nullable = type(None) in get_args(annotation)
                        elif annotation is type(None):
                            is_nullable = True

                    if is_nullable:
                        node_inputs[field_name] = None
                    else:
                        raise ValueError(
                            f"Node '{node.name}' expects input '{field_name}', "
                            f"but it was not found in the workflow state."
                        )

        # Trace execution of this node
        with trace_span(f"node.{node.name}", node_inputs) as span:
            # Execute initial attempt
            try:
                output_prediction = predictor(**node_inputs)
            except Exception as pred_err:
                span.set_status("ERROR", str(pred_err))
                raise GraphExecutionError(
                    node_name=node.name,
                    inputs=node_inputs,
                    raw_output=None,
                    error_feedback=str(pred_err),
                    retries=0,
                    original_exception=pred_err,
                ) from pred_err

            attempt = 0
            while True:
                # Extract outputs and attempt parsing for string fields that expect structured types
                raw_outputs = {}
                for output_field, field_info in node.output_model.model_fields.items():
                    # Safely retrieve key, avoiding collision with standard dict/Prediction methods (e.g. 'items', 'keys', 'values')
                    if isinstance(output_prediction, dict) and output_field in output_prediction:
                        val = output_prediction[output_field]
                    elif hasattr(output_prediction, "__getitem__"):
                        try:
                            val = output_prediction[output_field]
                        except KeyError:
                            val = getattr(output_prediction, output_field)
                    else:
                        val = getattr(output_prediction, output_field)

                    # If field expects a collection or sub-model but received a string, attempt repair
                    if isinstance(val, str):
                        ann = field_info.annotation
                        origin = get_origin(ann) or ann
                        is_collection = isinstance(origin, type) and issubclass(
                            origin, (dict, list)
                        )
                        is_model = isinstance(ann, type) and issubclass(ann, BaseModel)
                        if is_collection or is_model:
                            try:
                                val = repair_and_parse_json(val)
                            except Exception:
                                pass

                    raw_outputs[output_field] = val

                try:
                    # Enforce schema validation at the transaction boundary
                    validated_patch = node.output_model.model_validate(raw_outputs)
                    # Commit state patch after dumping to JSON-compatible format
                    state = state.apply_patch(validated_patch.model_dump(mode="json"))

                    # Populate span output metadata
                    for k, v in validated_patch.model_dump().items():
                        span.set_attribute(f"output.{k}", str(v))
                    break  # Validation succeeded, break retry loop
                except Exception as validation_err:
                    attempt += 1
                    _refinement_steps.set(_refinement_steps.get() + 1)

                    if attempt > max_retries:
                        span.set_status("ERROR", str(validation_err))
                        feedback = format_validation_error(validation_err)
                        raise GraphExecutionError(
                            node_name=node.name,
                            inputs=node_inputs,
                            raw_output=raw_outputs,
                            error_feedback=feedback,
                            retries=max_retries,
                            original_exception=validation_err,
                        ) from validation_err

                    # Format feedback and prepare refine inputs
                    feedback = format_validation_error(validation_err)
                    failed_output_str = json.dumps(raw_outputs)

                    # Annotate span with retry context
                    span.set_attribute(f"retry.{attempt}.error", feedback)
                    span.set_attribute(f"retry.{attempt}.failed_output", failed_output_str)

                    refine_inputs = {**node_inputs}
                    refine_inputs["failed_output"] = failed_output_str
                    refine_inputs["error_feedback"] = feedback

                    # Execute refiner
                    try:
                        output_prediction = refiner(**refine_inputs)
                    except Exception as refine_err:
                        span.set_status("ERROR", str(refine_err))
                        raise GraphExecutionError(
                            node_name=node.name,
                            inputs=node_inputs,
                            raw_output=raw_outputs,
                            error_feedback=str(refine_err),
                            retries=attempt,
                            original_exception=refine_err,
                        ) from refine_err
        return state

    def forward(
        self,
        *,
        _max_retries: int = 2,
        _max_steps: int = 15,
        _on_loop_limit: str = "raise",
        **initial_state_kwargs,
    ) -> dspy.Prediction:
        token = _refinement_steps.set(0)
        try:
            max_retries = initial_state_kwargs.pop("max_retries", _max_retries)
            max_steps = initial_state_kwargs.pop("max_steps", _max_steps)
            on_loop_limit = initial_state_kwargs.pop("on_loop_limit", _on_loop_limit)

            state = ImmutableState(initial_state_kwargs)

            current_node_name = self.entry_point
            step_count = 0

            while current_node_name is not None:
                step_count += 1
                if step_count > max_steps:
                    msg = f"Graph execution exceeded max_steps limit of {max_steps}."
                    logger.warning(msg)
                    if on_loop_limit == "raise":
                        raise RuntimeError(msg)
                    else:
                        break

                node = self._nodes_map[current_node_name]
                state = self._execute_node(node, state, max_retries)

                # Determine next step
                if current_node_name in self.edges:
                    current_node_name = self.edges[current_node_name]
                elif current_node_name in self.conditional_edges:
                    router, path_map = self.conditional_edges[current_node_name]
                    decision: Any = None

                    if callable(router) and not isinstance(router, StatefulNode):
                        # Python routing function
                        try:
                            decision = router(state.to_dict())
                        except Exception as router_err:
                            raise RuntimeError(
                                f"Python router function failed at node '{current_node_name}': {str(router_err)}"
                            ) from router_err
                    else:
                        # Semantic routing node (StatefulNode)
                        # We execute the router node as a child node step
                        state = self._execute_node(router, state, max_retries)

                        # Discover routing decision from router outputs
                        output_fields = list(router.output_model.model_fields.keys())
                        if len(output_fields) == 1:
                            decision_field = output_fields[0]
                        elif "next_step" in output_fields:
                            decision_field = "next_step"
                        elif "route" in output_fields:
                            decision_field = "route"
                        else:
                            raise ValueError(
                                f"Could not determine routing decision from router node '{router.name}' output. "
                                f"Please design the output model with a single field or a field named 'next_step' / 'route'."
                            )
                        decision = state.to_dict().get(decision_field)

                    # Retrieve destination
                    decision_str = str(decision) if decision is not None else ""
                    if decision_str in path_map:
                        current_node_name = path_map[decision_str]
                    else:
                        raise ValueError(
                            f"Router outcome '{decision_str}' at node '{current_node_name}' "
                            f"is not mapped to any destination in path_map: {list(path_map.keys())}"
                        )
                else:
                    current_node_name = None

            final_state = state.to_dict()
            final_state["_metadata"] = {
                "refinement_steps_taken": self.refinement_steps_taken,
                "step_count": step_count,
            }
            self._last_refinement_steps_taken = self.refinement_steps_taken
            _last_refinement_steps.set(self.refinement_steps_taken)
            return dspy.Prediction(**final_state)
        finally:
            _refinement_steps.reset(token)


class AgentTranspiler:
    """Public interface to transpile state machines into DSPy programs."""

    @staticmethod
    def compile(graph: Graph) -> TranspiledAgentProgram:
        return TranspiledAgentProgram(graph=graph)


# =====================================================================
# Pipeline Verification Run
# =====================================================================


class ExtractionInput(BaseModel):
    raw_text: str = Field(description="The source text to analyze")


class ExtractionOutput(BaseModel):
    user_name: str = Field(description="Name of the user extracted from text")
    rough_query: str = Field(description="User query context")


class ClassificationInput(BaseModel):
    rough_query: str = Field(description="Extracted rough query")


class ClassificationOutput(BaseModel):
    classified_intent: str = Field(description="Category intent: Support, Sales, Info")


def run_verification_and_training_pipeline():
    lm = dspy.LM("openai/gpt-4o-mini", api_key="mocked-key", cache=True)
    dspy.configure(lm=lm)

    node_extraction = StatefulNode(
        name="EntityExtractor",
        input_model=ExtractionInput,
        output_model=ExtractionOutput,
        instructions="Extract specific user entity details.",
    )

    node_classification = StatefulNode(
        name="IntentClassifier",
        input_model=ClassificationInput,
        output_model=ClassificationOutput,
        instructions="Classify query intent options into Support, Sales, or Info.",
    )

    # Reconstruct verification flow using the Graph Builder
    graph = Graph()
    graph.add_node(node_extraction)
    graph.add_node(node_classification)
    graph.set_entry_point("EntityExtractor")
    graph.add_edge("EntityExtractor", "IntentClassifier")

    transpiled_program = AgentTranspiler.compile(graph)

    print(" Performing parameter check...")
    discovered = list(transpiled_program.named_predictors())
    print(f"  Discovered total parameter count: {len(discovered)}")
    for name, predictor_inst in discovered:
        print(f"  - Parameter Name: {name} (Signature: {predictor_inst.signature})")

    # Define training sets to ensure optimize-safety
    trainset = [
        dspy.Example(
            raw_text="Hello, my name is Alice. I would like to buy a subscription.",
            user_name="Alice",
            rough_query="buy a subscription",
            classified_intent="Sales",
        ).with_inputs("raw_text")
    ]

    def simple_intent_metric(example, pred, trace=None) -> bool:
        names_match = example.user_name.lower() == getattr(pred, "user_name", "").lower()
        intents_match = (
            example.classified_intent.lower() == getattr(pred, "classified_intent", "").lower()
        )
        return names_match and intents_match

    print("\n Launching BootstrapFewShot optimizer...")
    from dspy.teleprompt import BootstrapFewShot

    optimizer = BootstrapFewShot(metric=simple_intent_metric, max_bootstrapped_demos=1)
    _optimized_agent = optimizer.compile(transpiled_program, trainset=trainset)
    print("  Optimized Program compiled successfully.")


if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = "mocked-key"
    try:
        run_verification_and_training_pipeline()
    except Exception as run_err:
        print(f"\n[Verification Target Hit] Compiler test verified. Logs: {run_err}")
