import functools
import inspect
from typing import Any, Callable, Dict, Optional, Tuple, Type

import dspy
from pydantic import BaseModel, ValidationError
from pydantic_core import PydanticCustomError


def parse_and_validate(
    prediction: Any,
    schema: Type[BaseModel],
    validator: Optional[Callable[[Any], bool]] = None,
) -> Tuple[BaseModel, Dict[str, Any]]:
    """
    Extracts output fields from a DSPy prediction, repairs and parses any JSON strings,
    and validates them against a Pydantic schema and an optional custom validator.
    """
    raw_data = {}
    if hasattr(prediction, "items"):
        for k, v in prediction.items():
            raw_data[k] = v
    else:
        # Fallback for custom prediction objects or mocks
        for k in getattr(prediction, "_store", {}).keys():
            raw_data[k] = getattr(prediction, k, None)

    # If there is a single output field and its value can be validated/parsed directly
    if len(raw_data) == 1:
        val = list(raw_data.values())[0]
        if isinstance(val, schema):
            if validator is not None and not validator(val):
                raise ValidationError.from_exception_data(
                    title=schema.__name__,
                    line_errors=[
                        {
                            "loc": ("custom_validator",),
                            "input": val,
                            "type": PydanticCustomError(
                                "assertion_error", "Custom validation check failed."
                            ),
                        }
                    ],
                )
            return val, raw_data
        if isinstance(val, dict):
            try:
                parsed = schema.model_validate(val)
                if validator is not None and not validator(parsed):
                    raise ValidationError.from_exception_data(
                        title=schema.__name__,
                        line_errors=[
                            {
                                "loc": ("custom_validator",),
                                "input": parsed,
                                "type": PydanticCustomError(
                                    "assertion_error", "Custom validation check failed."
                                ),
                            }
                        ],
                    )
                return parsed, raw_data
            except Exception:
                pass
        if isinstance(val, str):
            try:
                from dspy_transpiler.compiler import repair_and_parse_json

                parsed_json = repair_and_parse_json(val)
                parsed = schema.model_validate(parsed_json)
                if validator is not None and not validator(parsed):
                    raise ValidationError.from_exception_data(
                        title=schema.__name__,
                        line_errors=[
                            {
                                "loc": ("custom_validator",),
                                "input": parsed,
                                "type": PydanticCustomError(
                                    "assertion_error", "Custom validation check failed."
                                ),
                            }
                        ],
                    )
                return parsed, raw_data
            except Exception:
                pass

    # Reconstruct dictionary from raw fields, attempting to repair strings containing JSON
    cleaned_data = {}
    from dspy_transpiler.compiler import repair_and_parse_json

    for k, v in raw_data.items():
        if isinstance(v, str) and (v.strip().startswith("{") or v.strip().startswith("[")):
            try:
                cleaned_data[k] = repair_and_parse_json(v)
            except Exception:
                cleaned_data[k] = v
        else:
            cleaned_data[k] = v

    parsed = schema.model_validate(cleaned_data)
    if validator is not None and not validator(parsed):
        raise ValidationError.from_exception_data(
            title=schema.__name__,
            line_errors=[
                {
                    "loc": ("custom_validator",),
                    "input": parsed,
                    "type": PydanticCustomError(
                        "assertion_error", "Custom validation check failed."
                    ),
                }
            ],
        )
    return parsed, raw_data


def make_signature_from_args_and_schema(
    arg_names: list[str], schema: Type[BaseModel], instructions: str
) -> Type[dspy.Signature]:
    """Dynamically builds a DSPy Signature class from a list of input arguments and an output schema."""
    fields: Dict[str, Tuple[Any, Any]] = {}
    for name in arg_names:
        fields[name] = (str, dspy.InputField(desc=name.replace("_", " ").title()))

    for name, field_info in schema.model_fields.items():
        desc = field_info.description or name.replace("_", " ").title()
        fields[name] = (field_info.annotation, dspy.OutputField(desc=desc))

    return dspy.make_signature(
        signature=fields,
        instructions=instructions,
        signature_name=f"{schema.__name__}Signature",
    )


def wrap_predictor(
    predictor: Any,
    schema: Optional[Type[BaseModel]],
    max_retries: int,
    refine_instructions: Optional[str],
    validator: Optional[Callable[[Any], bool]] = None,
    dataset_log_path: Optional[str] = None,
    redact_hook: Optional[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = None,
) -> Any:
    """Wraps an individual dspy.Predict instance with a correction retry loop."""
    if getattr(predictor, "_wrapped_self_correcting", False):
        return predictor

    orig_forward = object.__getattribute__(predictor, "forward")

    # If schema is not explicitly provided, look for a Pydantic model annotation in signature output fields
    target_schema = schema
    if target_schema is None:
        for field in predictor.signature.output_fields.values():
            if isinstance(field.annotation, type) and issubclass(field.annotation, BaseModel):
                target_schema = field.annotation
                break

    if target_schema is None:
        return predictor

    @functools.wraps(orig_forward)
    def new_forward(*args, **kwargs):
        sig = inspect.signature(orig_forward)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        current_inputs = {}
        for k, v in bound.arguments.items():
            if k == "self":
                continue
            if k == "kwargs" and isinstance(v, dict):
                current_inputs.update(v)
            else:
                current_inputs[k] = v

        # Temporarily restore original forward to allow calling the module directly
        # and triggering the standard __call__ machinery (which sets up DSPy traces etc)
        # without warning.
        predictor.forward = orig_forward
        try:
            prediction = predictor.__class__.__call__(predictor, *args, **kwargs)
        finally:
            predictor.forward = new_forward

        attempt = 0
        while attempt < max_retries:
            try:
                parsed_model, raw_data = parse_and_validate(prediction, target_schema, validator)
                if attempt > 0 and dataset_log_path is not None:
                    from dspy_transpiler.utils import log_self_correction_example

                    log_self_correction_example(
                        dataset_log_path,
                        current_inputs,
                        parsed_model.model_dump(),
                        redact_hook,
                    )
                return prediction
            except ValidationError as val_err:
                attempt += 1

                # Format natural language feedback
                feedback_lines = []
                for err in val_err.errors():
                    loc = " -> ".join(str(x) for x in err["loc"])
                    msg = err["msg"]
                    inp = err.get("input", "")
                    feedback_lines.append(f"Field '{loc}': {msg} (Value got: {inp})")
                feedback_str = "\n".join(feedback_lines)

                # Record error trace if OpenTelemetry span is active
                try:
                    from dspy_transpiler.telemetry import get_current_span

                    span = get_current_span()
                    if span is not None and hasattr(span, "record_validation_error"):
                        span.record_validation_error(val_err)
                except Exception:
                    pass

                if attempt >= max_retries:
                    raise val_err

                # Build refinement signature dynamically on the predictor
                refiner_attr = f"_refiner_{target_schema.__name__}"
                if not hasattr(predictor, refiner_attr):
                    fields = {}
                    for name, field in predictor.signature.input_fields.items():
                        fields[name] = (field.annotation, field)
                    fields["failed_output"] = (
                        str,
                        dspy.InputField(
                            desc="The previous invalid output that failed schema validation."
                        ),
                    )
                    fields["error_feedback"] = (
                        str,
                        dspy.InputField(
                            desc="Natural language feedback outlining the schema validation errors to fix."
                        ),
                    )
                    for name, field in predictor.signature.output_fields.items():
                        fields[name] = (field.annotation, field)

                    instructions = refine_instructions or (
                        "Review the original inputs and the failed_output. "
                        "Correct the failed_output based on the error_feedback. "
                        "Ensure the output satisfies the expected schema."
                    )
                    refine_sig = dspy.make_signature(
                        signature=fields,
                        instructions=instructions,
                        signature_name=f"{predictor.signature.__name__}Refine",
                    )
                    setattr(predictor, refiner_attr, dspy.Predict(refine_sig))

                refiner = getattr(predictor, refiner_attr)
                refiner_inputs = {
                    **current_inputs,
                    "failed_output": str(prediction),
                    "error_feedback": feedback_str,
                }
                prediction = refiner(**refiner_inputs)

        return prediction

    predictor.forward = new_forward
    # Override standard __call__ trigger as well
    predictor.__call__ = new_forward
    predictor._wrapped_self_correcting = True
    return predictor


def self_correcting(
    schema: Optional[Type[BaseModel]] = None,
    max_retries: int = 2,
    refine_instructions: Optional[str] = None,
    validator: Optional[Callable[[Any], bool]] = None,
    dataset_log_path: Optional[str] = None,
    redact_hook: Optional[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = None,
) -> Callable[[Any], Any]:
    """
    Decorator to wrap a dspy.Module class or a dspy.Predict instance with schema-enforced
    automatic correction retry loops.
    """

    def decorator(target: Any) -> Any:
        if isinstance(target, type) and issubclass(target, dspy.Module):
            # 1. Class-level decoration
            orig_init = target.__init__  # type: ignore[misc]

            @functools.wraps(orig_init)
            def new_init(self, *args, **kwargs):
                orig_init(self, *args, **kwargs)  # type: ignore[misc]
                # Walk the module attributes and auto-wrap all Predict instances
                for name, attr in list(self.__dict__.items()):
                    if hasattr(attr, "signature") and hasattr(attr, "forward"):
                        setattr(
                            self,
                            name,
                            wrap_predictor(
                                attr,
                                schema,
                                max_retries,
                                refine_instructions,
                                validator,
                                dataset_log_path,
                                redact_hook,
                            ),
                        )

            target.__init__ = new_init  # type: ignore[misc]

            # Wrap forward as a final safety gate/corrector if schema is explicitly passed
            if hasattr(target, "forward") and schema is not None:
                orig_forward = target.forward

                @functools.wraps(orig_forward)
                def new_forward(self, *args, **kwargs):
                    sig_inspect = inspect.signature(orig_forward)
                    bound = sig_inspect.bind(self, *args, **kwargs)
                    bound.apply_defaults()
                    current_inputs = {k: v for k, v in bound.arguments.items() if k != "self"}

                    prediction = orig_forward(self, *args, **kwargs)

                    attempt = 0
                    while attempt < max_retries:
                        try:
                            parsed_model, raw_data = parse_and_validate(
                                prediction, schema, validator
                            )
                            if attempt > 0 and dataset_log_path is not None:
                                from dspy_transpiler.utils import log_self_correction_example

                                log_self_correction_example(
                                    dataset_log_path,
                                    current_inputs,
                                    parsed_model.model_dump(),
                                    redact_hook,
                                )
                            return prediction
                        except ValidationError as val_err:
                            attempt += 1

                            # Format error message
                            feedback_lines = []
                            for err in val_err.errors():
                                loc = " -> ".join(str(x) for x in err["loc"])
                                msg = err["msg"]
                                inp = err.get("input", "")
                                feedback_lines.append(f"Field '{loc}': {msg} (Value got: {inp})")
                            feedback_str = "\n".join(feedback_lines)

                            try:
                                from dspy_transpiler.telemetry import get_current_span

                                span = get_current_span()
                                if span is not None and hasattr(span, "record_validation_error"):
                                    span.record_validation_error(val_err)
                            except Exception:
                                pass

                            if attempt >= max_retries:
                                raise val_err

                            # Resolve or instantiate module refiner dynamically
                            refiner_attr = f"_refiner_{schema.__name__}"
                            if not hasattr(self, refiner_attr):
                                orig_sig = make_signature_from_args_and_schema(
                                    list(current_inputs.keys()),
                                    schema,
                                    orig_forward.__doc__
                                    or self.__class__.__doc__
                                    or f"Correct output for {schema.__name__}",
                                )
                                fields = {}
                                for name, field in orig_sig.input_fields.items():
                                    fields[name] = (field.annotation, field)
                                fields["failed_output"] = (
                                    str,
                                    dspy.InputField(
                                        desc="The previous invalid output that failed schema validation."
                                    ),
                                )
                                fields["error_feedback"] = (
                                    str,
                                    dspy.InputField(
                                        desc="Natural language feedback outlining the schema validation errors to fix."
                                    ),
                                )
                                for name, field in orig_sig.output_fields.items():
                                    fields[name] = (field.annotation, field)

                                refine_sig = dspy.make_signature(
                                    signature=fields,
                                    instructions=refine_instructions
                                    or "Review inputs and correct the failed_output utilizing feedback.",
                                    signature_name=f"{schema.__name__}RefineSignature",
                                )
                                setattr(self, refiner_attr, dspy.Predict(refine_sig))

                            refiner = getattr(self, refiner_attr)
                            refiner_inputs = {
                                **current_inputs,
                                "failed_output": str(prediction),
                                "error_feedback": feedback_str,
                            }
                            prediction = refiner(**refiner_inputs)

                    return prediction

                target.forward = new_forward
            return target

        elif hasattr(target, "signature") and hasattr(target, "forward"):
            # 2. Predictor instance wrapping
            return wrap_predictor(
                target,
                schema,
                max_retries,
                refine_instructions,
                validator,
                dataset_log_path,
                redact_hook,
            )

        elif inspect.isfunction(target):
            # 3. Typed function wrapping
            sig = inspect.signature(target)

            # Check return type schema
            return_anno = sig.return_annotation
            if (
                return_anno is inspect.Signature.empty
                or not isinstance(return_anno, type)
                or not issubclass(return_anno, BaseModel)
            ):
                raise TypeError(
                    f"Decorated function '{target.__name__}' must have a return type annotation that is a subclass of pydantic.BaseModel"
                )

            target_schema = return_anno

            # Determine instructions
            instructions = (
                target.__doc__ or f"Execute logic to produce {target_schema.__name__} outputs."
            )
            instructions = instructions.strip()

            # Build input & output fields for the signature
            fields = {}
            for param_name, param in sig.parameters.items():
                anno = param.annotation if param.annotation is not inspect.Parameter.empty else str
                desc = param_name.replace("_", " ").title()
                fields[param_name] = (anno, dspy.InputField(desc=desc))

            for name, field_info in target_schema.model_fields.items():
                desc = field_info.description or name.replace("_", " ").title()
                fields[name] = (field_info.annotation, dspy.OutputField(desc=desc))

            # Compile dynamic signature
            dyn_sig = dspy.make_signature(
                signature=fields,
                instructions=instructions,
                signature_name=f"{target_schema.__name__}Signature",
            )

            # Build and wrap predictor
            predictor = dspy.Predict(dyn_sig)
            wrapped_predictor = wrap_predictor(
                predictor,
                target_schema,
                max_retries,
                refine_instructions,
                validator,
                dataset_log_path,
                redact_hook,
            )

            @functools.wraps(target)
            def wrapper(*args, **kwargs):
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                # Run the self-correcting predictor
                prediction = wrapped_predictor(**bound.arguments)
                # Parse and validate returned outputs back into target BaseModel
                parsed, _ = parse_and_validate(prediction, target_schema, validator)
                return parsed

            return wrapper

        else:
            raise TypeError(
                "@self_correcting decorator must wrap a dspy.Module class, a dspy.Predict instance, or a typed Python function"
            )

    return decorator
