import warnings
from typing import Type

import dspy

from dspy_transpiler.graph import StatefulNode


class DynamicSignatureBuilder:
    """
    Safely creates dspy.Signature classes dynamically at runtime from StatefulNodes.
    Uses explicit type mappings passed to custom_types to bypass frame introspection issues.
    """

    @staticmethod
    def build(node: StatefulNode) -> Type[dspy.Signature]:
        fields = {}

        # Parse inputs
        for name, field_info in node.input_model.model_fields.items():
            if not field_info.description:
                warnings.warn(
                    f"Field '{name}' in input model '{node.input_model.__name__}' is missing a description. "
                    f"Using fallback Title Case description.",
                    UserWarning,
                    stacklevel=2,
                )
            desc = field_info.description or name.replace("_", " ").title()
            fields[name] = (field_info.annotation, dspy.InputField(desc=desc))

        # Parse outputs
        for name, field_info in node.output_model.model_fields.items():
            if not field_info.description:
                warnings.warn(
                    f"Field '{name}' in output model '{node.output_model.__name__}' is missing a description. "
                    f"Using fallback Title Case description.",
                    UserWarning,
                    stacklevel=2,
                )
            desc = field_info.description or name.replace("_", " ").title()
            fields[name] = (field_info.annotation, dspy.OutputField(desc=desc))

        instructions = node.instructions or f"Process dynamic step: {node.name}"

        # Bypass stack frame lookup failures by injecting types explicitly
        custom_types = {
            node.input_model.__name__: node.input_model,
            node.output_model.__name__: node.output_model,
        }

        # Programmatically construct the Signature class
        signature = dspy.make_signature(
            signature=fields,
            instructions=instructions,
            signature_name=f"{node.name}Signature",
            custom_types=custom_types,
        )
        return signature

    @staticmethod
    def build_refine(node: StatefulNode) -> Type[dspy.Signature]:
        """
        Creates a refinement signature that accepts the original input fields plus the failed output
        and parsed error feedback to perform self-correction.
        """
        fields = {}

        # Parse original inputs
        for name, field_info in node.input_model.model_fields.items():
            desc = field_info.description or name.replace("_", " ").title()
            fields[name] = (field_info.annotation, dspy.InputField(desc=desc))

        # Add refinement parameters
        fields["failed_output"] = (
            str,
            dspy.InputField(desc="The previous invalid output that failed schema validation."),
        )
        fields["error_feedback"] = (
            str,
            dspy.InputField(
                desc="Natural language feedback outlining the schema validation errors to fix."
            ),
        )

        # Parse original outputs
        for name, field_info in node.output_model.model_fields.items():
            desc = field_info.description or name.replace("_", " ").title()
            fields[name] = (field_info.annotation, dspy.OutputField(desc=desc))

        instructions = (
            "Review the original inputs and the failed_output. "
            "Correct the failed_output based on the error_feedback. "
            "Ensure the output satisfies the expected schema."
        )

        # Bypass frame lookup failures
        custom_types = {
            node.input_model.__name__: node.input_model,
            node.output_model.__name__: node.output_model,
        }

        # Programmatically construct the Refine Signature class
        signature = dspy.make_signature(
            signature=fields,
            instructions=instructions,
            signature_name=f"{node.name}RefineSignature",
            custom_types=custom_types,
        )
        return signature
