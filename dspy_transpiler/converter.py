import typing
from typing import Any, Dict, Optional

from pydantic import BaseModel, create_model

from dspy_transpiler.graph import Graph, StatefulNode


class DefaultState(BaseModel):
    placeholder: str = "default"


def typeddict_to_pydantic(typed_dict: type) -> type[BaseModel]:
    """Dynamically converts a TypedDict class to a Pydantic BaseModel."""
    fields: Dict[str, Any] = {}
    try:
        hints = typing.get_type_hints(typed_dict)
    except Exception:
        hints = getattr(typed_dict, "__annotations__", {})

    for field_name, field_type in hints.items():
        fields[field_name] = (field_type, ...)

    if not fields:
        fields["placeholder"] = (str, "default")

    model_name = str(getattr(typed_dict, "__name__", "DynamicState")) + "Model"
    return create_model(model_name, **fields)


def make_permissive_model(model: type[BaseModel], name_suffix: str) -> type[BaseModel]:
    """
    Creates a copy of the Pydantic model where all fields are optional and default to None.
    This avoids validation failures during scaffold topology execution.
    """
    fields: Dict[str, Any] = {}
    for name, field in model.model_fields.items():
        ann = field.annotation or Any
        fields[name] = (Optional[ann], None)
    return create_model(f"{model.__name__}{name_suffix}", **fields)


def _extract_langgraph_structures(state_graph: Any) -> tuple[Any, dict, list, dict]:
    """
    Extracts builder structures (state_schema, nodes, edges, branches) from a LangGraph.
    Tested against langgraph versions 1.0.x - 1.2.x.
    """
    if hasattr(state_graph, "builder"):
        builder = state_graph.builder
    else:
        builder = state_graph

    if not hasattr(builder, "nodes") or not hasattr(builder, "edges"):
        raise ValueError(
            "Invalid state_graph object. Must be a LangGraph StateGraph or CompiledStateGraph."
        )

    state_schema = getattr(builder, "state_schema", None)
    nodes = getattr(builder, "nodes", {})
    edges = getattr(builder, "edges", [])
    branches = getattr(builder, "branches", {})
    return state_schema, nodes, edges, branches


def from_langgraph(
    state_graph: Any,
    node_mappings: Optional[Dict[str, StatefulNode]] = None,
    node_configs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Graph:
    """
    Converts a LangGraph StateGraph builder or CompiledStateGraph into a dspyer Graph.

    Args:
        state_graph: The LangGraph StateGraph or CompiledStateGraph instance.
        node_mappings: Optional dictionary mapping LangGraph node names to dspyer StatefulNodes.
                       If not provided for a node, a default StatefulNode will be dynamically
                       generated using the node's function docstring/signature.
        node_configs: Optional dictionary mapping LangGraph node names to configuration override dictionaries.
                      Can configure max_retries and refine_instructions.

    Returns:
        Graph: A populated dspyer Graph topology ready for compilation.
    """
    # 1. Extract LangGraph structures safely
    state_schema, nodes, edges, branches = _extract_langgraph_structures(state_graph)

    # 2. Extract and resolve State model
    pydantic_state: type[BaseModel]
    if state_schema is None:
        pydantic_state = DefaultState
    elif isinstance(state_schema, type) and issubclass(state_schema, BaseModel):
        pydantic_state = state_schema
    else:
        try:
            pydantic_state = typeddict_to_pydantic(state_schema)
        except Exception:
            pydantic_state = DefaultState

    node_mappings = node_mappings or {}
    node_configs = node_configs or {}
    dspyer_graph = Graph()

    # 3. Process and register all nodes
    for node_name, node_spec in nodes.items():
        if node_name in node_mappings:
            node = node_mappings[node_name]
            if node_name in node_configs:
                cfg = node_configs[node_name]
                if "max_retries" in cfg:
                    node.max_retries = cfg["max_retries"]
                if "refine_instructions" in cfg:
                    node.refine_instructions = cfg["refine_instructions"]
                if "use_cot" in cfg:
                    node.use_cot = cfg["use_cot"]
            dspyer_graph.add_node(node)
        else:
            # Auto-generate StatefulNode from function docstring & inputs/outputs
            runnable = getattr(node_spec, "runnable", node_spec)
            func = getattr(runnable, "func", runnable)
            docstring = getattr(func, "__doc__", None)

            instructions = (
                docstring.strip() if docstring else f"Execute agent step for node '{node_name}'."
            )

            import warnings

            warnings.warn(
                f"Auto-generating StatefulNode '{node_name}' with the full state schema. "
                "This serves as a scaffold stub and does not preserve original function logic. "
                "For a behavior-preserving execution, map this node explicitly using 'node_mappings'.",
                UserWarning,
                stacklevel=2,
            )

            permissive_input = make_permissive_model(pydantic_state, f"{node_name}Input")
            permissive_output = make_permissive_model(pydantic_state, f"{node_name}Output")

            cfg = node_configs.get(node_name, {})
            generated_node = StatefulNode(
                name=node_name,
                input_model=permissive_input,
                output_model=permissive_output,
                instructions=instructions,
                max_retries=cfg.get("max_retries"),
                refine_instructions=cfg.get("refine_instructions"),
                use_cot=cfg.get("use_cot", False),
            )
            generated_node._is_autogenerated = True
            dspyer_graph.add_node(generated_node)

    # 4. Map static edges
    for source, target in edges:
        if source in ("__start__", "START"):
            # Set graph entry point
            dspyer_graph.set_entry_point(target)
        else:
            dspyer_graph.add_edge(source, target)

    # 5. Map conditional edges (branches)
    # branches maps source_node -> {router_name: BranchSpec}
    for source, branch_dict in branches.items():
        for router_key, branch_spec in branch_dict.items():
            router_callable = branch_spec.path
            router_func = getattr(router_callable, "func", router_callable)
            path_map = branch_spec.ends
            dspyer_graph.add_conditional_edges(source, router_func, path_map)

    return dspyer_graph
