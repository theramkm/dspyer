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


def from_langgraph(
    state_graph: Any, node_mappings: Optional[Dict[str, StatefulNode]] = None
) -> Graph:
    """
    Converts a LangGraph StateGraph builder or CompiledStateGraph into a dspyer Graph.

    Args:
        state_graph: The LangGraph StateGraph or CompiledStateGraph instance.
        node_mappings: Optional dictionary mapping LangGraph node names to dspyer StatefulNodes.
                       If not provided for a node, a default StatefulNode will be dynamically
                       generated using the node's function docstring/signature.

    Returns:
        Graph: A populated dspyer Graph topology ready for compilation.
    """
    # 1. Resolve to the underlying builder
    if hasattr(state_graph, "builder"):
        builder = state_graph.builder
    else:
        builder = state_graph

    if not hasattr(builder, "nodes") or not hasattr(builder, "edges"):
        raise ValueError(
            "Invalid state_graph object. Must be a LangGraph StateGraph or CompiledStateGraph."
        )

    # 2. Extract and resolve State model
    state_schema = getattr(builder, "state_schema", None)
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
    dspyer_graph = Graph()

    # 3. Process and register all nodes
    for node_name, node_spec in builder.nodes.items():
        if node_name in node_mappings:
            dspyer_graph.add_node(node_mappings[node_name])
        else:
            # Auto-generate StatefulNode from function docstring & inputs/outputs
            runnable = getattr(node_spec, "runnable", node_spec)
            func = getattr(runnable, "func", runnable)
            docstring = getattr(func, "__doc__", None)

            instructions = (
                docstring.strip() if docstring else f"Execute agent step for node '{node_name}'."
            )

            generated_node = StatefulNode(
                name=node_name,
                input_model=pydantic_state,
                output_model=pydantic_state,
                instructions=instructions,
            )
            dspyer_graph.add_node(generated_node)

    # 4. Map static edges
    for source, target in builder.edges:
        if source in ("__start__", "START"):
            # Set graph entry point
            dspyer_graph.set_entry_point(target)
        else:
            dspyer_graph.add_edge(source, target)

    # 5. Map conditional edges (branches)
    # builder.branches maps source_node -> {router_name: BranchSpec}
    for source, branch_dict in builder.branches.items():
        for router_key, branch_spec in branch_dict.items():
            router_callable = branch_spec.path
            router_func = getattr(router_callable, "func", router_callable)
            path_map = branch_spec.ends
            dspyer_graph.add_conditional_edges(source, router_func, path_map)

    return dspyer_graph
