import ast
import inspect
import textwrap
import typing
from typing import Any, Dict, Optional, Set, Tuple

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


class StateVisitor(ast.NodeVisitor):
    def __init__(self, state_var_name: str):
        self.state_var_name = state_var_name
        self.input_keys: Set[str] = set()
        self.output_keys: Set[str] = set()
        self.is_llm = False
        self.is_dynamic = False

    def visit_Name(self, node: ast.Name):
        if node.id in ("dspy", "openai", "anthropic", "litellm"):
            self.is_llm = True
        if node.id == self.state_var_name:
            self.is_dynamic = True
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr in ("dspy", "openai", "anthropic", "litellm"):
            self.is_llm = True
        if isinstance(node.value, ast.Name) and node.value.id == self.state_var_name:
            if node.attr not in (
                "get",
                "keys",
                "values",
                "items",
                "copy",
                "update",
                "clear",
                "pop",
                "popitem",
                "setdefault",
            ):
                self.input_keys.add(node.attr)
            return
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        if isinstance(node.value, ast.Name) and node.value.id == self.state_var_name:
            slice_node = node.slice
            if slice_node.__class__.__name__ == "Index":
                slice_node = getattr(slice_node, "value")
            if isinstance(slice_node, ast.Constant):
                val = slice_node.value
                if isinstance(val, str):
                    self.input_keys.add(val)
                else:
                    self.is_dynamic = True
            elif isinstance(slice_node, ast.Str):  # Python < 3.8
                s_val = slice_node.s
                if isinstance(s_val, str):
                    self.input_keys.add(s_val)
            else:
                self.is_dynamic = True
            return
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == self.state_var_name
        ):
            if node.func.attr == "get":
                if node.args:
                    first_arg = node.args[0]
                    if isinstance(first_arg, ast.Constant):
                        val = first_arg.value
                        if isinstance(val, str):
                            self.input_keys.add(val)
                        else:
                            self.is_dynamic = True
                    elif isinstance(first_arg, ast.Str):
                        s_val = first_arg.s
                        if isinstance(s_val, str):
                            self.input_keys.add(s_val)
                        else:
                            self.is_dynamic = True
                    else:
                        self.is_dynamic = True
                else:
                    self.is_dynamic = True
                for arg in node.args[1:]:
                    self.visit(arg)
                for kw in node.keywords:
                    self.visit(kw)
                return
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in ("invoke", "ainvoke", "generate", "predict"):
                self.is_llm = True
        elif isinstance(node.func, ast.Name):
            if node.func.id in ("invoke", "ainvoke", "generate", "predict"):
                self.is_llm = True
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == self.state_var_name
            ):
                slice_node = target.slice
                if slice_node.__class__.__name__ == "Index":
                    slice_node = getattr(slice_node, "value")
                if isinstance(slice_node, ast.Constant):
                    val = slice_node.value
                    if isinstance(val, str):
                        self.output_keys.add(val)
                    else:
                        self.is_dynamic = True
                elif isinstance(slice_node, ast.Str):
                    s_val = slice_node.s
                    if isinstance(s_val, str):
                        self.output_keys.add(s_val)
                    else:
                        self.is_dynamic = True
                else:
                    self.is_dynamic = True
                self.visit(node.value)
                continue
            self.visit(target)
        self.visit(node.value)

    def visit_Dict(self, node: ast.Dict):
        for key, value in zip(node.keys, node.values):
            if key is None:
                if isinstance(value, ast.Name) and value.id == self.state_var_name:
                    self.is_dynamic = True
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return):
        if node.value is None:
            pass
        elif isinstance(node.value, ast.Dict):
            for key_node in node.value.keys:
                if isinstance(key_node, ast.Constant):
                    val = key_node.value
                    if isinstance(val, str):
                        self.output_keys.add(val)
                    else:
                        self.is_dynamic = True
                elif isinstance(key_node, ast.Str):
                    s_val = key_node.s
                    if isinstance(s_val, str):
                        self.output_keys.add(s_val)
                    else:
                        self.is_dynamic = True
                else:
                    self.is_dynamic = True
            for val_node in node.value.values:
                self.visit(val_node)
        else:
            self.is_dynamic = True
            self.visit(node.value)


class FuncFinder(ast.NodeVisitor):
    def __init__(self):
        self.node = None

    def visit_FunctionDef(self, node):
        if self.node is None:
            self.node = node

    def visit_Lambda(self, node):
        if self.node is None:
            self.node = node


def analyze_node_function(func: Any) -> Tuple[bool, Set[str], Set[str], bool]:
    """
    Statically analyzes a node function or callable using AST.
    Returns: (is_llm, input_keys, output_keys, is_dynamic)
    """
    try:
        source = inspect.getsource(func)
        source = textwrap.dedent(source)
        tree = ast.parse(source)
    except Exception:
        return False, set(), set(), True

    finder = FuncFinder()
    finder.visit(tree)
    func_node = finder.node
    if func_node is None:
        return False, set(), set(), True

    state_var_name = None
    if func_node.args.args:
        state_var_name = func_node.args.args[0].arg
    elif func_node.args.posonlyargs:
        state_var_name = func_node.args.posonlyargs[0].arg

    if not state_var_name:
        return False, set(), set(), True

    visitor = StateVisitor(state_var_name)
    visitor.visit(func_node)
    return visitor.is_llm, visitor.input_keys, visitor.output_keys, visitor.is_dynamic


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
                if "dataset_log_path" in cfg:
                    node.dataset_log_path = cfg["dataset_log_path"]
                if "redact_hook" in cfg:
                    node.redact_hook = cfg["redact_hook"]
                if "validation_log_path" in cfg:
                    node.validation_log_path = cfg["validation_log_path"]
            dspyer_graph.add_node(node)
        else:
            runnable = getattr(node_spec, "runnable", node_spec)
            func = getattr(runnable, "func", runnable)

            # Check if function is explicitly decorated
            if hasattr(func, "_dspyer_is_llm"):
                is_llm = getattr(func, "_dspyer_is_llm", True)
                input_model = getattr(func, "_dspyer_input_model", None) or make_permissive_model(
                    pydantic_state, f"{node_name}Input"
                )
                output_model = getattr(func, "_dspyer_output_model", None) or make_permissive_model(
                    pydantic_state, f"{node_name}Output"
                )
                instructions = (
                    getattr(func, "_dspyer_instructions", None)
                    or getattr(func, "__doc__", None)
                    or f"Execute agent step for node '{node_name}'."
                )
                if instructions:
                    instructions = instructions.strip()

                cfg = node_configs.get(node_name, {})
                node = StatefulNode(
                    name=node_name,
                    input_model=input_model,
                    output_model=output_model,
                    instructions=instructions,
                    max_retries=cfg.get("max_retries"),
                    refine_instructions=cfg.get("refine_instructions"),
                    use_cot=cfg.get("use_cot", False),
                    dataset_log_path=cfg.get("dataset_log_path"),
                    redact_hook=cfg.get("redact_hook"),
                    validation_log_path=cfg.get("validation_log_path"),
                )
                if not is_llm:
                    node.is_passthrough = True
                    node.callable = func
                else:
                    node._is_autogenerated = True
                dspyer_graph.add_node(node)
                continue

            is_llm, input_keys, output_keys, is_dynamic = analyze_node_function(func)

            if not is_llm:
                # Deterministic node: Execute as native Python passthrough
                permissive_input = make_permissive_model(pydantic_state, f"{node_name}Input")
                permissive_output = make_permissive_model(pydantic_state, f"{node_name}Output")
                cfg = node_configs.get(node_name, {})
                passthrough_node = StatefulNode(
                    name=node_name,
                    input_model=permissive_input,
                    output_model=permissive_output,
                    instructions=None,
                    max_retries=cfg.get("max_retries"),
                    refine_instructions=cfg.get("refine_instructions"),
                    use_cot=cfg.get("use_cot", False),
                    dataset_log_path=cfg.get("dataset_log_path"),
                    redact_hook=cfg.get("redact_hook"),
                    validation_log_path=cfg.get("validation_log_path"),
                )
                passthrough_node.is_passthrough = True
                passthrough_node.callable = func
                dspyer_graph.add_node(passthrough_node)
            else:
                if is_dynamic:
                    raise ValueError(
                        f"LLM-containing node '{node_name}' cannot be dynamically converted because it utilizes dynamic "
                        f"state accesses (e.g. dynamic keys, unpackings, or dynamic returns). "
                        f"Please map this node explicitly by providing a mapped dspyer.StatefulNode in 'node_mappings'."
                    )

                docstring = getattr(func, "__doc__", None)
                instructions = (
                    docstring.strip()
                    if docstring
                    else f"Execute agent step for node '{node_name}'."
                )

                import warnings

                warnings.warn(
                    f"Auto-generating LLM StatefulNode '{node_name}' with a dynamically derived narrow state schema. "
                    "This serves as a compiled DSPy scaffold stub and does not execute the original Python function body. "
                    "For a behavior-preserving execution or custom prompts, map this node explicitly using 'node_mappings'.",
                    UserWarning,
                    stacklevel=2,
                )

                input_fields: Dict[str, Any] = {}
                for key in input_keys:
                    if key in pydantic_state.model_fields:
                        field_info = pydantic_state.model_fields[key]
                        annotation = field_info.annotation or Any
                        input_fields[key] = (Optional[annotation], None)
                    else:
                        input_fields[key] = (Any, None)
                if not input_fields:
                    input_fields["placeholder"] = (Optional[str], "default")

                output_fields: Dict[str, Any] = {}
                for key in output_keys:
                    if key in pydantic_state.model_fields:
                        field_info = pydantic_state.model_fields[key]
                        annotation = field_info.annotation or Any
                        output_fields[key] = (Optional[annotation], None)
                    else:
                        output_fields[key] = (Any, None)
                if not output_fields:
                    output_fields["placeholder"] = (Optional[str], "default")

                narrow_input = create_model(f"{node_name}Input", **input_fields)
                narrow_output = create_model(f"{node_name}Output", **output_fields)

                cfg = node_configs.get(node_name, {})
                generated_node = StatefulNode(
                    name=node_name,
                    input_model=narrow_input,
                    output_model=narrow_output,
                    instructions=instructions,
                    max_retries=cfg.get("max_retries"),
                    refine_instructions=cfg.get("refine_instructions"),
                    use_cot=cfg.get("use_cot", False),
                    dataset_log_path=cfg.get("dataset_log_path"),
                    redact_hook=cfg.get("redact_hook"),
                    validation_log_path=cfg.get("validation_log_path"),
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
