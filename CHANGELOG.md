# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.3] - 2026-06-23

### Added
- **Extensive Getting Started Documentation**: Created [getting-started.md](file:///Users/ram/play/dspyer/docs/getting-started.md) showing offline mock LM testing configurations and citation synthesizer graphs.
- **Detailed State Management Guide**: Added [state.md](file:///Users/ram/play/dspyer/docs/state.md) to document Copy-on-Write (COW) optimization math models and list branch conflict resolution policies.
- **Custom Decorator Walkthrough**: Added [decorators.md](file:///Users/ram/play/dspyer/docs/decorators.md) explaining `@self_correcting` wrapper scopes and `@dspyer_node` escape hatches.
- **Async & Streaming Guide**: Added [async-streaming.md](file:///Users/ram/play/dspyer/docs/async-streaming.md) showing how to leverage `aforward` and `astream` execution pipelines inside ASGI web environments.
- **Storage & Observability Guide**: Added [storage.md](file:///Users/ram/play/dspyer/docs/storage.md) detailing custom `BaseStorageAdapter` database sinks, validation reports, and flywheel datasets.
- **Reference Docstrings**: Added comprehensive developer docstrings for async pipelines [aforward](file:///Users/ram/play/dspyer/dspy_transpiler/compiler.py) and [astream](file:///Users/ram/play/dspyer/dspy_transpiler/compiler.py) in the codebase.

## [0.3.2] - 2026-06-23

### Added
- **Async Execution Pipeline**: Added native support for asynchronous execution via [aforward](file:///Users/ram/play/dspyer/dspy_transpiler/compiler.py) on the compiled program to run non-blocking flows inside web frameworks.
- **Event Streaming**: Introduced [astream](file:///Users/ram/play/dspyer/dspy_transpiler/compiler.py) step event streaming for tracking real-time execution states and streaming tokens.
- **Decorator Metadata Bypasses**: Implemented [@dspyer_node](file:///Users/ram/play/dspyer/dspy_transpiler/decorator.py) decorator to let developers explicitly declare input/output schemas and override instructions on node functions, bypassing AST static parsing.
- **Pluggable Storage Logging**: Exposed [BaseStorageAdapter](file:///Users/ram/play/dspyer/dspy_transpiler/utils.py) to enable registering custom database or filesystem backends for logging. Default behavior falls back to a thread-safe, thread-pooled [FileStorageAdapter](file:///Users/ram/play/dspyer/dspy_transpiler/utils.py).
- **Customizable Error Formatting**: Added customizable `error_formatter` callback support in the compiler options to format model correction prompts.
- **Automated Async Coverage**: Created [test_async_execution.py](file:///Users/ram/play/dspyer/tests/test_async_execution.py) verifying async forwards, streaming generators, decorator bypasses, and pluggable storage.

### Changed
- **Copy-on-Write State Optimization**: Optimized state manipulation inside [ImmutableState](file:///Users/ram/play/dspyer/dspy_transpiler/state.py) to use a Copy-on-Write (COW) merge algorithm, copying dictionaries only when keys are modified.
- **Generic Compiler Return Types**: Refactored the [TranspiledAgentProgram](file:///Users/ram/play/dspyer/dspy_transpiler/compiler.py) signature to support generic output typing for type checking and autocomplete in editors.

## [0.3.1] - 2026-06-22

### Added
- Standard Python packaging metadata, URLs, and PEP 561 type checking indicators.
- Automated release workflows via GitHub Actions using PyPI trusted publishing.

## [0.3.0] - 2026-06-21

### Added
- Standalone self-correcting wrapper decorator and OpenTelemetry (Arize Phoenix) logging telemetry.
- Self-correction to training dataset flywheel.
- Dynamic LangGraph converter support (`from_langgraph`).
- Namespace protection for validation control parameters.
