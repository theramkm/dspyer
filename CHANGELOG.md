# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.4] - 2026-06-23

### Changed
- **Package Rename & Unified Import Name**: Renamed the package directory from `dspyer` to `dspyer`. This aligns the distribution name with the import name, ensuring `import dspyer` works successfully upon installation.
- **Curated Public Namespace**: Restructured [dspyer/__init__.py](dspyer/__init__.py) to expose only the four core package entry points (`AgentTranspiler`, `from_langgraph`, `self_correcting`, and `dspyer_node`) at the root level. All advanced utilities and state structures are kept inside their respective submodules to reduce public API pollution.
- **Import Realignment**: Updated all tests, examples, and documentation to import from the new `dspyer` package namespace and submodule paths.

## [0.3.3] - 2026-06-23

### Added
- **Extensive Getting Started Documentation**: Created [docs/getting-started.md](docs/getting-started.md) showing offline mock LM testing configurations and citation synthesizer graphs.
- **Detailed State Management Guide**: Added [docs/state.md](docs/state.md) to document Copy-on-Write (COW) optimization math models and list branch conflict resolution policies.
- **Custom Decorator Walkthrough**: Added [docs/decorators.md](docs/decorators.md) explaining `@self_correcting` wrapper scopes and `@dspyer_node` escape hatches.
- **Async & Streaming Guide**: Added [docs/async-streaming.md](docs/async-streaming.md) showing how to leverage `aforward` and `astream` execution pipelines inside ASGI web environments.
- **Storage & Observability Guide**: Added [docs/storage.md](docs/storage.md) detailing custom `BaseStorageAdapter` database sinks, validation reports, and flywheel datasets.
- **Reference Docstrings**: Added comprehensive developer docstrings for async pipelines [dspyer/compiler.py](dspyer/compiler.py) and [dspyer/compiler.py](dspyer/compiler.py) in the codebase.

## [0.3.2] - 2026-06-23

### Added
- **Async Execution Pipeline**: Added native support for asynchronous execution via [dspyer/compiler.py](dspyer/compiler.py) on the compiled program to run non-blocking flows inside web frameworks.
- **Event Streaming**: Introduced [dspyer/compiler.py](dspyer/compiler.py) step event streaming for tracking real-time execution states and streaming tokens.
- **Decorator Metadata Bypasses**: Implemented [@dspyer_node](dspyer/decorator.py) decorator to let developers explicitly declare input/output schemas and override instructions on node functions, bypassing AST static parsing.
- **Pluggable Storage Logging**: Exposed [dspyer/utils.py](dspyer/utils.py) to enable registering custom database or filesystem backends for logging. Default behavior falls back to a thread-safe, thread-pooled [dspyer/utils.py](dspyer/utils.py).
- **Customizable Error Formatting**: Added customizable `error_formatter` callback support in the compiler options to format model correction prompts.
- **Automated Async Coverage**: Created [tests/test_async_execution.py](tests/test_async_execution.py) verifying async forwards, streaming generators, decorator bypasses, and pluggable storage.

### Changed
- **Copy-on-Write State Optimization**: Optimized state manipulation inside [dspyer/state.py](dspyer/state.py) to use a Copy-on-Write (COW) merge algorithm, copying dictionaries only when keys are modified.
- **Generic Compiler Return Types**: Refactored the [dspyer/compiler.py](dspyer/compiler.py) signature to support generic output typing for type checking and autocomplete in editors.

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
