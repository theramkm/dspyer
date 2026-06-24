# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.5] - 2026-06-24

### Added
- **First-Class Async Decorator Support**: Added native support for asynchronous (`async def`) functions to the `@self_correcting` decorator. It returns a proper coroutine and executes the underlying DSPy predictor in a separate thread pool using `asyncio.to_thread` to prevent event loop blockages, making it fully production-ready for ASGI web frameworks (like FastAPI).
- **Automated Fenced Snippet Verifier**: Developed [scripts/verify_doc_snippets.py](https://github.com/theramkm/dspyer/blob/main/scripts/verify_doc_snippets.py) to extract, compile, and execute Python code blocks in `README.md` and `docs/getting-started.md` under mock conditions, preventing documentation examples from rotting.
- **Strict Quality Gates in CI/CD**: Integrated the document snippets verifier and strict MkDocs builds (`--strict` flag) directly into the main PR/push CI workflow (`ci.yml`) and pre-release validation gates (`run_release_check.sh`), catching any documentation regressions immediately on every commit.
- **Documentation Badge**: Added a high-visibility, custom documentation badge at the top of the repository `README.md` linking directly to the published GitHub Pages site.
- **Clear Import Guards for Extras**: Added helpful import guards using `importlib.util.find_spec` in `examples/benchmark.py` and `examples/optimize_compiled_graph.py` to print clear, user-friendly instructions to install `dspyer[langgraph]` instead of throwing raw tracebacks on base package installations.

### Changed
- **Curated Public Namespace Re-expansion**: Re-expanded `dspyer/__init__.py` root exports to include all major user-facing classes (`Graph`, `StatefulNode`, `ImmutableState`, `DirectLM`, `DirectClient`, `MockCompletionResult`, and various storage adapters) to resolve import errors in the getting-started tutorial.
- **Absolute Link Purification**: Converted all relative source code references (e.g., `../dspyer/compiler.py` pointing outside the docs folder) to absolute GitHub blob URLs, resolving 404 errors on the published Pages site.
- **Provider-Agnostic Quickstart**: Redesigned `examples/quickstart.py` to be completely provider-neutral by dropping the hardcoded OpenAI API key check, allowing it to run out-of-the-box with local Ollama, Google Gemini, Anthropic Claude, or OpenAI.
- **Colab Notebook Setup**: Updated the Colab playground notebook (`notebooks/dspyer_playground.ipynb`) to install from the official PyPI release instead of a slow, pre-release Git URL.

### Fixed
- **Homepage HTML Rendering Bug**: Resolved the Python-Markdown parsing bug on the docs homepage by enabling the `md_in_html` extension in `mkdocs.yml` and adding `markdown="1"` to the README's centered `div` container.
- **MkDocs CLI Build Error**: Resolved the docs deployment workflow failure by updating `docs.yml` to run using the correct `docs` dependency group containing the `mkdocs` packages.
- **Griffe Signature Warnings**: Resolved strict documentation build warnings by adding proper type annotations (`**initial_state_kwargs: Any`) to the compiler's `forward`, `aforward`, and `astream` methods.

## [0.3.4] - 2026-06-23

### Changed
- **Package Rename & Unified Import Name**: Renamed the package directory from `dspyer` to `dspyer`. This aligns the distribution name with the import name, ensuring `import dspyer` works successfully upon installation.
- **Curated Public Namespace**: Restructured [dspyer/__init__.py](https://github.com/theramkm/dspyer/blob/main/dspyer/__init__.py) to expose only the four core package entry points (`AgentTranspiler`, `from_langgraph`, `self_correcting`, and `dspyer_node`) at the root level. All advanced utilities and state structures are kept inside their respective submodules to reduce public API pollution.
- **Import Realignment**: Updated all tests, examples, and documentation to import from the new `dspyer` package namespace and submodule paths.

## [0.3.3] - 2026-06-23

### Added
- **Extensive Getting Started Documentation**: Created [docs/getting-started.md](docs/getting-started.md) showing offline mock LM testing configurations and citation synthesizer graphs.
- **Detailed State Management Guide**: Added [docs/state.md](docs/state.md) to document Copy-on-Write (COW) optimization math models and list branch conflict resolution policies.
- **Custom Decorator Walkthrough**: Added [docs/decorators.md](docs/decorators.md) explaining `@self_correcting` wrapper scopes and `@dspyer_node` escape hatches.
- **Async & Streaming Guide**: Added [docs/async-streaming.md](docs/async-streaming.md) showing how to leverage `aforward` and `astream` execution pipelines inside ASGI web environments.
- **Storage & Observability Guide**: Added [docs/storage.md](docs/storage.md) detailing custom `BaseStorageAdapter` database sinks, validation reports, and flywheel datasets.
- **Reference Docstrings**: Added comprehensive developer docstrings for async pipelines [dspyer/compiler.py](https://github.com/theramkm/dspyer/blob/main/dspyer/compiler.py) and [dspyer/compiler.py](https://github.com/theramkm/dspyer/blob/main/dspyer/compiler.py) in the codebase.

## [0.3.2] - 2026-06-23

### Added
- **Async Execution Pipeline**: Added native support for asynchronous execution via [dspyer/compiler.py](https://github.com/theramkm/dspyer/blob/main/dspyer/compiler.py) on the compiled program to run non-blocking flows inside web frameworks.
- **Event Streaming**: Introduced [dspyer/compiler.py](https://github.com/theramkm/dspyer/blob/main/dspyer/compiler.py) step event streaming for tracking real-time execution states and streaming tokens.
- **Decorator Metadata Bypasses**: Implemented [@dspyer_node](https://github.com/theramkm/dspyer/blob/main/dspyer/decorator.py) decorator to let developers explicitly declare input/output schemas and override instructions on node functions, bypassing AST static parsing.
- **Pluggable Storage Logging**: Exposed [dspyer/utils.py](https://github.com/theramkm/dspyer/blob/main/dspyer/utils.py) to enable registering custom database or filesystem backends for logging. Default behavior falls back to a thread-safe, thread-pooled [dspyer/utils.py](https://github.com/theramkm/dspyer/blob/main/dspyer/utils.py).
- **Customizable Error Formatting**: Added customizable `error_formatter` callback support in the compiler options to format model correction prompts.
- **Automated Async Coverage**: Created [tests/test_async_execution.py](https://github.com/theramkm/dspyer/blob/main/tests/test_async_execution.py) verifying async forwards, streaming generators, decorator bypasses, and pluggable storage.

### Changed
- **Copy-on-Write State Optimization**: Optimized state manipulation inside [dspyer/state.py](https://github.com/theramkm/dspyer/blob/main/dspyer/state.py) to use a Copy-on-Write (COW) merge algorithm, copying dictionaries only when keys are modified.
- **Generic Compiler Return Types**: Refactored the [dspyer/compiler.py](https://github.com/theramkm/dspyer/blob/main/dspyer/compiler.py) signature to support generic output typing for type checking and autocomplete in editors.

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
