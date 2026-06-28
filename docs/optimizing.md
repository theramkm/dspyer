# Optimizing Prompts

Prompt engineering is often a manual process of trial and error. The primary benefit of transpiling your workflows into DSPy modules with `dspyer` is that it allows you to optimize your prompts programmatically using standard DSPy optimizers (teleprompters).

This guide walks through defining evaluation metrics, compiling (optimizing) your steps against a dataset, saving/loading the tuned instructions, and bootstrapping from self-correction logs.

---

## 1. How Optimization Works

When you define a self-correcting function using `@self_correcting` or compile a stateful graph using `AgentTranspiler.compile(graph)`, `dspyer` generates a standard `dspy.Module` underneath. This module exposes prompt instructions and prefix fields as trainable parameters.

Instead of manually editing prompt strings when changing models or tasks, you provide a dataset of inputs/outputs and an evaluation metric. The DSPy optimizer then runs, scores outputs, selects successful demonstrations, and constructs optimized prompts.

---

## 2. Defining an Evaluation Metric

An evaluation metric is a standard Python function that takes a ground-truth dataset example and the model's prediction, and returns a score (usually a boolean `True`/`False` or a float score between `0.0` and `1.0`).

```python
def sentiment_metric(example, pred, trace=None) -> bool:
    # Compare the predicted sentiment value with ground truth
    is_correct = example.sentiment.lower() == pred.sentiment.lower()
    return is_correct
```

---

## 3. Running the Optimizer

To run an optimizer (like `BootstrapFewShot`), you compile your program using a small training dataset:

```python
import dspy
from dspy.teleprompt import BootstrapFewShot
from dspyer import self_correcting

# 1. Prepare your dataset
trainset = [
    dspy.Example(query="This product is amazing!", sentiment="Positive").with_inputs("query"),
    dspy.Example(query="It broke on the first day.", sentiment="Negative").with_inputs("query"),
]

# 2. Wrap your target function
@self_correcting(max_retries=3)
def analyze_sentiment(query: str) -> SentimentOutput:
    """Classify the sentiment of the user query."""
    pass

# 3. Configure the optimizer
optimizer = BootstrapFewShot(metric=sentiment_metric, max_bootstrapped_demos=2)

# 4. Compile (optimize) the program
optimized_program = optimizer.compile(analyze_sentiment, trainset=trainset)
```

---

## 4. Saving & Loading Tuned Prompts

Once optimized, you do not need to repeat the tuning process in production. You can serialize the compiled instructions and prefix configurations to a JSON file:

### Saving Prompts

```python
# Save the optimized prompt configurations to disk
optimized_program.save_prompts("prompts_config.json")
```

### Loading Prompts in Production

In your production application, instantiate the clean module/decorator and load the saved configuration directly:

```python
# Load the configuration directly into your production module instance
analyze_sentiment.load_prompts("prompts_config.json")

# Now, calls will utilize the optimized prompt templates automatically
result = analyze_sentiment(query="This is a great library!")
```

---

## 5. The Self-Correction Dataset Flywheel

Self-correction loops are excellent at correcting errors dynamically, but they add execution latency and token costs. You can capture successful self-correction runs and convert them into permanent training examples.

This creates a **flywheel**: the model's own runtime self-corrections are collected to train it, so it learns to succeed on the *first* attempt in the future.

### Recording Successful Retries

Pass the `dataset_log_path` parameter to compile or decorate:

```python
# Logs inputs and final valid outputs only when retries/self-corrections succeed
program = AgentTranspiler.compile(graph, dataset_log_path="logs/flywheel.jsonl")
```

### Loading and Replaying

Load the logged examples into a standard `dspy.Example` list, ready to be passed directly to your optimizer:

```python
from dspyer.utils import load_logged_dataset

# We must declare which fields serve as inputs to the model
trainset = load_logged_dataset(
    dataset_log_path="logs/flywheel.jsonl",
    input_keys=["query"]
)

# Run prompt optimization using the captured self-correction examples
from dspy.teleprompt import BootstrapFewShot
optimizer = BootstrapFewShot(metric=sentiment_metric)
optimized_module = optimizer.compile(program, trainset=trainset)
```
