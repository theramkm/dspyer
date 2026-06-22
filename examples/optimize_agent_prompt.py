import os
import sys

from pydantic import BaseModel, Field

# Allow direct script execution from subdirectories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dspy
from dspy.teleprompt import BootstrapFewShot

from dspy_transpiler.compiler import AgentTranspiler
from dspy_transpiler.graph import Graph, StatefulNode


# 1. Define Step Schemas
class TranslationInput(BaseModel):
    english_text: str = Field(description="The source text in English")


class TranslationOutput(BaseModel):
    french_text: str = Field(description="The translated text in French")


# 2. Build a local mock LLM client to support few-shot bootsrapping
class OptimizerMockLM(dspy.LM):
    def __init__(self):
        super().__init__(model="mock-optimizer-model")

    def forward(self, prompt=None, messages=None, **kwargs):
        prompt_str = str(prompt or messages)

        # Basic translation simulations
        if "Hello, how are you?" in prompt_str:
            content = '{"french_text": "Bonjour, comment ça va ?"}'
        elif "Thank you very much" in prompt_str:
            content = '{"french_text": "Merci beaucoup"}'
        elif "Good morning" in prompt_str:
            content = '{"french_text": "Bonjour"}'
        else:
            content = '{"french_text": "Salut"}'

        class MockChoiceMessage:
            def __init__(self, content_str: str):
                self.content = content_str
                self.role = "assistant"
                self.reasoning_content = None

        class MockChoice:
            def __init__(self, content_str: str):
                self.message = MockChoiceMessage(content_str)
                self.finish_reason = "stop"
                self.index = 0

        class MockResult:
            def __init__(self, content_str: str):
                self.choices = [MockChoice(content_str)]
                self.model = "mock-optimizer-model"
                self.usage = {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}

        return MockResult(content)


def main():
    print("[*] Configuring local OptimizerMockLM simulator...")
    lm = OptimizerMockLM()
    dspy.configure(lm=lm)

    # 3. Create a stateful translation graph node
    translation_node = StatefulNode(
        name="Translator",
        input_model=TranslationInput,
        output_model=TranslationOutput,
        instructions="Translate the provided English text into French.",
    )

    graph = Graph()
    graph.add_node(translation_node)
    graph.set_entry_point("Translator")

    # 4. Compile to declarative DSPy Module
    print("[*] Compiling graph into declarative DSPy Module...")
    program = AgentTranspiler.compile(graph)

    # 5. Define Training Dataset for few-shot learning
    # In DSPy, input fields must be flagged with .with_inputs()
    trainset = [
        dspy.Example(
            english_text="Hello, how are you?", french_text="Bonjour, comment ça va ?"
        ).with_inputs("english_text"),
        dspy.Example(english_text="Thank you very much", french_text="Merci beaucoup").with_inputs(
            "english_text"
        ),
    ]

    # 6. Define a metric function to validate optimization progress
    def exact_match_metric(example, pred, trace=None) -> bool:
        return example.french_text.strip().lower() == pred.french_text.strip().lower()

    # 7. Setup and run the BootstrapFewShot prompt optimizer
    print("[*] Running BootstrapFewShot teleprompter to optimize prompt weights...")
    optimizer = BootstrapFewShot(metric=exact_match_metric, max_bootstrapped_demos=2)
    optimized_program = optimizer.compile(program, trainset=trainset)

    print("\n[+] Optimization compilation completed successfully!")
    print("\n[*] Inspecting optimized program predictors...")
    for name, predictor in optimized_program.named_predictors():
        print(f"  Predictor: {name}")
        # Show that bootstrapped few-shot examples were injected into the optimized prompt signature
        demos = predictor.demos if hasattr(predictor, "demos") else []
        print(f"    Number of bootstrapped few-shot demos: {len(demos)}")
        for i, demo in enumerate(demos):
            print(
                f"      Demo {i + 1}: English: '{demo.english_text}' -> French: '{demo.french_text}'"
            )

    # 8. Test execution of the optimized program
    print("\n[*] Running optimized program on test input: 'Good morning'...")
    res = optimized_program(english_text="Good morning")
    print(f"  Result French Text: '{res.french_text}'")


if __name__ == "__main__":
    main()
