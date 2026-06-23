import os
import sys

from pydantic import BaseModel, Field, field_validator

# Allow direct script execution from subdirectories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dspy

from dspyer.compiler import AgentTranspiler
from dspyer.graph import Graph, StatefulNode


# 1. Define strict Pydantic schemas for the workflow
class UserQuery(BaseModel):
    query: str = Field(description="The query asked by the user")


class RetrievedDocuments(BaseModel):
    query: str = Field(description="The user's query")
    context: str = Field(description="The retrieved chunks and background context")


class RAGResponse(BaseModel):
    answer: str = Field(description="The synthesized answer referencing the context")
    citations: list[str] = Field(
        description="List of source document names cited in the answer (e.g. ['doc_1'])"
    )

    @field_validator("citations")
    @classmethod
    def must_have_citations(cls, val):
        if not val or len(val) == 0:
            raise ValueError("Synthesized answer must contain at least one valid source citation.")
        return val


# 2. Build a local mock LLM client to simulate a validation failure followed by a correction
class RAGMockLM(dspy.LM):
    def __init__(self):
        super().__init__(model="mock-rag-model")
        self.call_count = 0

    def forward(self, prompt=None, messages=None, **kwargs):
        self.call_count += 1
        prompt_str = str(prompt or messages)

        # Identify which node/stage is executing based on schema fields in prompt/messages
        if "context" in prompt_str and "citations" not in prompt_str:
            # Retriever node
            content = (
                '{"query": "What is the license of dspyer?", '
                '"context": "[doc_1] dspyer is released under the Apache-2.0 License. '
                '[doc_2] Previous versions were licensed under GPL-3.0."}'
            )
        elif "citations" in prompt_str:
            if "failed_output" in prompt_str or "feedback" in prompt_str:
                # Refiner correction pass: provide valid citations
                content = (
                    '{"answer": "dspyer is licensed under the Apache-2.0 License as of 2026 '
                    'according to [doc_1].", "citations": ["doc_1"]}'
                )
            else:
                # First pass: fail validation by returning empty citations list
                content = (
                    '{"answer": "dspyer is licensed under the Apache-2.0 License.", '
                    '"citations": []}'
                )
        else:
            content = '{"error": "unknown prompt"}'

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
                self.model = "mock-rag-model"
                self.usage = {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}

        return MockResult(content)


def main():
    print("[*] Configuring local RAGMockLM simulator...")
    lm = RAGMockLM()
    dspy.configure(lm=lm)

    # 3. Declare nodes with input/output boundaries and natural language instructions
    node_retriever = StatefulNode(
        name="DocRetriever",
        input_model=UserQuery,
        output_model=RetrievedDocuments,
        instructions="Retrieve relevant documents and text chunks for the query.",
    )

    node_synthesizer = StatefulNode(
        name="AnswerSynthesizer",
        input_model=RetrievedDocuments,
        output_model=RAGResponse,
        instructions="Synthesize a cited answer based on the retrieved context.",
    )

    # 4. Define graph execution flow
    graph = Graph()
    graph.add_node(node_retriever)
    graph.add_node(node_synthesizer)
    graph.set_entry_point("DocRetriever")
    graph.add_edge("DocRetriever", "AnswerSynthesizer")

    # 5. Compile into a standard, optimizable DSPy program
    print("[*] Compiling RAG workflow into declarative DSPy Module...")
    program = AgentTranspiler.compile(graph)

    # 6. Run execution
    print("\n[+] Running RAG verifier workflow...")
    result = program(query="What is the license of dspyer?")

    print("\n[+] Execution completed successfully!")
    print("Answer:   ", result.answer)
    print("Citations:", result.citations)
    print("Self-correction loops:", result["_metadata"]["refinement_steps_taken"])


if __name__ == "__main__":
    main()
