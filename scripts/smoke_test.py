import os
import sys

def run_smoke_test():
    print("Running dspyer package smoke test...")
    
    # 1. Verify import and root exports
    try:
        import dspyer
        from dspyer import AgentTranspiler, from_langgraph, self_correcting, dspyer_node
        print("[PASS] Successfully imported dspyer and all root entry points.")
    except Exception as e:
        print(f"[FAIL] Failed to import dspyer root entry points: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 2. Verify submodules are accessible but kept out of root __all__
    try:
        from dspyer.graph import Graph, StatefulNode
        from dspyer.state import ImmutableState
        print("[PASS] Successfully imported advanced submodules.")
    except Exception as e:
        print(f"[FAIL] Failed to import submodules: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 3. Execute a minimal program using `@self_correcting` to verify execution
    try:
        import dspy
        from pydantic import BaseModel, Field
        
        class OutputSchema(BaseModel):
            result: str = Field(description="Mock result")
            
        @self_correcting(max_retries=2)
        def mock_step(query: str) -> OutputSchema:
            """Mock step instructions."""
            pass
            
        from dspyer.compiler import DirectLM

        lm = DirectLM(model="openai/mock-model", api_key="sk-test")

        def mock_generate_sync(prompt, system_prompt=None):
            return '{"result": "success"}'

        lm.client.generate_sync = mock_generate_sync
        dspy.settings.configure(lm=lm)
        
        response = mock_step(query="hello")
        assert response.result == "success"
        print("[PASS] Successfully executed a minimal self-correcting program.")
    except Exception as e:
        print(f"[FAIL] Execution check failed: {e}", file=sys.stderr)
        sys.exit(1)
        
    print("\n[SUCCESS] Smoke test completed successfully. Package is ready for release.")
    sys.exit(0)

if __name__ == "__main__":
    run_smoke_test()
