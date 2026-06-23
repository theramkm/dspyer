import argparse
import os
import sys

# Allow direct script execution from subdirectories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dspyer.compiler import DirectClient


def main():
    parser = argparse.ArgumentParser(
        description="Run raw LLM queries directly using the zero-dependency DirectClient wrapper."
    )
    parser.add_argument(
        "--provider",
        default=os.environ.get("DSPYER_PROVIDER"),
        help="Model provider (e.g. google, anthropic, openai, ollama). Read from DSPYER_PROVIDER environment variable if not specified.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("DSPYER_MODEL"),
        help="Model name (e.g. gemini-3.5-flash, claude-4.8, gpt-5.5, llama3). Read from DSPYER_MODEL environment variable if not specified.",
    )
    parser.add_argument(
        "--prompt",
        default="Write a 1-sentence welcome message for the dspyer agent library.",
        help="Prompt text to query the model with.",
    )
    parser.add_argument(
        "--system", default="You are a helpful assistant.", help="Optional system instructions."
    )

    args = parser.parse_args()

    # Enforce parameters if not set by CLI or environment
    if not args.provider or not args.model:
        print(
            "Error: Both --provider and --model must be specified via CLI arguments "
            "or DSPYER_PROVIDER and DSPYER_MODEL environment variables.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[*] Initializing direct client: {args.provider}/{args.model}...")
    client = DirectClient(provider=args.provider, model=args.model)

    print(f"[*] Querying model direct endpoint with prompt: {repr(args.prompt)}")
    try:
        response = client.generate_sync(prompt=args.prompt, system_prompt=args.system)
        print("\n[+] Direct Client response received:")
        print(response)
    except Exception as run_err:
        print(f"\n[-] Direct execution failed: {run_err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
