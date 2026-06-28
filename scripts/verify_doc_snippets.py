import os
import re
import sys
from typing import Any, List


def extract_python_blocks(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Regex to find fenced python code blocks
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    blocks = pattern.findall(content)
    return blocks


def verify_doc_snippets():
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    print("==========================================================")
    # 1. Test README.md Quickstart blocks (concatenated)
    readme_path = os.path.join(workspace_dir, "README.md")
    print("Extracting and testing README.md Quickstart snippets...")
    readme_blocks = extract_python_blocks(readme_path)

    if len(readme_blocks) < 1:
        print(
            f"[FAIL] Expected at least 1 python block in README.md, found {len(readme_blocks)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # The new README.md has the entire runnable quickstart in the first block
    readme_tutorial = readme_blocks[0]

    try:
        # Use the same dictionary for globals and locals to resolve class scoping lookups correctly
        readme_ns: dict[str, Any] = {}
        exec(readme_tutorial, readme_ns, readme_ns)
        print("[PASS] README.md Quickstart snippets executed successfully.")
    except Exception as e:
        print(f"[FAIL] README.md Quickstart snippets failed: {e}", file=sys.stderr)
        print("\n--- Failed Snippet Content ---", file=sys.stderr)
        print(readme_tutorial, file=sys.stderr)
        print("------------------------------\n", file=sys.stderr)
        sys.exit(1)

    # 2. Test docs/getting-started.md concatenated blocks
    getting_started_path = os.path.join(workspace_dir, "docs", "getting-started.md")
    print("Extracting and testing docs/getting-started.md concatenated snippets...")

    gs_blocks = extract_python_blocks(getting_started_path)
    if len(gs_blocks) < 3:
        print(
            f"[FAIL] Expected at least 3 python blocks in getting-started.md, found {len(gs_blocks)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Concatenate the first 3 blocks (Schemas, Graph Configuration, Run Program)
    gs_tutorial = "\n".join(gs_blocks[:3])

    try:
        gs_ns: dict[str, Any] = {}
        exec(gs_tutorial, gs_ns, gs_ns)
        print("[PASS] docs/getting-started.md tutorial snippets executed successfully.")
    except Exception as e:
        print(f"[FAIL] docs/getting-started.md tutorial snippets failed: {e}", file=sys.stderr)
        print("\n--- Failed Snippet Content ---", file=sys.stderr)
        print(gs_tutorial, file=sys.stderr)
        print("------------------------------\n", file=sys.stderr)
        sys.exit(1)

    print("\n[SUCCESS] All user-facing documentation code snippets verified and passed.")
    sys.exit(0)


if __name__ == "__main__":
    verify_doc_snippets()
