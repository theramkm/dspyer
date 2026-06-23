import os
import re
import sys


def verify_links():
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    markdown_files = []

    # Gather all markdown files
    for root, dirs, files in os.walk(workspace_dir):
        # Skip virtualenvs and git folders
        if any(
            x in root for x in [".venv", "venv", ".git", "site", ".mypy_cache", ".pytest_cache"]
        ):
            continue
        for file in files:
            if file.endswith(".md"):
                markdown_files.append(os.path.join(root, file))

    link_pattern = re.compile(r"\[([^\]]+)\]\(([^\)]+)\)")
    errors = []

    print(f"Scanning {len(markdown_files)} markdown files for link verification...")

    for file_path in markdown_files:
        rel_file_path = os.path.relpath(file_path, workspace_dir)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Warning: Could not read {rel_file_path}: {e}")
            continue

        for line_num, line in enumerate(lines, 1):
            matches = link_pattern.findall(line)
            for label, url in matches:
                # 1. Check for absolute local file:// leaks
                if "file://" in url:
                    errors.append(
                        f"{rel_file_path}:{line_num} - Leak error: Found absolute local link '{url}'"
                    )
                    continue

                # 2. Skip web URLs
                if url.startswith(("http://", "https://", "mailto:", "#")):
                    continue

                # 3. Handle relative file paths
                # Strip query/anchor parameters
                clean_url = url.split("#")[0].split("?")[0]
                if not clean_url:
                    continue

                # Resolve relative path
                file_dir = os.path.dirname(file_path)
                target_path = os.path.abspath(os.path.join(file_dir, clean_url))

                # Check if target exists on disk
                if not os.path.exists(target_path):
                    errors.append(
                        f"{rel_file_path}:{line_num} - Broken link error: Target '{clean_url}' (resolved to '{os.path.relpath(target_path, workspace_dir)}') does not exist"
                    )

    if errors:
        print("\n[FAIL] Link verification failed with the following errors:\n", file=sys.stderr)
        for err in errors:
            print(f"  * {err}", file=sys.stderr)
        sys.exit(1)

    print(
        "\n[SUCCESS] All links verified successfully. Zero absolute leaks or broken relative links found."
    )
    sys.exit(0)


if __name__ == "__main__":
    verify_links()
