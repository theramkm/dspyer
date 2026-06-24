import os
import re


def make_links_absolute():
    workspace_dir = "/Users/ram/play/dspyer"
    exclude_dirs = {".git", ".venv", ".mypy_cache", ".pytest_cache", "site", "__pycache__"}

    modified_files = []

    # We want to replace:
    # 1. ../dspyer/ -> https://github.com/theramkm/dspyer/blob/main/dspyer/
    # 2. dspyer/ -> https://github.com/theramkm/dspyer/blob/main/dspyer/
    # 3. ../tests/ -> https://github.com/theramkm/dspyer/blob/main/tests/
    # 4. tests/ -> https://github.com/theramkm/dspyer/blob/main/tests/

    replacements = [
        (r"\]\(\.\./dspyer/", "](https://github.com/theramkm/dspyer/blob/main/dspyer/"),
        (r"\]\(dspyer/", "](https://github.com/theramkm/dspyer/blob/main/dspyer/"),
        (r"\]\(\.\./tests/", "](https://github.com/theramkm/dspyer/blob/main/tests/"),
        (r"\]\(tests/", "](https://github.com/theramkm/dspyer/blob/main/tests/"),
    ]

    for root, dirs, files in os.walk(workspace_dir):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception as e:
                    print(f"Skipping {file_path}: {e}")
                    continue

                new_content = content
                changed = False
                for pattern, repl in replacements:
                    if re.search(pattern, new_content):
                        new_content = re.sub(pattern, repl, new_content)
                        changed = True

                if changed:
                    try:
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(new_content)
                        modified_files.append(file_path)
                        print(f"Updated links in: {file_path}")
                    except Exception as e:
                        print(f"Error writing to {file_path}: {e}")

    print(f"Link conversion complete. Total files modified: {len(modified_files)}")


if __name__ == "__main__":
    make_links_absolute()
