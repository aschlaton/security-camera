#!/usr/bin/env python3
"""Create folder structure for a new person under people/. Usage: add_person.py <name>"""

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: add_person.py <name>", file=sys.stderr)
        return 1
    name = sys.argv[1].strip()
    if not name:
        print("Name cannot be empty.", file=sys.stderr)
        return 1

    root = Path(__file__).resolve().parent.parent
    person_dir = root / "people" / name
    pictures_dir = person_dir / "pictures"

    if person_dir.exists():
        print(f"Error: people/{name}/ already exists.", file=sys.stderr)
        return 1

    person_dir.mkdir(parents=True)
    pictures_dir.mkdir()
    (person_dir / "prompt.md").write_text("")

    print(f"Created {person_dir}/, {pictures_dir}/, and people/{name}/prompt.md")
    print(f"Add face images in people/{name}/pictures/ and edit people/{name}/prompt.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
