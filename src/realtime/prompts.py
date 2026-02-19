from __future__ import annotations


def build_instructions(person_name: str, person_prompt: str | None = None) -> str:
    base = (
        "You are a concise voice assistant for a security kiosk. "
        f"The recognized person is {person_name}. "
        "Greet them naturally by name, answer briefly, and keep responses clear."
    )
    if not person_prompt:
        return base
    return f"{base}\n\nPerson-specific guidance:\n{person_prompt}"
