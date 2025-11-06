def format_history_for_prompt(memory: dict, agent_name: str) -> str:
    """Return full chat history formatted as text."""
    hist = memory.get(agent_name, [])
    lines = []
    for m in hist:
        role = m.get("role", "user")
        prefix = "User:" if role == "user" else "Assistant:"
        lines.append(f"{prefix} {m.get('text','')}")
    return "\n".join(lines)
