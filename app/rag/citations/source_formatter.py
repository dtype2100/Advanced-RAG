"""Source formatter — renders citation lists into human-readable strings."""

from __future__ import annotations


def format_sources(citations: list[dict[str, str]], style: str = "numbered") -> str:
    """Format a list of citation dicts into a readable string.

    Args:
        citations: List of ``{source, page, score}`` dicts.
        style:     ``"numbered"`` (default) or ``"inline"``.

    Returns:
        Formatted source string.
    """
    if not citations:
        return "No sources available."

    if style == "inline":
        parts = []
        for c in citations:
            ref = c["source"]
            if c.get("page"):
                ref += f", p.{c['page']}"
            parts.append(f"[{ref}]")
        return " ".join(parts)

    lines = ["Sources:"]
    for i, c in enumerate(citations, start=1):
        line = f"  {i}. {c['source']}"
        if c.get("page"):
            line += f", page {c['page']}"
        lines.append(line)
    return "\n".join(lines)
