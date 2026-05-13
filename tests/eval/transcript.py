"""Clean transcripts for LLM consumption — strip ANSI, extract meaningful content."""

import re

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text):
    return _ANSI_RE.sub("", text)


def clean_transcript(round_num, prompt, commands, output, agent_text):
    """Convert raw agent output to a clean transcript entry."""
    return {
        "round": round_num,
        "user": prompt,
        "agent_commands": [c["command"] if isinstance(c, dict) else c for c in (commands or [])],
        "command_output": strip_ansi(output or ""),
        "agent_response": strip_ansi(agent_text or ""),
    }


def format_transcript(transcript):
    """Format clean transcript for LLM context."""
    lines = []
    for turn in transcript:
        lines.append(f"--- Round {turn['round']} ---")
        lines.append(f"User: {turn['user']}")
        if turn["agent_commands"]:
            for cmd in turn["agent_commands"]:
                lines.append(f"Agent ran: {cmd}")
        if turn["command_output"]:
            lines.append(f"Output:\n{turn['command_output']}")
        if turn["agent_response"]:
            lines.append(f"Agent: {turn['agent_response']}")
    return "\n".join(lines)
