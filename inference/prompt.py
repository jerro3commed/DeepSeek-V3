"""Prompt formatting utilities for DeepSeek-V3."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """A single conversation message."""

    role: Role
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, Role):
            self.role = Role(self.role)
        if not self.content:
            raise ValueError("Message content must not be empty.")


CHAT_TEMPLATE = "<|{role}|>\n{content}<|end|>\n"
ASSISTANT_PROMPT = "<|assistant|>\n"

# Default system prompt I use for most of my personal experiments.
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. "
    "Answer clearly and avoid unnecessary verbosity."
)


def format_chat_prompt(
    messages: List[Message],
    system_prompt: Optional[str] = None,
) -> str:
    """Format a list of messages into a single prompt string.

    Args:
        messages: Ordered list of conversation messages.
        system_prompt: Optional system instruction prepended before all messages.

    Returns:
        A single formatted prompt string ready for tokenisation.
    """
    parts: List[str] = []

    if system_prompt:
        parts.append(CHAT_TEMPLATE.format(role=Role.SYSTEM.value, content=system_prompt))

    for msg in messages:
        parts.append(CHAT_TEMPLATE.format(role=msg.role.value, content=msg.content))

    # Signal the model to start generating the assistant turn.
    parts.append(ASSISTANT_PROMPT)
    return "".join(parts)


def build_messages(
    user_input: str,
    history: Optional[List[Message]] = None,
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
) -> List[Message]:
    """Convenience helper that constructs a message list from raw inputs.

    Args:
        user_input: The latest user message.
        history: Previous messages in the conversation (may be None).
        system_prompt: If provided, prepend a system message. Defaults to
            ``DEFAULT_SYSTEM_PROMPT``.

    Returns:
        A list of :class:`Message` objects.
    """
    messages: List[Message] = []
    if system_prompt:
        messages.append(Message(role=Role.SYSTEM, content=system_prompt))
    if history:
        messages.extend(history)
    messages.append(Message(role=Role.USER, content=user_input))
    return messages
