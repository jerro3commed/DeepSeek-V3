"""Unit tests for inference.prompt."""

import pytest

from inference.prompt import (
    Message,
    Role,
    build_messages,
    format_chat_prompt,
)


class TestMessage:
    def test_role_coercion(self):
        msg = Message(role="user", content="hi")  # type: ignore[arg-type]
        assert msg.role is Role.USER

    def test_empty_content_raises(self):
        with pytest.raises(ValueError, match="content must not be empty"):
            Message(role=Role.USER, content="")

    def test_invalid_role_raises(self):
        with pytest.raises(ValueError):
            Message(role="unknown", content="hi")  # type: ignore[arg-type]


class TestFormatChatPrompt:
    def test_single_user_message(self):
        msgs = [Message(role=Role.USER, content="Hello")]
        prompt = format_chat_prompt(msgs)
        assert "<|user|>" in prompt
        assert "Hello" in prompt
        assert "<|assistant|>" in prompt

    def test_system_prompt_prepended(self):
        msgs = [Message(role=Role.USER, content="Hi")]
        prompt = format_chat_prompt(msgs, system_prompt="Be concise.")
        assert prompt.index("<|system|>") < prompt.index("<|user|>")
        assert "Be concise." in prompt

    def test_assistant_turn_appended(self):
        msgs = [Message(role=Role.USER, content="Ping")]
        prompt = format_chat_prompt(msgs)
        assert prompt.endswith("<|assistant|>\n")

    def test_multi_turn_order(self):
        msgs = [
            Message(role=Role.USER, content="Q1"),
            Message(role=Role.ASSISTANT, content="A1"),
            Message(role=Role.USER, content="Q2"),
        ]
        prompt = format_chat_prompt(msgs)
        assert prompt.index("Q1") < prompt.index("A1") < prompt.index("Q2")


class TestBuildMessages:
    def test_basic(self):
        msgs = build_messages("Hello")
        assert len(msgs) == 1
        assert msgs[0].role is Role.USER
        assert msgs[0].content == "Hello"

    def test_with_system(self):
        msgs = build_messages("Hello", system_prompt="You are helpful.")
        assert msgs[0].role is Role.SYSTEM
        assert msgs[-1].role is Role.USER

    def test_with_history(self):
        history = [
            Message(role=Role.USER, content="prev q"),
            Message(role=Role.ASSISTANT, content="prev a"),
        ]
        msgs = build_messages("new q", history=history)
        assert len(msgs) == 3
        assert msgs[-1].content == "new q"

    def test_system_plus_history(self):
        history = [Message(role=Role.USER, content="old")]
        msgs = build_messages("new", history=history, system_prompt="sys")
        assert msgs[0].role is Role.SYSTEM
        assert msgs[1].content == "old"
        assert msgs[2].content == "new"
