from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    ChatMessageUser,
    ChatMessageAssistant,
    ChatMessageTool,
    ContentText,
    ContentReasoning,
)


def messages_to_transcript(messages: list[ChatMessage], include_system_and_user: bool = True) -> str:
    parts = []

    for msg in messages:
        if isinstance(msg, ChatMessageSystem):
            if include_system_and_user:
                parts.append(f"[SYSTEM]\n{msg.content}\n")

        elif isinstance(msg, ChatMessageUser):
            if include_system_and_user:
                content = msg.content if isinstance(msg.content, str) else _extract_text(msg.content)
                parts.append(f"[USER]\n{content}\n")

        elif isinstance(msg, ChatMessageAssistant):
            section = ["[ASSISTANT]"]

            for block in (msg.content if isinstance(msg.content, list) else []):
                if isinstance(block, ContentReasoning) and block.reasoning:
                    section.append(f"<thinking>\n{block.reasoning}\n</thinking>")
                elif isinstance(block, ContentText) and block.text:
                    section.append(block.text)

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    section.append(f"<tool_call name={tc.function!r}>\n{tc.arguments}\n</tool_call>")

            parts.append("\n".join(section) + "\n")

        elif isinstance(msg, ChatMessageTool):
            header = f"[TOOL RESULT: {msg.function}]"
            if msg.error:
                header += f" ERROR: {msg.error}"
            content = msg.content if isinstance(msg.content, str) else _extract_text(msg.content)
            parts.append(f"{header}\n{content}\n")

    return "\n".join(parts)


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    texts = []
    for block in content:
        if isinstance(block, ContentText) and block.text:
            texts.append(block.text)
        elif hasattr(block, "text") and block.text:
            texts.append(block.text)
    return "\n".join(texts)