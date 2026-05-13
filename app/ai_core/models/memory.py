from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.ai_core.models.llm_loader import get_response_text


@dataclass
class DialogueTurn:
    user: str
    assistant: str


@dataclass
class ConversationSummaryBuffer:
    """Small production-friendly memory inspired by ConversationSummaryBufferMemory."""

    max_turns: int = 6
    max_chars_before_summary: int = 4_000
    summary: str = ''
    turns: list[DialogueTurn] = field(default_factory=list)

    def add(self, user: str, assistant: str, llm: Any | None = None) -> None:
        self.turns.append(DialogueTurn(user=user, assistant=assistant))
        if len(self.render()) > self.max_chars_before_summary and llm is not None:
            self._summarize(llm)
        if len(self.turns) > self.max_turns:
            old = self.turns[:-self.max_turns]
            if old and not self.summary:
                self.summary = ' '.join(f"Người dùng: {t.user}; Bot: {t.assistant[:120]}" for t in old)
            self.turns = self.turns[-self.max_turns:]

    def render(self) -> str:
        parts: list[str] = []
        if self.summary:
            parts.append(f'Tóm tắt trước đó: {self.summary}')
        for turn in self.turns[-self.max_turns:]:
            parts.append(f'Người dùng: {turn.user}\nBot y tế: {turn.assistant[:500]}')
        return '\n'.join(parts) if parts else 'Chưa có lịch sử.'

    def clear(self) -> None:
        self.summary = ''
        self.turns.clear()

    def _summarize(self, llm: Any) -> None:
        prompt = (
            'Tóm tắt ngắn gọn lịch sử hội thoại y tế sau bằng tiếng Việt, '
            'chỉ giữ lại thông tin cần thiết cho câu hỏi tiếp theo:\n\n'
            f'{self.render()}\n\nTóm tắt:'
        )
        try:
            self.summary = get_response_text(llm.invoke(prompt)).strip()[:1_000]
            self.turns = self.turns[-self.max_turns:]
        except Exception:
            # Do not fail chat on memory summarization errors.
            self.summary = self.summary[:1_000]
