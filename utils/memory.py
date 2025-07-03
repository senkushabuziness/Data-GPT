# utils/memory.py

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.schema import BaseMessage
from typing import Dict, List

class ChatHistoryMemory(BaseChatMessageHistory):
    sessions: Dict[str, ChatMessageHistory] = {}
    context_limit: int = 10  # Latest 10 messages

    def __init__(self, session_id: str):
        self.session_id = session_id
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatMessageHistory(session_id=session_id)
        self.chat_history = self.sessions[session_id]

    @property
    def messages(self) -> List[BaseMessage]:
        return self.chat_history.messages[-self.context_limit:]

    def add_user_message(self, message: str) -> None:
        self.chat_history.add_user_message(message)

    def add_ai_message(self, message: str) -> None:
        self.chat_history.add_ai_message(message)

    def clear(self) -> None:
        self.chat_history.clear()

    def add_message(self, message: BaseMessage) -> None:
        if hasattr(self.chat_history, 'add_message'):
            self.chat_history.add_message(message)
        else:
            self.messages.append(message)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        if hasattr(self.chat_history, 'add_messages'):
            self.chat_history.add_messages(messages)
        else:
            for message in messages:
                self.add_message(message)
