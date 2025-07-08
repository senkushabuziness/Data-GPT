#session/session_manager.py
import re
import chainlit as cl
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from utils.memory import ChatHistoryMemory
from llm.llama_hosted import HostedLLM

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are DATA GPT, a helpful assistant. Respond only to the user’s exact message. "
     "Do not guess their next question or include extra information unless asked. "
     "Do not include summary of previous messages. Simply answer what is asked."
     "Keep answers well explained, direct, and friendly."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

llm = HostedLLM()

def get_session_history(session_id: str) -> ChatHistoryMemory:
    return ChatHistoryMemory(session_id)

@cl.on_chat_start
async def start_chat():
    user = cl.user_session.get("user")
    if not user:
        await cl.Message("Please log in to continue.").send()
        return

    session_id = cl.context.session.thread_id or user.identifier
    chain = RunnableWithMessageHistory(
        prompt | llm,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )
    cl.user_session.set("chat_chain", chain)
    cl.user_session.set("session_id", session_id)

    await cl.Message(f"Welcome {user.display_name}! How can I assist you today?").send()

@cl.on_message
async def handle_message(msg: cl.Message):
    chain = cl.user_session.get("chat_chain")
    session_id = cl.user_session.get("session_id")

    if not chain:
        await cl.Message("Chat chain not found. Please restart.").send()
        return

    raw_response = await chain.ainvoke(
        {"input": msg.content},
        config={"configurable": {"session_id": session_id}}
    )
    clean_response = re.sub(r"<think>.*?</think>", "", str(raw_response), flags=re.DOTALL).strip()
    await cl.Message(clean_response).send()

@cl.on_chat_resume
async def resume_chat(thread: dict):
    user = cl.user_session.get("user")
    if not user:
        await cl.Message("Please log in first.").send()
        return

    session_id = thread["id"]
    memory = get_session_history(session_id)
    memory.clear()

    for step in thread["steps"][-10:]:
        if step["type"] == "user_message":
            memory.add_user_message(step["output"])
        elif step["type"] == "ai_message":
            memory.add_ai_message(step["output"])

    chat_chain = RunnableWithMessageHistory(
        prompt | llm,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )
    cl.user_session.set("chat_chain", chat_chain)
