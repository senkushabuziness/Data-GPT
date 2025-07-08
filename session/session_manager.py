import re
import chainlit as cl
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from utils.memory import ChatHistoryMemory
from llm.llama_hosted import HostedLLM
import httpx
from uuid import UUID, uuid4

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are DATA GPT, a helpful assistant. Respond only to the user’s exact message. "
     "Do not guess their next question or include extra information unless asked. "
     "Keep answers well explained, direct, and friendly."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

llm = HostedLLM()

def get_session_history(session_id: str) -> ChatHistoryMemory:
    return ChatHistoryMemory(session_id)

async def share_session_with_fastapi(session_id: str):
    """Send the session_id to the FastAPI /create-session endpoint."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://localhost:8000/create-session",  # Replace with your FastAPI URL
                json={"session_id": session_id},
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            if data["session_id"] == session_id:
                print(f"Session {session_id} successfully shared with FastAPI")
            else:
                print(f"Session ID mismatch: expected {session_id}, got {data['session_id']}")
        except httpx.RequestError as e:
            print(f"Network error sharing session with FastAPI: {str(e)}")
        except httpx.HTTPStatusError as e:
            print(f"HTTP error sharing session with FastAPI: {e.response.status_code} {e.response.text}")
        except Exception as e:
            print(f"Unexpected error sharing session with FastAPI: {str(e)}")

@cl.on_chat_start
async def start_chat():
    user = cl.user_session.get("user")
    if not user:
        await cl.Message("Please log in to continue.").send()
        return

    # Generate or validate session_id
    session_id = cl.context.session.thread_id or user.identifier
    try:
        UUID(session_id)  # Validate if session_id is a UUID
    except ValueError:
        session_id = str(uuid4())  # Generate a new UUID if invalid

    chain = RunnableWithMessageHistory(
        prompt | llm,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )
    cl.user_session.set("chat_chain", chain)
    cl.user_session.set("session_id", session_id)

    # Share session_id with FastAPI after login
    await share_session_with_fastapi(session_id)

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

    session_id = thread.get("id")
    if not isinstance(session_id, str) or not session_id:
        session_id = str(uuid4())
    else:
        try:
            UUID(session_id)
        except ValueError:
            session_id = str(uuid4())

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
    cl.user_session.set("session_id", session_id)

    # Share session_id with FastAPI on resume
    await share_session_with_fastapi(session_id)