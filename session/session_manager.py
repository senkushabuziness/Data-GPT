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
        session_id_str = str(session_id)
        print(f"Attempting to share session {session_id_str}")
        try:
            response = await client.post(
                "http://localhost:8001/create-session",
                content=session_id_str,
                headers={"Content-Type": "text/plain"},
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            returned_session_id = data.get("session_id")
            print(f"Received response: {data}")
            if returned_session_id == session_id_str:
                print(f"Session {returned_session_id} successfully shared with FastAPI")
            else:
                print(f"Session ID mismatch: expected {session_id_str}, got {returned_session_id}")
            # Update session cookie if present
            set_cookie = response.headers.get("set-cookie")
            if set_cookie and "session_cookie" in set_cookie:
                new_session_id = set_cookie.split("session_cookie=")[1].split(";")[0]
                cl.user_session.set("session_id", new_session_id)
                print(f"Updated session_id to {new_session_id}")
            return returned_session_id
        except httpx.HTTPStatusError as e:
            print(f"HTTP error with session {session_id_str}: {e.response.status_code} {e.response.text}")
            raise
        except httpx.RequestError as e:
            print(f"Network error with session {session_id_str}: {str(e)}")
            raise
        except Exception as e:
            print(f"Unexpected error with session {session_id_str}: {str(e)}")
            raise

@cl.on_chat_start
async def start_chat():
    user = cl.user_session.get("user")
    if not user:
        await cl.Message("Please log in to continue.").send()
        return

    print("Chat start triggered")
    # Check if session_id already exists
    session_id = cl.user_session.get("session_id")
    if session_id:
        try:
            UUID(session_id)  # Validate existing session_id
            print(f"Reusing existing session_id: {session_id}")
            await cl.Message(f"Welcome {user.display_name}! Using existing session. How can I assist you today?").send()
            return
        except ValueError:
            print(f"Invalid existing session_id: {session_id}. Generating new one.")

    # Generate new session_id
    session_id = cl.context.session.thread_id or user.identifier
    try:
        UUID(session_id)
    except ValueError:
        session_id = str(uuid4())
    print(f"Generated new session_id: {session_id}")

    # Share with FastAPI
    try:
        returned_session_id = await share_session_with_fastapi(session_id)
        cl.user_session.set("session_id", returned_session_id)
    except Exception as e:
        print(f"Failed to share session with FastAPI: {str(e)}. Using local session_id {session_id}")
        cl.user_session.set("session_id", session_id)

    chain = RunnableWithMessageHistory(
        prompt | llm,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )
    cl.user_session.set("chat_chain", chain)

    await cl.Message(f"Welcome {user.display_name}! How can I assist you today?").send()

@cl.on_chat_resume
async def resume_chat(thread: dict):
    user = cl.user_session.get("user")
    if not user:
        await cl.Message("Please log in first.").send()
        return

    print("Chat resume triggered")
    # Check if session_id already exists
    session_id = cl.user_session.get("session_id")
    if session_id:
        try:
            UUID(session_id)  # Validate existing session_id
            print(f"Reusing existing session_id: {session_id}")
            # Reinitialize chat chain with existing session
            chat_chain = RunnableWithMessageHistory(
                prompt | llm,
                get_session_history=get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history"
            )
            cl.user_session.set("chat_chain", chat_chain)
            return
        except ValueError:
            print(f"Invalid existing session_id: {session_id}. Generating new one.")

    # Generate new session_id
    session_id = thread.get("id")
    if not isinstance(session_id, str) or not session_id:
        session_id = str(uuid4())
    else:
        try:
            UUID(session_id)
        except ValueError:
            session_id = str(uuid4())
    print(f"Generated new session_id for resume: {session_id}")

    # Share with FastAPI
    try:
        returned_session_id = await share_session_with_fastapi(session_id)
        cl.user_session.set("session_id", returned_session_id)
    except Exception as e:
        print(f"Failed to share session with FastAPI: {str(e)}. Using local session_id {session_id}")
        cl.user_session.set("session_id", session_id)

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

async def handle_file_upload(msg: cl.Message):
    """Handle file uploads from Chainlit and send them to the backend /upload endpoint."""
    if not msg.elements:
        await cl.Message("No files were uploaded.").send()
        return

    session_id = cl.user_session.get("session_id")
    headers = {"Cookie": f"session_cookie={session_id}"} if session_id else {}

    async with httpx.AsyncClient() as client:
        for file_element in msg.elements:
            file_name = file_element.name or "unknown_file"
            print(f"Processing file: {file_name}, MIME: {file_element.mime}")
            try:
                if not file_element.content:
                    await cl.Message(f"File '{file_name}' has no content. Please upload a valid file.").send()
                    continue

                # Read the file content
                file_content = await file_element.content.read()
                if not file_content:
                    await cl.Message(f"File '{file_name}' is empty. Please upload a non-empty file.").send()
                    continue

                # Prepare the file for the POST request
                files = {"file": (file_name, file_content, file_element.mime)}
                
                # Send the file to the backend
                print(f"Sending file '{file_name}' to /upload with session_id: {session_id}")
                response = await client.post(
                    "http://localhost:8001/upload",
                    files=files,
                    headers=headers,
                    timeout=30.0
                )
                response.raise_for_status()

                # Process the response from the backend
                response_data = response.json()
                await cl.Message(f"File '{file_name}' uploaded successfully: {response_data.get('message', 'No message provided')}").send()

                # Update session cookie if a new one is set
                set_cookie = response.headers.get("set-cookie")
                if set_cookie and "session_cookie" in set_cookie:
                    new_session_id = set_cookie.split("session_cookie=")[1].split(";")[0]
                    cl.user_session.set("session_id", new_session_id)
                    print(f"Updated session_id to {new_session_id}")

            except httpx.RequestError as e:
                await cl.Message(f"Network error uploading file '{file_name}': {str(e)}").send()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 403:
                    await cl.Message(f"Session error uploading file '{file_name}': No valid session. A new session may have been created.").send()
                else:
                    await cl.Message(f"HTTP error uploading file '{file_name}': {e.response.status_code} {e.response.text}").send()
            except Exception as e:
                await cl.Message(f"Unexpected error uploading file '{file_name}': {str(e)}").send()
                print(f"Error details for '{file_name}': {str(e)}")
                
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