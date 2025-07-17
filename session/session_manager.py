import re
import chainlit as cl
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from utils.memory import ChatHistoryMemory
import httpx
from uuid import UUID, uuid4
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Groq LLM with Qwen3-32B
try:
    llm = ChatGroq(
        model="qwen/qwen3-32b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7,
        max_tokens=4000
    )
except Exception as e:
    print(f"Error initializing Groq LLM: {str(e)}")
    raise ValueError("Failed to initialize Qwen3-32B on Groq. Verify model availability or use an alternative model.")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are DATA GPT, a helpful assistant. Respond only to the user's exact message. "
     "Do not guess their next question or include extra information unless asked. "
     "Keep answers well explained, direct, and friendly. For complex queries, use thinking mode with step-by-step reasoning."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

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
                json={"session_id": session_id_str},  # Send as JSON body
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
    if not session_id:
        await cl.Message("Session ID not found. Please restart the chat.").send()
        return

    async with httpx.AsyncClient() as client:
        for file_element in msg.elements:
            file_name = file_element.name or "unknown_file"
            print(f"Processing file: {file_name}, MIME: {file_element.mime}")
            try:
                # Check if file_element has content attribute
                if not hasattr(file_element, 'content') or not file_element.content:
                    # Try alternative approaches for reading file content
                    if hasattr(file_element, 'path'):
                        with open(file_element.path, 'rb') as f:
                            file_content = f.read()
                    elif hasattr(file_element, 'url'):
                        response = await client.get(file_element.url)
                        file_content = response.content
                    else:
                        await cl.Message(f"File '{file_name}' has no accessible content. Please try uploading again.").send()
                        continue
                else:
                    file_content = await file_element.content.read()

                if not file_content:
                    await cl.Message(f"File '{file_name}' is empty. Please upload a non-empty file.").send()
                    continue

                # Prepare the file for the POST request
                files = {"file": (file_name, file_content, file_element.mime)}
                data = {"session_id": session_id}  # Send session_id as form data

                # Send the file to the backend
                print(f"Sending file '{file_name}' to /upload with session_id: {session_id}")
                response = await client.post(
                    "http://localhost:8001/upload",
                    files=files,
                    data=data,
                    timeout=30.0
                )
                response.raise_for_status()

                # Process the response from the backend
                response_data = response.json()
                await cl.Message(f"File '{file_name}' uploaded successfully: {response_data.get('message', 'No message provided')}").send()

            except httpx.RequestError as e:
                await cl.Message(f"Network error uploading file '{file_name}': {str(e)}").send()
            except httpx.HTTPStatusError as e:
                await cl.Message(f"HTTP error uploading file '{file_name}': {e.response.status_code} {e.response.text}").send()
            except Exception as e:
                await cl.Message(f"Unexpected error uploading file '{file_name}': {str(e)}").send()
                print(f"Error details for '{file_name}': {str(e)}")

@cl.on_message
async def handle_message(msg: cl.Message):
    # First, check if there are any file uploads to handle
    if msg.elements:
        await handle_file_upload(msg)
        
        # If there's also text content, process it after file upload
        if not msg.content.strip():
            return  # Only files, no text message to process
    
    # Handle text message
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