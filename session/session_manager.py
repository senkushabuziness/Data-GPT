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
from logger import logger
from api.states import app_state

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

async def share_session_with_fastapi(session_id: str, user_id: str):
    """Send the session_id and user_id to the FastAPI /create-session endpoint."""
    async with httpx.AsyncClient() as client:
        session_id_str = str(session_id)
        user_id_str = str(user_id)
        print(f"Attempting to share session {session_id_str} for user {user_id_str}")
        try:
            response = await client.post(
                "http://localhost:8001/create-session",
                json={"session_id": session_id_str, "user_id": user_id_str},
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            returned_session_id = data.get("session_id")
            returned_user_id = data.get("user_id")
            print(f"Received response: {data}")
            if returned_session_id == session_id_str and returned_user_id == user_id_str:
                print(f"Session {returned_session_id} for user {returned_user_id} successfully shared with FastAPI")
            else:
                print(f"ID mismatch: expected session {session_id_str}/user {user_id_str}, got session {returned_session_id}/user {returned_user_id}")
            return returned_session_id, returned_user_id
        except httpx.HTTPStatusError as e:
            print(f"HTTP error with session {session_id_str}/user {user_id_str}: {e.response.status_code} {e.response.text}")
            raise
        except httpx.RequestError as e:
            print(f"Network error with session {session_id_str}/user {user_id_str}: {str(e)}")
            raise
        except Exception as e:
            print(f"Unexpected error with session {session_id_str}/user {user_id_str}: {str(e)}")
            raise

@cl.on_chat_start
async def start_chat():
    # Get user from session, handle case where authentication is disabled
    user = cl.user_session.get("user")
    user_id = user.id if user else str(uuid4())
    print(f"User ID: {user_id}")
    display_name = getattr(user, "display_name", "User") if user else "User"
    
    print("Chat start triggered")
    # Check if session_id and user_id already exist
    session_id = cl.user_session.get("session_id")
    stored_user_id = cl.user_session.get("user_id")
    
    if session_id and stored_user_id:
        try:
            UUID(session_id)  # Validate existing session_id
            UUID(stored_user_id)  # Validate existing user_id
            print(f"Reusing existing session_id: {session_id}, user_id: {stored_user_id}")
            await cl.Message(f"Welcome {display_name}! Using existing session. How can I assist you today?").send()
            return
        except ValueError:
            print(f"Invalid existing session_id: {session_id} or user_id: {stored_user_id}. Generating new ones.")

    # Generate new session_id and validate user_id
    session_id = cl.context.session.thread_id or user_id
    try:
        UUID(session_id)
    except ValueError:
        session_id = str(uuid4())
    try:
        UUID(user_id)
    except ValueError:
        user_id = str(uuid4())
    print(f"Generated new session_id: {session_id}, user_id: {user_id}")

    # Share with FastAPI
    try:
        returned_session_id, returned_user_id = await share_session_with_fastapi(session_id, user_id)
        cl.user_session.set("session_id", returned_session_id)
        cl.user_session.set("user_id", returned_user_id)
    except Exception as e:
        print(f"Failed to share session with FastAPI: {str(e)}. Using local session_id {session_id}, user_id {user_id}")
        cl.user_session.set("session_id", session_id)
        cl.user_session.set("user_id", user_id)

    chain = RunnableWithMessageHistory(
        prompt | llm,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )
    cl.user_session.set("chat_chain", chain)

    await cl.Message(f"Welcome {display_name}! How can I assist you today?").send()

@cl.on_chat_resume
async def resume_chat(thread: dict):
    # Get user from session, handle case where authentication is disabled
    user = cl.user_session.get("user")
    user_id = user.id if user else str(uuid4())
    
    print("Chat resume triggered")
    # Check if session_id and user_id already exist
    session_id = cl.user_session.get("session_id")
    stored_user_id = cl.user_session.get("user_id")
    
    if session_id and stored_user_id:
        try:
            UUID(session_id)  # Validate existing session_id
            UUID(stored_user_id)  # Validate existing user_id
            print(f"Reusing existing session_id: {session_id}, user_id: {stored_user_id}")
            # Reinitialize chat chain with existing session
            chain = RunnableWithMessageHistory(
                prompt | llm,
                get_session_history=get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history"
            )
            cl.user_session.set("chat_chain", chain)
            return
        except ValueError:
            print(f"Invalid existing session_id: {session_id} or user_id: {stored_user_id}. Generating new ones.")

    # Generate new session_id and validate user_id
    session_id = thread.get("id") or user_id
    if not isinstance(session_id, str) or not session_id:
        session_id = str(uuid4())
    else:
        try:
            UUID(session_id)
        except ValueError:
            session_id = str(uuid4())
    try:
        UUID(user_id)
    except ValueError:
        user_id = str(uuid4())
    print(f"Generated new session_id for resume: {session_id}, user_id: {user_id}")

    # Share with FastAPI
    try:
        returned_session_id, returned_user_id = await share_session_with_fastapi(session_id, user_id)
        cl.user_session.set("session_id", returned_session_id)
        cl.user_session.set("user_id", returned_user_id)
    except Exception as e:
        print(f"Failed to share session with FastAPI: {str(e)}. Using local session_id {session_id}, user_id {user_id}")
        cl.user_session.set("session_id", session_id)
        cl.user_session.set("user_id", user_id)

    memory = get_session_history(session_id)
    memory.clear()

    for step in thread["steps"][-10:]:
        if step["type"] == "user_message":
            memory.add_user_message(step["output"])
        elif step["type"] == "ai_message":
            memory.add_ai_message(step["output"])

    chain = RunnableWithMessageHistory(
        prompt | llm,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )
    cl.user_session.set("chat_chain", chain)

async def handle_file_upload(msg: cl.Message):
    """Handle file uploads from Chainlit and send them to the backend /upload endpoint."""
    if not msg.elements:
        await cl.Message("No files were uploaded.").send()
        return

    session_id = cl.user_session.get("session_id")
    user_id = cl.user_session.get("user_id")
    if not session_id or not user_id:
        await cl.Message("Session ID or User ID not found. Please restart the chat.").send()
        return

    async with httpx.AsyncClient() as client:
        for file_element in msg.elements:
            file_name = file_element.name or "unknown_file"
            logger.info(f"Processing file: {file_name}, MIME: {file_element.mime}")
            
            try:
                # Read file content using multiple approaches
                file_content = None
                
                if hasattr(file_element, 'content') and file_element.content:
                    file_content = await file_element.content.read()
                elif hasattr(file_element, 'path') and file_element.path:
                    with open(file_element.path, 'rb') as f:
                        file_content = f.read()
                elif hasattr(file_element, 'url') and file_element.url:
                    url_response = await client.get(file_element.url)
                    file_content = url_response.content
                
                if not file_content:
                    await cl.Message(f"File '{file_name}' has no accessible content or is empty. Please try uploading again.").send()
                    continue

                # Prepare the multipart form data
                files = {"file": (file_name, file_content, file_element.mime)}
                data = {"session_id": session_id, "user_id": user_id}

                # Send file to backend upload endpoint
                logger.info(f"Uploading '{file_name}' to /upload with session_id: {session_id}, user_id: {user_id}")
                
                upload_response = await client.post(
                    "http://localhost:8001/upload",
                    files=files,
                    data=data,
                    timeout=45.0
                )
                upload_response.raise_for_status()

                # Process successful response
                response_data = upload_response.json()
                logger.info(f"Upload response: {response_data}")
                logger.info(f"Response keys: {list(response_data.keys())}")
                logger.info(f"Raw response text: {upload_response.text}")
                logger.info(f"Has app_state key: {'app_state' in response_data}")
                logger.info(f"app_state value: {response_data.get('app_state', 'NOT_FOUND')}")
                # Send success message to user
                success_message = response_data.get('message', f"File '{file_name}' uploaded successfully")
                logger.info(f"===========\n\n\n\n\n\n\n\n\n\n {response_data}")
                # Update global app_state if present in response
                if response_data.get("app_state"):
                    logger.info("====================================\n\n\n")
                    # app_state.update({response_data["app_state"]})
                    logger.info(f"app_state updated: {app_state}")
                else:
                    logger.warning(f" app_state not found in upload response for file '{file_name}'")
                await cl.Message(success_message).send()
            except httpx.HTTPStatusError as e:
                error_msg = f"HTTP {e.response.status_code} error uploading '{file_name}': {e.response.text}"
                logger.error(error_msg, exc_info=True)
                await cl.Message(f"Upload failed for '{file_name}': {error_msg}").send()
                
            except httpx.RequestError as e:
                error_msg = f"Network error uploading '{file_name}': {str(e)}"
                logger.error(error_msg, exc_info=True)
                await cl.Message(f"Network error for '{file_name}': {str(e)}").send()
                
            except Exception as e:
                error_msg = f"Unexpected error uploading '{file_name}': {str(e)}"
                logger.error(error_msg, exc_info=True)
                await cl.Message(f"Upload failed for '{file_name}': {str(e)}").send()
@cl.on_message
async def handle_message(msg: cl.Message):
    if msg.elements:
        await handle_file_upload(msg)
        if not msg.content.strip():
            return  # Only files, no text message to process
    
    # Handle text message
    chain = cl.user_session.get("chat_chain")
    session_id = cl.user_session.get("session_id")
    user_id = cl.user_session.get("user_id")

    if not chain or not session_id or not user_id:
        await cl.Message("Chat chain, session ID, or user ID not found. Please restart.").send()
        return

    raw_response = await chain.ainvoke(
        {"input": msg.content},
        config={"configurable": {"session_id": session_id}}
    )
    clean_response = re.sub(r"<think>.*?</think>", "", str(raw_response), flags=re.DOTALL).strip()
    await cl.Message(clean_response).send()