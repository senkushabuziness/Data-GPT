#main 
import sys
import os
import chainlit as cl
from utils import config  # Assuming you store env vars in utils/config.py

sys.path.append(os.path.dirname(os.path.abspath(__file__)))



from auth import auth_config
from session import session_manager
# from sql_executor import executor
@cl.on_chat_start
async def chat_start():
    await session_manager.start_chat()

@cl.on_chat_resume
async def chat_resume(thread):
    await session_manager.resume_chat(thread)

@cl.on_message
async def message_router(msg: cl.Message):
    if msg.elements:
        await session_manager.handle_file_upload(msg)
    else:
        await session_manager.handle_message(msg)
