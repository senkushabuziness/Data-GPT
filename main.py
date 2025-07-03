# main.py

from auth import auth_config
from session import session_manager
from sql_executor import executor
#from charts import chart_generator
import chainlit as cl
from session.session_manager import start_chat, resume_chat, handle_message



@cl.on_chat_start
async def chat_start():
    await session_manager.start_chat()

@cl.on_chat_resume
async def chat_resume(thread):
    await session_manager.resume_chat(thread)

@cl.on_message
async def message_router(msg: cl.Message):
    if msg.elements:
        await executor.handle_file_upload(msg)
    else:
        await session_manager.handle_message(msg)
