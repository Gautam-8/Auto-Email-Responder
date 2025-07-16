import asyncio
import base64
from fastmcp import Client
from gmail_service import get_gmail_service

# service = get_gmail_service()
# print("✅ Gmail service is ready.")

client = Client("server.py")

async def call_all_tools():
    async with client:
        # result = await client.call_tool("get_unread_emails", {
        #     "limit": 5
        # })
        # print('Unread Emails:', result)

        # result = await client.call_tool("send_email", {
        #     "to": "hgautambpetrm@gmail.com",
        #     "subject": "Test Email",
        #     "body": "This is a test email yeh gautam bhai"
        # })
        # print('Email Sent:', result)

        result = await client.call_tool("batch_process_emails", {
            "limit": 2
        })
        print('Batch Process Emails:', result)

asyncio.run(call_all_tools())