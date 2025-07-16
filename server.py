from fastmcp import FastMCP
from gmail_service import get_gmail_service
import base64
from bs4 import BeautifulSoup
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

mcp = FastMCP()
openai_client = OpenAI(api_key=os.getenv("API_KEY"), base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

def get_unread_emails_helper(limit: int = 5) -> list:
    """
    Fetch up to `limit` unread emails with sender, subject, and plain text body.
    """
    service = get_gmail_service()
    response = service.users().messages().list(userId='me', labelIds=['INBOX', 'UNREAD'], maxResults=limit).execute()
    messages = response.get('messages', [])
    
    emails = []

    for msg in messages:
        msg_detail = service.users().messages().get(userId='me', id=msg['id']).execute()
        headers = msg_detail['payload']['headers']
        payload = msg_detail['payload']
        
        email = {
            "from": next((h['value'] for h in headers if h['name'] == 'From'), ''),
            "subject": next((h['value'] for h in headers if h['name'] == 'Subject'), ''),
            "body": ""
        }

        # Extract plain text body (could be inside 'parts' or 'body')
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    data = part['body'].get('data')
                    if data:
                        email['body'] = base64.urlsafe_b64decode(data).decode("utf-8")
        elif payload.get('body') and payload['body'].get('data'):
            data = payload['body']['data']
            email['body'] = base64.urlsafe_b64decode(data).decode("utf-8")

        # Clean up HTML if necessary
        if not email['body']:
            soup = BeautifulSoup(msg_detail.get('snippet', ''), "html.parser")
            email['body'] = soup.get_text()

        emails.append(email)
    return emails

def send_email_helper(to: str, subject: str, body: str):
    service = get_gmail_service()
    message = MIMEMultipart()
    message['to'] = to
    message['subject'] = subject
    message.attach(MIMEText(body, 'plain'))
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    service.users().messages().send(userId='me', body={'raw': raw}).execute()
    return "Email sent successfully"


@mcp.tool
def get_unread_emails(limit: int = 5) -> list:
    return get_unread_emails_helper(limit=limit)

@mcp.tool
def send_email(to: str, subject: str, body: str):
    """
    Send an email to the given recipient with the given subject and body.
    """
    return send_email_helper(to=to, subject=subject, body=body)

@mcp.tool
def batch_process_emails(limit: int = 3) -> list:
    """
    Fetch unread emails, generate intelligent replies using LM Studio, and send them.
    """
    emails = get_unread_emails_helper(limit=limit)
    results = []

    for email in emails:
        prompt = f"""
You are an HR assistant. Respond professionally to the following email related to HR.

Email:
---
{email['body']}
---
"""

        # Call LM Studio local API
        try:
            # response = requests.post(
            #     "http://localhost:1234/v1/chat/completions",
            #     headers={"Content-Type": "application/json"},
            #     json={
            #         "model": "your-model-name-here", 
            #         "messages": [
            #             {"role": "system", "content": "You are an HR auto-responder."},
            #             {"role": "user", "content": prompt}
            #         ],
            #         "temperature": 0.7
            #     }
            # )
            response = openai_client.chat.completions.create(
                model="gemini-1.5-flash",
                messages=[
                    {"role": "system", "content": "You are an HR auto-responder."},
                    {"role": "user", "content": prompt}
                ],
            )
            llm_reply = response.choices[0].message.content

            # Send reply
            send_result = send_email_helper(
                to=email["from"],
                subject="RE: " + email["subject"],
                body=llm_reply or "No response from LLM"
            )

            results.append({
                "to": email["from"],
                "status": "✅ Replied",
                "llm_reply": llm_reply,
                "send_result": send_result
            })

        except Exception as e:
            results.append({
                "to": email["from"],
                "status": f"❌ Error: {e}"
            })

    return results

if __name__ == "__main__":
    mcp.run()
