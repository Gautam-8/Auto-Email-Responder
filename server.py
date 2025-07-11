#!/usr/bin/env python3
"""
Auto Email Responder MCP Server
Intelligent email response system with policy retrieval and semantic search
"""

import asyncio
import json
import logging
import sqlite3
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import imaplib
import email
from email.header import decode_header
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

from fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Policy:
    """Company policy data structure"""
    id: str
    title: str
    content: str
    category: str
    tags: List[str]
    created_at: datetime
    updated_at: datetime
    priority: int = 1  # 1-5, higher is more important

@dataclass
class EmailTemplate:
    """Email template data structure"""
    id: str
    name: str
    subject: str
    body: str
    category: str
    variables: List[str]  # Placeholders like {customer_name}
    created_at: datetime

@dataclass
class IncomingEmail:
    """Incoming email data structure"""
    id: str
    sender: str
    subject: str
    body: str
    received_at: datetime
    processed: bool = False
    response_sent: bool = False

@dataclass
class EmailResponse:
    """Email response data structure"""
    id: str
    original_email_id: str
    template_id: str
    subject: str
    body: str
    recipient: str
    sent_at: Optional[datetime] = None
    status: str = "pending"  # pending, sent, failed

class PolicyCache:
    """In-memory cache for frequently accessed policies"""
    
    def __init__(self, max_size: int = 100):
        self.cache: Dict[str, Policy] = {}
        self.access_count: Dict[str, int] = {}
        self.max_size = max_size
    
    def get(self, policy_id: str) -> Optional[Policy]:
        if policy_id in self.cache:
            self.access_count[policy_id] = self.access_count.get(policy_id, 0) + 1
            return self.cache[policy_id]
        return None
    
    def put(self, policy: Policy):
        if len(self.cache) >= self.max_size:
            # Remove least accessed policy
            least_accessed = min(self.access_count.items(), key=lambda x: x[1])[0]
            del self.cache[least_accessed]
            del self.access_count[least_accessed]
        
        self.cache[policy.id] = policy
        self.access_count[policy.id] = 1
    
    def clear(self):
        self.cache.clear()
        self.access_count.clear()

class SemanticSearch:
    """Semantic search using sentence transformers and FAISS"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.policy_ids = []
        self.embeddings_cache = {}
    
    def build_index(self, policies: List[Policy]):
        """Build FAISS index from policies"""
        texts = []
        policy_ids = []
        
        for policy in policies:
            # Combine title and content for better search
            text = f"{policy.title} {policy.content}"
            texts.append(text)
            policy_ids.append(policy.id)
        
        if not texts:
            return
        
        # Generate embeddings
        embeddings = self.model.encode(texts)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        self.policy_ids = policy_ids
        
        # Cache embeddings
        for i, policy_id in enumerate(policy_ids):
            self.embeddings_cache[policy_id] = embeddings[i]
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search for relevant policies"""
        if not self.index:
            return []
        
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # Valid result
                policy_id = self.policy_ids[idx]
                results.append((policy_id, float(score)))
        
        return results

class EmailAutoResponder:
    """Main email auto-responder system"""
    
    def __init__(self, db_path: str = "email_responder.db"):
        self.db_path = db_path
        self.policy_cache = PolicyCache()
        self.semantic_search = SemanticSearch()
        self.init_database()
        self.load_semantic_index()
    
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Policies table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS policies (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                category TEXT NOT NULL,
                tags TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                priority INTEGER DEFAULT 1
            )
        ''')
        
        # Email templates table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS email_templates (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                subject TEXT NOT NULL,
                body TEXT NOT NULL,
                category TEXT NOT NULL,
                variables TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Incoming emails table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS incoming_emails (
                id TEXT PRIMARY KEY,
                sender TEXT NOT NULL,
                subject TEXT NOT NULL,
                body TEXT NOT NULL,
                received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed BOOLEAN DEFAULT FALSE,
                response_sent BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Email responses table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS email_responses (
                id TEXT PRIMARY KEY,
                original_email_id TEXT NOT NULL,
                template_id TEXT,
                subject TEXT NOT NULL,
                body TEXT NOT NULL,
                recipient TEXT NOT NULL,
                sent_at TIMESTAMP,
                status TEXT DEFAULT 'pending',
                FOREIGN KEY (original_email_id) REFERENCES incoming_emails (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_semantic_index(self):
        """Load semantic search index from stored policies"""
        policies = self.get_all_policies()
        if policies:
            self.semantic_search.build_index(policies)
    
    def generate_id(self, content: str) -> str:
        """Generate unique ID from content"""
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def add_policy(self, title: str, content: str, category: str, tags: List[str], priority: int = 1) -> str:
        """Add a new company policy"""
        policy_id = self.generate_id(f"{title}{content}")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO policies 
            (id, title, content, category, tags, priority, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (policy_id, title, content, category, json.dumps(tags), priority, datetime.now()))
        
        conn.commit()
        conn.close()
        
        # Update cache and search index
        policy = Policy(
            id=policy_id,
            title=title,
            content=content,
            category=category,
            tags=tags,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            priority=priority
        )
        self.policy_cache.put(policy)
        self.load_semantic_index()  # Rebuild index
        
        return policy_id
    
    def get_policy(self, policy_id: str) -> Optional[Policy]:
        """Get a policy by ID (with caching)"""
        # Check cache first
        policy = self.policy_cache.get(policy_id)
        if policy:
            return policy
        
        # Load from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM policies WHERE id = ?', (policy_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            policy = Policy(
                id=row[0],
                title=row[1],
                content=row[2],
                category=row[3],
                tags=json.loads(row[4]),
                created_at=datetime.fromisoformat(row[5]),
                updated_at=datetime.fromisoformat(row[6]),
                priority=row[7]
            )
            self.policy_cache.put(policy)
            return policy
        
        return None
    
    def get_all_policies(self) -> List[Policy]:
        """Get all policies"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM policies ORDER BY priority DESC, updated_at DESC')
        rows = cursor.fetchall()
        conn.close()
        
        policies = []
        for row in rows:
            policy = Policy(
                id=row[0],
                title=row[1],
                content=row[2],
                category=row[3],
                tags=json.loads(row[4]),
                created_at=datetime.fromisoformat(row[5]),
                updated_at=datetime.fromisoformat(row[6]),
                priority=row[7]
            )
            policies.append(policy)
        
        return policies
    
    def search_policies(self, query: str, max_results: int = 5) -> List[Tuple[Policy, float]]:
        """Search policies using semantic search"""
        search_results = self.semantic_search.search(query, max_results)
        
        policies_with_scores = []
        for policy_id, score in search_results:
            policy = self.get_policy(policy_id)
            if policy:
                policies_with_scores.append((policy, score))
        
        return policies_with_scores
    
    def add_email_template(self, name: str, subject: str, body: str, category: str, variables: List[str]) -> str:
        """Add email template"""
        template_id = self.generate_id(f"{name}{subject}")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO email_templates 
            (id, name, subject, body, category, variables)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (template_id, name, subject, body, category, json.dumps(variables)))
        
        conn.commit()
        conn.close()
        
        return template_id
    
    def get_email_template(self, template_id: str) -> Optional[EmailTemplate]:
        """Get email template by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM email_templates WHERE id = ?', (template_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return EmailTemplate(
                id=row[0],
                name=row[1],
                subject=row[2],
                body=row[3],
                category=row[4],
                variables=json.loads(row[5]),
                created_at=datetime.fromisoformat(row[6])
            )
        
        return None
    
    def get_templates_by_category(self, category: str) -> List[EmailTemplate]:
        """Get templates by category"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM email_templates WHERE category = ?', (category,))
        rows = cursor.fetchall()
        conn.close()
        
        templates = []
        for row in rows:
            template = EmailTemplate(
                id=row[0],
                name=row[1],
                subject=row[2],
                body=row[3],
                category=row[4],
                variables=json.loads(row[5]),
                created_at=datetime.fromisoformat(row[6])
            )
            templates.append(template)
        
        return templates
    
    def store_incoming_email(self, sender: str, subject: str, body: str) -> str:
        """Store incoming email"""
        email_id = self.generate_id(f"{sender}{subject}{body}{datetime.now()}")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO incoming_emails 
            (id, sender, subject, body, received_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (email_id, sender, subject, body, datetime.now()))
        
        conn.commit()
        conn.close()
        
        return email_id
    
    def get_unprocessed_emails(self) -> List[IncomingEmail]:
        """Get unprocessed emails"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM incoming_emails WHERE processed = FALSE')
        rows = cursor.fetchall()
        conn.close()
        
        emails = []
        for row in rows:
            email_obj = IncomingEmail(
                id=row[0],
                sender=row[1],
                subject=row[2],
                body=row[3],
                received_at=datetime.fromisoformat(row[4]),
                processed=bool(row[5]),
                response_sent=bool(row[6])
            )
            emails.append(email_obj)
        
        return emails
    
    def generate_response(self, email: IncomingEmail) -> EmailResponse:
        """Generate intelligent response for incoming email"""
        # Search for relevant policies
        query = f"{email.subject} {email.body}"
        relevant_policies = self.search_policies(query, max_results=3)
        
        # Determine response category based on email content
        category = self.categorize_email(email)
        
        # Get appropriate template
        templates = self.get_templates_by_category(category)
        if not templates:
            # Use default template
            templates = self.get_templates_by_category("general")
        
        if not templates:
            # Create basic response
            response_subject = f"Re: {email.subject}"
            response_body = f"Thank you for your email. We have received your message and will respond shortly.\n\nBest regards,\nCustomer Service Team"
            template_id = None
        else:
            template = templates[0]  # Use first matching template
            template_id = template.id
            
            # Fill template variables
            variables = {
                "customer_name": self.extract_sender_name(email.sender),
                "original_subject": email.subject,
                "current_date": datetime.now().strftime("%B %d, %Y"),
                "relevant_policies": self.format_policies_for_response(relevant_policies[:2])
            }
            
            response_subject = self.fill_template(template.subject, variables)
            response_body = self.fill_template(template.body, variables)
        
        # Create response
        response_id = self.generate_id(f"response_{email.id}_{datetime.now()}")
        response = EmailResponse(
            id=response_id,
            original_email_id=email.id,
            template_id=template_id,
            subject=response_subject,
            body=response_body,
            recipient=email.sender
        )
        
        # Store response
        self.store_email_response(response)
        
        return response
    
    def categorize_email(self, email: IncomingEmail) -> str:
        """Categorize email based on content"""
        content = f"{email.subject} {email.body}".lower()
        
        # Simple keyword-based categorization
        if any(word in content for word in ["refund", "return", "money back"]):
            return "refund"
        elif any(word in content for word in ["billing", "payment", "invoice", "charge"]):
            return "billing"
        elif any(word in content for word in ["support", "help", "problem", "issue"]):
            return "support"
        elif any(word in content for word in ["complaint", "unhappy", "dissatisfied"]):
            return "complaint"
        else:
            return "general"
    
    def extract_sender_name(self, sender_email: str) -> str:
        """Extract name from sender email"""
        # Simple extraction - just use part before @
        if "@" in sender_email:
            return sender_email.split("@")[0].replace(".", " ").title()
        return "Valued Customer"
    
    def format_policies_for_response(self, policies_with_scores: List[Tuple[Policy, float]]) -> str:
        """Format policies for inclusion in response"""
        if not policies_with_scores:
            return ""
        
        formatted = "Based on our company policies:\n\n"
        for policy, score in policies_with_scores:
            formatted += f"• {policy.title}: {policy.content[:200]}...\n\n"
        
        return formatted
    
    def fill_template(self, template: str, variables: Dict[str, str]) -> str:
        """Fill template with variables"""
        result = template
        for key, value in variables.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result
    
    def store_email_response(self, response: EmailResponse):
        """Store email response"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO email_responses 
            (id, original_email_id, template_id, subject, body, recipient, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (response.id, response.original_email_id, response.template_id, 
              response.subject, response.body, response.recipient, response.status))
        
        conn.commit()
        conn.close()
    
    def mark_email_processed(self, email_id: str):
        """Mark email as processed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('UPDATE incoming_emails SET processed = TRUE WHERE id = ?', (email_id,))
        
        conn.commit()
        conn.close()
    
    def process_batch_emails(self, batch_size: int = 10) -> List[EmailResponse]:
        """Process batch of unprocessed emails"""
        unprocessed_emails = self.get_unprocessed_emails()[:batch_size]
        responses = []
        
        for email in unprocessed_emails:
            try:
                response = self.generate_response(email)
                responses.append(response)
                self.mark_email_processed(email.id)
                logger.info(f"Processed email {email.id} from {email.sender}")
            except Exception as e:
                logger.error(f"Error processing email {email.id}: {str(e)}")
        
        return responses
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Count policies
        cursor.execute('SELECT COUNT(*) FROM policies')
        policy_count = cursor.fetchone()[0]
        
        # Count templates
        cursor.execute('SELECT COUNT(*) FROM email_templates')
        template_count = cursor.fetchone()[0]
        
        # Count emails
        cursor.execute('SELECT COUNT(*) FROM incoming_emails')
        total_emails = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM incoming_emails WHERE processed = TRUE')
        processed_emails = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM email_responses WHERE status = "sent"')
        sent_responses = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "policies": policy_count,
            "templates": template_count,
            "total_emails": total_emails,
            "processed_emails": processed_emails,
            "pending_emails": total_emails - processed_emails,
            "sent_responses": sent_responses,
            "cache_size": len(self.policy_cache.cache),
            "search_index_size": len(self.semantic_search.policy_ids) if self.semantic_search.policy_ids else 0
        }

# Initialize the auto-responder system
auto_responder = EmailAutoResponder()

# Create FastMCP server
mcp = FastMCP("Email Auto-Responder")

@mcp.tool()
def add_company_policy(title: str, content: str, category: str, tags: str, priority: int = 1) -> str:
    """Add a new company policy to the system"""
    tags_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
    policy_id = auto_responder.add_policy(title, content, category, tags_list, priority)
    return f"Policy added successfully with ID: {policy_id}"

@mcp.tool()
def search_policies(query: str, max_results: int = 5) -> str:
    """Search for relevant company policies using semantic search"""
    results = auto_responder.search_policies(query, max_results)
    
    if not results:
        return "No relevant policies found."
    
    response = f"Found {len(results)} relevant policies:\n\n"
    for i, (policy, score) in enumerate(results, 1):
        response += f"{i}. {policy.title} (Score: {score:.2f})\n"
        response += f"   Category: {policy.category}\n"
        response += f"   Content: {policy.content[:200]}...\n\n"
    
    return response

@mcp.tool()
def add_email_template(name: str, subject: str, body: str, category: str, variables: str = "") -> str:
    """Add a new email response template"""
    variables_list = [var.strip() for var in variables.split(",") if var.strip()]
    template_id = auto_responder.add_email_template(name, subject, body, category, variables_list)
    return f"Email template added successfully with ID: {template_id}"

@mcp.tool()
def receive_email(sender: str, subject: str, body: str) -> str:
    """Receive and store an incoming email"""
    email_id = auto_responder.store_incoming_email(sender, subject, body)
    return f"Email received and stored with ID: {email_id}"

@mcp.tool()
def process_emails(batch_size: int = 10) -> str:
    """Process batch of unprocessed emails and generate responses"""
    responses = auto_responder.process_batch_emails(batch_size)
    
    if not responses:
        return "No emails to process."
    
    result = f"Processed {len(responses)} emails:\n\n"
    for response in responses:
        result += f"Response ID: {response.id}\n"
        result += f"To: {response.recipient}\n"
        result += f"Subject: {response.subject}\n"
        result += f"Status: {response.status}\n\n"
    
    return result

@mcp.tool()
def send_email_response(response_id: str, smtp_server: str = "smtp.gmail.com", smtp_port: int = 587, 
                       username: str = "", password: str = "") -> str:
    """Send an email response (mock implementation)"""
    # In a real implementation, this would use SMTP to send emails
    # For this demo, we'll just mark as sent
    
    conn = sqlite3.connect(auto_responder.db_path)
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM email_responses WHERE id = ?', (response_id,))
    row = cursor.fetchone()
    
    if not row:
        return f"Response {response_id} not found."
    
    # Mark as sent
    cursor.execute('UPDATE email_responses SET status = "sent", sent_at = ? WHERE id = ?', 
                  (datetime.now(), response_id))
    
    # Mark original email as response sent
    cursor.execute('UPDATE incoming_emails SET response_sent = TRUE WHERE id = ?', (row[1],))
    
    conn.commit()
    conn.close()
    
    return f"Email response {response_id} sent successfully to {row[5]}"

@mcp.tool()
def get_system_statistics() -> str:
    """Get system statistics and performance metrics"""
    stats = auto_responder.get_system_stats()
    
    result = "Email Auto-Responder System Statistics:\n\n"
    result += f"📋 Company Policies: {stats['policies']}\n"
    result += f"📝 Email Templates: {stats['templates']}\n"
    result += f"📧 Total Emails: {stats['total_emails']}\n"
    result += f"✅ Processed Emails: {stats['processed_emails']}\n"
    result += f"⏳ Pending Emails: {stats['pending_emails']}\n"
    result += f"📤 Sent Responses: {stats['sent_responses']}\n"
    result += f"🔄 Cache Size: {stats['cache_size']}\n"
    result += f"🔍 Search Index Size: {stats['search_index_size']}\n"
    
    return result

@mcp.tool()
def get_pending_emails() -> str:
    """Get list of pending emails that need responses"""
    unprocessed = auto_responder.get_unprocessed_emails()
    
    if not unprocessed:
        return "No pending emails."
    
    result = f"Pending Emails ({len(unprocessed)}):\n\n"
    for email in unprocessed:
        result += f"ID: {email.id}\n"
        result += f"From: {email.sender}\n"
        result += f"Subject: {email.subject}\n"
        result += f"Received: {email.received_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
        result += f"Body: {email.body[:100]}...\n\n"
    
    return result

@mcp.tool()
def simulate_email_scenario(scenario: str = "customer_complaint") -> str:
    """Simulate different email scenarios for testing"""
    scenarios = {
        "customer_complaint": {
            "sender": "unhappy.customer@example.com",
            "subject": "Terrible service - want refund",
            "body": "I am extremely dissatisfied with your service. The product doesn't work as advertised and I want a full refund immediately. This is unacceptable!"
        },
        "billing_inquiry": {
            "sender": "john.smith@example.com",
            "subject": "Question about my invoice",
            "body": "Hi, I received an invoice for $299 but I'm not sure what this charge is for. Could you please provide more details about this billing?"
        },
        "support_request": {
            "sender": "help.needed@example.com",
            "subject": "Need help with setup",
            "body": "Hello, I'm having trouble setting up my account. The instructions aren't clear and I keep getting error messages. Can someone help me?"
        }
    }
    
    if scenario not in scenarios:
        return f"Unknown scenario. Available scenarios: {', '.join(scenarios.keys())}"
    
    email_data = scenarios[scenario]
    
    # Store the email
    email_id = auto_responder.store_incoming_email(
        email_data["sender"], 
        email_data["subject"], 
        email_data["body"]
    )
    
    # Process it
    responses = auto_responder.process_batch_emails(1)
    
    result = f"Simulated {scenario} scenario:\n\n"
    result += f"📧 Email ID: {email_id}\n"
    result += f"From: {email_data['sender']}\n"
    result += f"Subject: {email_data['subject']}\n\n"
    
    if responses:
        response = responses[0]
        result += f"Generated Response:\n"
        result += f"Subject: {response.subject}\n"
        result += f"Body: {response.body[:300]}...\n"
    
    return result

if __name__ == "__main__":
    # Add some sample data
    print("Initializing Email Auto-Responder...")
    
    # Add sample policies
    auto_responder.add_policy(
        "Refund Policy",
        "We offer full refunds within 30 days of purchase for unused products. Please contact customer service with your order number.",
        "customer_service",
        ["refund", "return", "policy"],
        5
    )
    
    auto_responder.add_policy(
        "Billing Support",
        "For billing inquiries, please provide your account number and invoice date. We typically respond within 24 hours.",
        "billing",
        ["billing", "invoice", "payment"],
        4
    )
    
    auto_responder.add_policy(
        "Technical Support",
        "Our technical support team is available Monday-Friday 9am-5pm. For urgent issues, please call our hotline.",
        "support",
        ["technical", "support", "help"],
        3
    )
    
    # Add sample templates
    auto_responder.add_email_template(
        "Refund Request Response",
        "Re: {original_subject} - Refund Request",
        "Dear {customer_name},\n\nThank you for contacting us regarding your refund request.\n\n{relevant_policies}\n\nPlease reply with your order number and we will process your request within 2-3 business days.\n\nBest regards,\nCustomer Service Team",
        "refund",
        ["customer_name", "original_subject", "relevant_policies"]
    )
    
    auto_responder.add_email_template(
        "Billing Inquiry Response",
        "Re: {original_subject} - Billing Inquiry",
        "Dear {customer_name},\n\nThank you for your billing inquiry dated {current_date}.\n\n{relevant_policies}\n\nWe will review your account and respond with detailed information within 24 hours.\n\nBest regards,\nBilling Department",
        "billing",
        ["customer_name", "original_subject", "current_date", "relevant_policies"]
    )
    
    auto_responder.add_email_template(
        "General Support Response",
        "Re: {original_subject} - We're Here to Help",
        "Dear {customer_name},\n\nThank you for contacting us. We have received your message and our support team will respond within 24 hours.\n\n{relevant_policies}\n\nIf this is urgent, please call our support hotline.\n\nBest regards,\nSupport Team",
        "support",
        ["customer_name", "original_subject", "relevant_policies"]
    )
    
    auto_responder.add_email_template(
        "Complaint Response",
        "Re: {original_subject} - Your Feedback is Important",
        "Dear {customer_name},\n\nWe sincerely apologize for your experience and take your feedback seriously.\n\n{relevant_policies}\n\nA senior member of our team will contact you within 24 hours to resolve this matter.\n\nBest regards,\nCustomer Experience Team",
        "complaint",
        ["customer_name", "original_subject", "relevant_policies"]
    )
    
    auto_responder.add_email_template(
        "General Response",
        "Re: {original_subject} - Thank You for Contacting Us",
        "Dear {customer_name},\n\nThank you for your email. We have received your message and will respond shortly.\n\n{relevant_policies}\n\nBest regards,\nCustomer Service Team",
        "general",
        ["customer_name", "original_subject", "relevant_policies"]
    )
    
    print("Sample data loaded successfully!")
    print("Starting MCP server...")
    
    # Run the MCP server
    mcp.run()