#!/usr/bin/env python3
"""
Communication MCPs Demo Script

This script demonstrates the usage of the Task 23 Communication MCPs with Sequential Thinking integration:
- EmailClientMCP with external service integration following RAG-MCP extensibility
- APIClientMCP for REST/GraphQL interactions using general-purpose integration
- CalendarSchedulingMCP for time management and external service coordination
- MessagingMCP for various communication platforms

All MCPs include Sequential Thinking integration for complex coordination tasks.

@module CommunicationMCPsDemo
@author Enhanced Alita KGoT Team
@date 2025
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "mcp_toolbox"))

# Import the Communication MCPs
from communication_mcps import (
    EmailClientMCP,
    APIClientMCP,
    CalendarSchedulingMCP,
    MessagingMCP,
    EmailClientConfig,
    APIClientConfig,
    CalendarConfig,
    MessagingConfig,
    create_communication_mcps,
    get_communication_mcp_specifications
)

# Import Sequential Thinking integration (if available)
SEQUENTIAL_THINKING_AVAILABLE = False
SequentialThinkingIntegration = None

# Note: Sequential Thinking integration disabled for demo to avoid import conflicts
# In production, this would be properly configured with the full system


def demonstrate_email_client():
    """Demonstrate EmailClientMCP functionality"""
    print("\n=== Email Client MCP Demo ===")
    
    # Configure email client
    email_config = EmailClientConfig(
        smtp_server="smtp.gmail.com",
        smtp_port=587,
        username="demo@example.com",
        enable_sequential_thinking=True,
        complexity_threshold=7.0
    )
    
    # Create email client MCP
    email_mcp = EmailClientMCP(config=email_config)
    
    # Demo 1: Simple email send
    print("\n1. Sending a simple email...")
    email_params = {
        "operation": "send",
        "to_addresses": ["recipient@example.com"],
        "subject": "Test Email from Communication MCP",
        "body": "This is a test email sent using the EmailClientMCP with Sequential Thinking integration.",
        "use_sequential_thinking": False
    }
    
    result = email_mcp._run(**email_params)
    print(f"Result: {result}")
    
    # Demo 2: Complex email operation (triggers Sequential Thinking)
    print("\n2. Sending complex email (multiple recipients, attachments)...")
    complex_email_params = {
        "operation": "send",
        "to_addresses": ["user1@example.com", "user2@example.com", "user3@example.com", 
                        "user4@example.com", "user5@example.com", "user6@example.com"],
        "subject": "Complex Email with Sequential Thinking",
        "body": "This complex email should trigger Sequential Thinking due to multiple recipients.",
        "attachments": ["/tmp/file1.pdf", "/tmp/file2.docx", "/tmp/file3.xlsx", "/tmp/file4.pptx"],
        "use_sequential_thinking": True
    }
    
    result = email_mcp._run(**complex_email_params)
    print(f"Result: {result}")


def demonstrate_api_client():
    """Demonstrate APIClientMCP functionality"""
    print("\n=== API Client MCP Demo ===")
    
    # Configure API client
    api_config = APIClientConfig(
        base_url="https://jsonplaceholder.typicode.com",
        auth_type="none",
        enable_sequential_thinking=True,
        complexity_threshold=6.0
    )
    
    # Create API client MCP
    api_mcp = APIClientMCP(config=api_config)
    
    # Demo 1: Simple GET request
    print("\n1. Making a simple GET request...")
    get_params = {
        "operation": "get",
        "endpoint": "/posts/1",
        "use_sequential_thinking": False
    }
    
    result = api_mcp._run(**get_params)
    print(f"Result: {result}")
    
    # Demo 2: Complex GraphQL operation (triggers Sequential Thinking)
    print("\n2. Making complex GraphQL query...")
    graphql_params = {
        "operation": "graphql",
        "endpoint": """
        query {
            posts {
                id
                title
                body
                user {
                    name
                    email
                    company {
                        name
                    }
                }
            }
        }
        """,
        "data": {"limit": 10, "offset": 0},
        "use_sequential_thinking": True
    }
    
    result = api_mcp._run(**graphql_params)
    print(f"Result: {result}")


def demonstrate_calendar_scheduling():
    """Demonstrate CalendarSchedulingMCP functionality"""
    print("\n=== Calendar Scheduling MCP Demo ===")
    
    # Configure calendar
    calendar_config = CalendarConfig(
        calendar_service="icalendar",
        timezone="UTC",
        enable_conflict_detection=True,
        enable_sequential_thinking=True,
        complexity_threshold=5.0
    )
    
    # Create calendar MCP
    calendar_mcp = CalendarSchedulingMCP(config=calendar_config)
    
    # Demo 1: Simple event creation
    print("\n1. Creating a simple calendar event...")
    event_params = {
        "operation": "create_event",
        "event_title": "Team Meeting",
        "start_time": "2025-01-25T10:00:00Z",
        "end_time": "2025-01-25T11:00:00Z",
        "attendees": ["alice@example.com", "bob@example.com"],
        "location": "Conference Room A",
        "use_sequential_thinking": False
    }
    
    result = calendar_mcp._run(**event_params)
    print(f"Result: {result}")
    
    # Demo 2: Complex recurring event (triggers Sequential Thinking)
    print("\n2. Creating complex recurring event with many attendees...")
    complex_event_params = {
        "operation": "create_event",
        "event_title": "All-Hands Meeting",
        "start_time": "2025-01-27T14:00:00Z",
        "end_time": "2025-01-27T16:00:00Z",
        "attendees": ["alice@example.com", "bob@example.com", "charlie@example.com", 
                     "diana@example.com", "eve@example.com", "frank@example.com"],
        "description": "Monthly all-hands meeting for project updates",
        "location": "Main Conference Room",
        "recurrence": "monthly",
        "use_sequential_thinking": True
    }
    
    result = calendar_mcp._run(**complex_event_params)
    print(f"Result: {result}")
    
    # Demo 3: Conflict detection
    print("\n3. Checking for scheduling conflicts...")
    conflict_params = {
        "operation": "check_conflicts",
        "start_time": "2025-01-25T10:30:00Z",
        "end_time": "2025-01-25T11:30:00Z",
        "attendees": ["alice@example.com", "bob@example.com"]
    }
    
    result = calendar_mcp._run(**conflict_params)
    print(f"Result: {result}")


def demonstrate_messaging():
    """Demonstrate MessagingMCP functionality"""
    print("\n=== Messaging MCP Demo ===")
    
    # Configure messaging
    messaging_config = MessagingConfig(
        supported_platforms=["slack", "discord", "teams", "telegram"],
        enable_cross_platform=True,
        enable_sequential_thinking=True,
        complexity_threshold=6.0
    )
    
    # Create messaging MCP
    messaging_mcp = MessagingMCP(config=messaging_config)
    
    # Demo 1: Simple message send
    print("\n1. Sending a simple message...")
    message_params = {
        "operation": "send_message",
        "platform": "slack",
        "recipient": "#general",
        "message": "Hello from the MessagingMCP!",
        "priority": "normal",
        "use_sequential_thinking": False
    }
    
    result = messaging_mcp._run(**message_params)
    print(f"Result: {result}")
    
    # Demo 2: Auto-platform selection with high priority (triggers Sequential Thinking)
    print("\n2. Sending urgent message with auto-platform selection...")
    urgent_message_params = {
        "operation": "send_message",
        "platform": "auto",
        "recipient": "emergency-team",
        "message": "URGENT: System alert requires immediate attention. Please respond ASAP.",
        "priority": "urgent",
        "attachments": ["/tmp/system_log.txt", "/tmp/error_report.pdf"],
        "use_sequential_thinking": True
    }
    
    result = messaging_mcp._run(**urgent_message_params)
    print(f"Result: {result}")
    
    # Demo 3: Thread management
    print("\n3. Managing message thread...")
    thread_params = {
        "operation": "manage_thread",
        "platform": "slack",
        "thread_id": "thread_12345",
        "message": "Adding follow-up message to thread"
    }
    
    result = messaging_mcp._run(**thread_params)
    print(f"Result: {result}")


def demonstrate_sequential_thinking_integration():
    """Demonstrate Sequential Thinking integration across all MCPs"""
    print("\n=== Sequential Thinking Integration Demo ===")
    
    if not SEQUENTIAL_THINKING_AVAILABLE:
        print("Sequential Thinking integration not available. Install the sequential_thinking_integration module.")
        return
    
    # Initialize Sequential Thinking
    print("\n1. Initializing Sequential Thinking integration...")
    sequential_thinking = SequentialThinkingIntegration({
        'complexityThreshold': 6,
        'errorThreshold': 3,
        'maxThoughts': 10,
        'enableCrossSystemCoordination': True,
        'enableAdaptiveThinking': True
    })
    
    # Create all MCPs with Sequential Thinking
    print("\n2. Creating Communication MCPs with Sequential Thinking...")
    mcps = create_communication_mcps(
        email_config=EmailClientConfig(enable_sequential_thinking=True),
        api_config=APIClientConfig(enable_sequential_thinking=True),
        calendar_config=CalendarConfig(enable_sequential_thinking=True),
        messaging_config=MessagingConfig(enable_sequential_thinking=True),
        sequential_thinking=sequential_thinking
    )
    
    print(f"Created {len(mcps)} MCPs with Sequential Thinking integration:")
    for mcp in mcps:
        print(f"  - {mcp.name}: {mcp.description[:60]}...")


def demonstrate_mcp_specifications():
    """Demonstrate MCP specifications and metadata"""
    print("\n=== MCP Specifications Demo ===")
    
    specifications = get_communication_mcp_specifications()
    
    print(f"\nCommunication MCP Specifications ({len(specifications)} tools):")
    for spec in specifications:
        print(f"\n{spec.name}:")
        print(f"  Category: {spec.category}")
        print(f"  Description: {spec.description}")
        print(f"  Capabilities: {spec.capabilities}")
        print(f"  Dependencies: {spec.dependencies}")
        print(f"  Sequential Thinking Enabled: {spec.sequential_thinking_enabled}")
        print(f"  Complexity Threshold: {spec.complexity_threshold}")


def main():
    """Main demonstration function"""
    print("Communication MCPs Demo - Task 23 Implementation")
    print("=" * 60)
    print("Demonstrating Core High-Value MCPs for Communication:")
    print("- EmailClientMCP with external service integration")
    print("- APIClientMCP for REST/GraphQL interactions")
    print("- CalendarSchedulingMCP for time management")
    print("- MessagingMCP for various communication platforms")
    print("All with Sequential Thinking integration for complex workflows")
    print("=" * 60)
    
    try:
        # Run all demonstrations
        demonstrate_email_client()
        demonstrate_api_client()
        demonstrate_calendar_scheduling()
        demonstrate_messaging()
        demonstrate_sequential_thinking_integration()
        demonstrate_mcp_specifications()
        
        print("\n" + "=" * 60)
        print("Communication MCPs Demo completed successfully!")
        print("All MCPs are ready for integration with the Alita-KGoT system.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nDemo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 