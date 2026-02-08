#!/usr/bin/env python3
"""
Demo Script: Agent Pipeline Visualization
==========================================

Shows the step-by-step agent execution trace with:
- Agent ID and action taken
- Input/Output summaries
- Reasoning chain
- Confidence scores
- Priority and sentiment tracking

Run: python scripts/demo_agent_pipeline.py
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.adapters import LLMAdapter, EmbeddingAdapter
from src.memory import MemoryManager
from src.agents import OrchestratorAgent, RetrievalAgent, FraudAgent, PerceptionAgent, CriticAgent
from src.agents.base import AgentContext
from loguru import logger


def print_header(text: str, char: str = "="):
    """Print a formatted header"""
    print(f"\n{char * 70}")
    print(f" {text}")
    print(f"{char * 70}")


def print_agent_log(log, index: int):
    """Pretty-print a single agent log entry"""
    agent_icons = {
        "orchestrator": "🧠",
        "fraud": "🕵️",
        "retrieval": "📚",
        "perception": "👁️",
        "critic": "✅"
    }
    
    icon = agent_icons.get(log.agent_id, "🤖")
    time_str = log.timestamp.strftime("%H:%M:%S.%f")[:-3]
    
    print(f"\n┌{'─' * 68}┐")
    print(f"│ {icon} STEP {index}: {log.agent_id.upper()} → {log.action:<30} │")
    print(f"├{'─' * 68}┤")
    print(f"│ ⏱️  Time: {time_str:<56} │")
    print(f"│ 🎯 Confidence: {log.confidence:<51.2f} │")
    
    if hasattr(log, 'priority') and log.priority:
        priority_colors = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
        print(f"│ 🚨 Priority: {priority_colors.get(log.priority, '⚪')} {log.priority:<50} │")
    
    if hasattr(log, 'user_sentiment') and log.user_sentiment:
        print(f"│ 🎭 Sentiment: {log.user_sentiment:<53} │")
    
    if hasattr(log, 'is_critical') and log.is_critical:
        print(f"│ ⚠️  CRITICAL FLAG RAISED!{'':^43} │")
    
    print(f"├{'─' * 68}┤")
    print(f"│ 📥 INPUT:{'':^58} │")
    input_lines = log.input_summary[:100].split('\n')
    for line in input_lines[:2]:
        print(f"│    {line[:62]:<62} │")
    
    print(f"│ 💭 REASONING:{'':^54} │")
    reasoning_lines = log.reasoning[:150].split('\n')
    for line in reasoning_lines[:3]:
        print(f"│    {line[:62]:<62} │")
    
    print(f"│ 📤 OUTPUT:{'':^57} │")
    output_lines = log.output_summary[:100].split('\n')
    for line in output_lines[:2]:
        print(f"│    {line[:62]:<62} │")
    
    print(f"└{'─' * 68}┘")


async def demo_agent_pipeline():
    """Demonstrate the agent pipeline with full trace"""
    
    print_header("🔍 AGENT PIPELINE DEMONSTRATION", "=")
    print("\nThis demo shows the multi-agent collaboration and reasoning chain.")
    print("Each agent's decision is logged to Qdrant's working_memory collection.\n")
    
    # Initialize
    print("📌 Step 1: Initializing system...")
    llm = LLMAdapter()
    embeddings = EmbeddingAdapter()
    memory = MemoryManager(embedding_adapter=embeddings)
    
    # Initialize all agents
    retrieval_agent = RetrievalAgent(llm, memory)
    fraud_agent = FraudAgent(llm, memory)
    perception_agent = PerceptionAgent(llm, memory)
    critic_agent = CriticAgent(llm, memory)
    
    orchestrator = OrchestratorAgent(llm, memory)
    orchestrator.register_agents([retrieval_agent, fraud_agent, perception_agent, critic_agent])
    
    print("✅ System initialized with 5 agents: Orchestrator, Retrieval, Fraud, Perception, Critic\n")
    
    # Test query that triggers multiple agents
    test_query = "Someone called saying my bank account will be blocked if I don't share OTP. Is this a scam?"
    
    print_header("📌 Step 2: Processing Query", "-")
    print(f"\n💬 User Query: \"{test_query}\"")
    print("\n⏳ Processing through agent pipeline...\n")
    
    # Create context and process
    context = AgentContext(
        user_input=test_query,
        language="en"
    )
    
    result = await orchestrator.process(context)
    interaction_id = context.interaction_id
    
    print_header("📌 Step 3: Agent Execution Trace", "-")
    print(f"\n🆔 Interaction ID: {interaction_id[:8]}...\n")
    
    # Fetch agent trace from Qdrant
    logs = await memory.get_agent_trace(interaction_id)
    
    if logs:
        print(f"📊 Found {len(logs)} agent actions in the trace:\n")
        for i, log in enumerate(logs, 1):
            print_agent_log(log, i)
    else:
        print("⚠️  No agent logs found. The orchestrator may not have logged the trace.")
    
    print_header("📌 Step 4: Final Response", "-")
    print(f"\n🤖 Orchestrator Final Output:\n")
    print("-" * 70)
    print(result.content)
    print("-" * 70)
    
    # Show metadata
    if result.metadata:
        print("\n📋 Response Metadata:")
        if result.metadata.get("is_fraud"):
            print("   🚨 FRAUD DETECTED: Yes")
        if result.metadata.get("plan"):
            print(f"   📝 Execution Plan: (see XML below)")
    
    print_header("📌 Step 5: Qdrant Storage Stats", "-")
    stats = memory.get_collection_stats()
    print("\n📊 Collection Statistics:")
    for name, info in stats.items():
        print(f"   • {name}: {info.get('points_count', 0)} entries")
    
    print_header("🎯 SUMMARY", "=")
    print("""
    This demo showed:
    
    ✅ Multi-Agent Collaboration
       - Orchestrator planned the execution
       - Fraud agent analyzed for scam indicators
       - Retrieval agent searched knowledge base
       - Critic agent validated the response
    
    ✅ Full Audit Trail
       - Every agent action logged to Qdrant
       - Input, reasoning, and output captured
       - Timestamps for execution ordering
    
    ✅ Sentiment & Priority Tracking
       - User emotion detected from text
       - Priority escalation for OTP/fraud queries
    
    ✅ Qdrant as Blackboard
       - Agents communicate via shared memory
       - Traces stored in working_memory collection
       - Fully auditable and retrievable
    """)
    
    print("\n💡 TIP: Compare this output with the Streamlit UI's 'Agent Pipeline' tab\n")


if __name__ == "__main__":
    asyncio.run(demo_agent_pipeline())
