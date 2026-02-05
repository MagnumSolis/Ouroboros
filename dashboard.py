"""
Sahayak Dashboard - Streamlit UI
The Vernacular Financial Sentinel
"""

import streamlit as st
import asyncio
from datetime import datetime
import json
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sahayak.config import settings
from sahayak.adapters import LLMAdapter, EmbeddingAdapter
from sahayak.memory import MemoryManager
from sahayak.agents import (
    OrchestratorAgent, RetrievalAgent, FraudAgent, 
    PerceptionAgent, CriticAgent, AgentContext
)

# Page config
st.set_page_config(
    page_title="Sahayak - Financial Sentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    /* Dark theme enhancements */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Hero header */
    .hero-title {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0;
    }
    
    .hero-subtitle {
        color: #a0a0b0;
        text-align: center;
        font-size: 1.2rem;
        margin-top: 0;
    }
    
    /* Agent cards */
    .agent-card {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .agent-card:hover {
        border-color: #667eea;
        transform: translateY(-2px);
    }
    
    .status-online {
        color: #00ff88;
        font-weight: bold;
    }
    
    .status-offline {
        color: #ff4444;
        font-weight: bold;
    }
    
    /* Chat styling */
    .chat-message {
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin-left: 20%;
    }
    
    .assistant-message {
        background: rgba(255,255,255,0.08);
        margin-right: 20%;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Fraud alert */
    .fraud-alert {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        padding: 1rem;
        border-radius: 12px;
        color: white;
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    /* Metrics */
    .metric-card {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "interaction_count" not in st.session_state:
        st.session_state.interaction_count = 0
    if "fraud_alerts" not in st.session_state:
        st.session_state.fraud_alerts = 0
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "agents" not in st.session_state:
        st.session_state.agents = {}


@st.cache_resource
def initialize_system():
    """Initialize the agent system (cached for performance)"""
    try:
        # Initialize adapters
        llm = LLMAdapter()
        embeddings = EmbeddingAdapter()
        memory = MemoryManager(embedding_adapter=embeddings)
        
        # Initialize agents
        orchestrator = OrchestratorAgent(llm=llm, memory=memory)
        retrieval = RetrievalAgent(llm=llm, memory=memory)
        fraud = FraudAgent(llm=llm, memory=memory)
        perception = PerceptionAgent(llm=llm, memory=memory)
        critic = CriticAgent(llm=llm, memory=memory)
        
        # Register agents with orchestrator
        orchestrator.register_agents([retrieval, fraud, perception, critic])
        
        return {
            "llm": llm,
            "memory": memory,
            "orchestrator": orchestrator,
            "agents": {
                "orchestrator": orchestrator,
                "retrieval": retrieval,
                "fraud": fraud,
                "perception": perception,
                "critic": critic
            },
            "status": "online"
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def process_query(system: dict, query: str, language: str = "hi") -> dict:
    """Process a user query through the orchestrator"""
    import uuid
    
    context = AgentContext(
        interaction_id=str(uuid.uuid4()),
        user_input=query,
        language=language,
        modality="text"
    )
    
    result = await system["orchestrator"].process(context)
    
    return {
        "response": result.content,
        "success": result.success,
        "metadata": result.metadata,
        "is_fraud": result.metadata.get("plan", {}).get("requires_fraud_check", False)
    }


def render_header():
    """Render the hero header"""
    st.markdown('<h1 class="hero-title">🛡️ Sahayak</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">The Vernacular Financial Sentinel • वित्तीय सुरक्षा सहायक</p>', unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with system status and controls"""
    with st.sidebar:
        st.markdown("### ⚙️ System Status")
        
        system = st.session_state.get("system", {})
        
        if system.get("status") == "online":
            st.success("🟢 System Online")
            
            # Show agent status
            st.markdown("#### Agent Status")
            agents = system.get("agents", {})
            for name, agent in agents.items():
                state = getattr(agent, 'state', 'IDLE')
                icon = "🟢" if str(state) == "AgentState.IDLE" else "🔄"
                st.markdown(f"{icon} **{name.title()}**: `{state}`")
        else:
            st.error("🔴 System Offline")
            if system.get("error"):
                st.error(f"Error: {system.get('error')}")
        
        st.markdown("---")
        
        # Language selector
        st.markdown("#### 🌐 Language / भाषा")
        language = st.radio(
            "Select language",
            ["hi", "en", "mixed"],
            format_func=lambda x: {"hi": "हिंदी", "en": "English", "mixed": "Mixed/मिश्रित"}[x],
            label_visibility="collapsed"
        )
        st.session_state.language = language
        
        st.markdown("---")
        
        # Stats
        st.markdown("#### 📊 Session Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Queries", st.session_state.interaction_count)
        with col2:
            st.metric("Alerts", st.session_state.fraud_alerts, delta_color="inverse")
        
        st.markdown("---")
        
        # Clear chat
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def render_agent_panel():
    """Render the agent monitoring panel"""
    st.markdown("### 🤖 Agent Activity Monitor")
    
    agents_info = [
        ("Orchestrator", "🧠", "Plans and coordinates workflow"),
        ("Retrieval", "📚", "Searches knowledge base"),
        ("Fraud", "🔍", "Detects scams and threats"),
        ("Perception", "👁️", "Processes audio/images"),
        ("Critic", "✅", "Validates responses"),
    ]
    
    cols = st.columns(5)
    for i, (name, icon, desc) in enumerate(agents_info):
        with cols[i]:
            st.markdown(f"""
            <div class="agent-card">
                <h3 style="margin:0">{icon}</h3>
                <p style="margin:0;font-weight:bold">{name}</p>
                <p style="margin:0;font-size:0.8rem;color:#888">{desc}</p>
                <p class="status-online">● Ready</p>
            </div>
            """, unsafe_allow_html=True)


def render_chat_interface():
    """Render the main chat interface"""
    st.markdown("### 💬 Chat with Sahayak")
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("is_fraud"):
                st.markdown(f'<div class="fraud-alert">⚠️ FRAUD ALERT DETECTED</div>', unsafe_allow_html=True)
            st.markdown(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("आपका प्रश्न / Your question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.interaction_count += 1
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process with orchestrator
        with st.chat_message("assistant"):
            with st.spinner("सोच रहा हूं... / Thinking..."):
                system = st.session_state.get("system", {})
                
                if system.get("status") == "online":
                    try:
                        result = asyncio.run(process_query(
                            system, 
                            prompt,
                            st.session_state.get("language", "hi")
                        ))
                        
                        response = result["response"]
                        is_fraud = result.get("is_fraud", False)
                        
                        if is_fraud:
                            st.session_state.fraud_alerts += 1
                            st.markdown(f'<div class="fraud-alert">⚠️ संभावित धोखाधड़ी / Potential Fraud Detected</div>', unsafe_allow_html=True)
                        
                        st.markdown(response)
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "is_fraud": is_fraud
                        })
                        
                    except Exception as e:
                        error_msg = f"Error processing request: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })
                else:
                    st.error("System not initialized. Please check your API keys.")


def render_demo_queries():
    """Render demo query buttons"""
    st.markdown("### 🎯 Try These Examples")
    
    examples = [
        ("🏦 PM Jan Dhan", "PM Jan Dhan Yojana के बारे में बताइए"),
        ("💳 KYC Update", "किसी ने call करके मेरा KYC update करने को कहा"),
        ("🌾 PM Kisan", "PM Kisan Samman Nidhi eligibility kya hai?"),
        ("📱 OTP Scam", "Someone asked for my OTP over WhatsApp, is it safe?"),
        ("💰 Mudra Loan", "MUDRA loan के लिए कैसे apply करें?"),
    ]
    
    cols = st.columns(5)
    for i, (label, query) in enumerate(examples):
        with cols[i]:
            if st.button(label, use_container_width=True, key=f"demo_{i}"):
                st.session_state.demo_query = query
                st.rerun()


def main():
    """Main application entry point"""
    init_session_state()
    
    # Initialize system
    if not st.session_state.initialized:
        with st.spinner("Initializing Sahayak System..."):
            system = initialize_system()
            st.session_state.system = system
            st.session_state.initialized = True
    
    # Check for demo query
    if "demo_query" in st.session_state:
        query = st.session_state.pop("demo_query")
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.interaction_count += 1
    
    # Render UI components
    render_header()
    render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "🤖 Agents", "📊 Blackboard"])
    
    with tab1:
        render_demo_queries()
        st.markdown("---")
        render_chat_interface()
    
    with tab2:
        render_agent_panel()
        
        st.markdown("### 📋 Recent Agent Activity")
        if st.session_state.messages:
            last_msg = st.session_state.messages[-1]
            if last_msg.get("role") == "assistant":
                st.json(last_msg.get("metadata", {"status": "No recent activity"}))
        else:
            st.info("No agent activity yet. Start a conversation!")
    
    with tab3:
        st.markdown("### 🗃️ Memory Blackboard")
        st.info("Connect to Qdrant to view memory entries")
        
        # Show collection info
        cols = st.columns(4)
        collections = [
            ("episodic_memory", "🧠", "Interaction logs"),
            ("knowledge_base", "📚", "Schemes & policies"),
            ("fraud_patterns", "🔍", "Scam patterns"),
            ("working_memory", "⚡", "Active tasks"),
        ]
        
        for i, (name, icon, desc) in enumerate(collections):
            with cols[i]:
                st.metric(f"{icon} {name.replace('_', ' ').title()}", "0 entries", desc)


if __name__ == "__main__":
    main()
