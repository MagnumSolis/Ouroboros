"""Pipeline Viewer - Unified Agent Visualization"""
import streamlit as st
import asyncio
import nest_asyncio
from datetime import datetime

# Apply nest_asyncio to allow nested event loops (required for Streamlit + Async)
nest_asyncio.apply()

def get_status_color(priority):
    if priority == "CRITICAL": return "🔴"
    if priority == "HIGH": return "🟠"
    if priority == "MEDIUM": return "🟡"
    return "🟢"

def render_pipeline_viewer(memory):
    """Render the unified agent pipeline view"""
    st.markdown("## 🔍 Agent Pipeline")
    st.caption("Real-time view of agent collaboration, reasoning, and handoffs")

    # Get recent interactions
    if "messages" in st.session_state and st.session_state.messages:
        # Get the ID of the last assistant message
        last_id = None
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "assistant" and "interaction_id" in msg:
                last_id = msg["interaction_id"]
                break
        
        if not last_id:
            st.info("Start a conversation to see the pipeline active.")
            return
            
        # UI: Interaction Selector (optional if we want history)
        # For now, just show the latest one by default
        
        # Fetch logs via Async
        with st.spinner("Fetching agent trace..."):
             try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                logs = loop.run_until_complete(memory.get_agent_trace(last_id))
                loop.close()
             except Exception as e:
                st.error(f"Error fetching traces: {e}")
                return

        if not logs:
            st.warning("No agent logs found for this interaction yet.")
            return

        # 1. Header Params (Sentiment, Priority)
        # We usually get this from the first log (Orchestrator PLAN)
        plan_log = next((l for l in logs if l.action == "PLAN"), None)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Interaction ID**")
            st.caption(f"`{last_id[:8]}...`")
        
        with col2:
            sentiment = getattr(plan_log, "user_sentiment", "N/A") if plan_log else "N/A"
            st.markdown(f"**🎭 Sentiment**")
            st.info(f"{sentiment.title()}" if sentiment else "Neutral")
            
        with col3:
            priority = getattr(plan_log, "priority", "LOW") if plan_log else "LOW"
            icon = get_status_color(priority)
            st.markdown(f"**🚨 Priority**")
            st.markdown(f"### {icon} {priority}")

        st.divider()

        # 2. Timeline Visualization
        st.markdown("### ⚡ Execution Flow")
        
        # CSS for connecting lines
        st.markdown("""
        <style>
            .step-container {
                border-left: 2px solid #333;
                margin-left: 20px;
                padding-left: 20px;
                padding-bottom: 20px;
                position: relative;
            }
            .step-dot {
                position: absolute;
                left: -9px;
                top: 0;
                width: 16px;
                height: 16px;
                border-radius: 50%;
                background: #667eea;
                border: 2px solid #0e1117;
            }
            .step-header {
                font-weight: bold;
                font-size: 1.1rem;
                margin-bottom: 5px;
            }
            .step-meta {
                font-size: 0.8rem;
                color: #888;
                margin-bottom: 10px;
            }
            .log-box {
                background: rgba(255,255,255,0.05);
                border-radius: 8px;
                padding: 10px;
                font-family: monospace;
                font-size: 0.9rem;
            }
        </style>
        """, unsafe_allow_html=True)
        
        for i, log in enumerate(logs):
            # Determine icon based on agent
            agent_icons = {
                "orchestrator": "🧠",
                "fraud": "🕵️",
                "retrieval": "📚",
                "perception": "👁️",
                "critic": "✅"
            }
            icon = agent_icons.get(log.agent_id, "🤖")
            
            # Format timestamp
            time_str = log.timestamp.strftime("%H:%M:%S")
            
            with st.container():
                col_icon, col_content = st.columns([1, 15])
                with col_icon:
                    st.markdown(f"## {icon}")
                
                with col_content:
                    st.markdown(f"**{log.agent_id.upper()}** — *{log.action}*")
                    st.caption(f"🕒 {time_str} | Conf: {log.confidence}")
                    
                    with st.expander(f"Details: {log.input_summary[:40]}...", expanded=(i==len(logs)-1)):
                        # Priority badge on individual log if critical
                        if hasattr(log, "is_critical") and log.is_critical:
                            st.error("⚠️ CRITICAL FLAG RAISED")
                        
                        st.markdown("**📥 Input:**")
                        st.code(log.input_summary, language="text")
                        
                        st.markdown("**🧠 Reasoning:**")
                        st.info(log.reasoning)
                        
                        st.markdown("**📤 Output:**")
                        st.code(log.output_summary, language="text")
                        
                        if log.metadata:
                            st.markdown("**Metadata:**")
                            st.json(log.metadata)

    else:
        st.info("Waiting for conversation to start...")
        st.markdown("""
        <div style="text-align: center; padding: 50px; opacity: 0.5;">
            <h1>🕸️</h1>
            <p>Ready to visualize agent interaction network</p>
        </div>
        """, unsafe_allow_html=True)
