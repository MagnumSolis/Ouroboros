"""Reasoning Trace UI - Displays agent decision logs"""
import streamlit as st

def render_reasoning_trace(memory, interaction_id):
    """Render the agent reasoning trace for an interaction"""
    st.header("🧠 Agent Reasoning Trace")
    
    if not interaction_id:
        st.info("Start a conversation to see the reasoning trace.")
        return

    # Use synchronous wrapper for Streamlit compatibility
    try:
        import asyncio
        import nest_asyncio
        nest_asyncio.apply()
        
        # Create fresh loop for each call
        loop = asyncio.new_event_loop()
        try:
            logs = loop.run_until_complete(memory.get_agent_trace(interaction_id))
        finally:
            loop.close()
            
    except Exception as e:
        st.error(f"Error fetching trace: {e}")
        st.caption(f"Interaction ID: {interaction_id}")
        return
    
    if not logs:
        st.warning("No logs found for this interaction.")
        st.caption(f"Interaction ID: {interaction_id}")
        return
        
    for log in logs:
        with st.chat_message(name=log.agent_id, avatar="🤖"):
            st.write(f"**{log.agent_id.upper()}** - *{log.action}*")
            
            with st.expander("Details", expanded=False):
                st.markdown(f"**Input**: {log.input_summary}")
                st.markdown(f"**Reasoning**: {log.reasoning}")
                st.markdown(f"**Output**: {log.output_summary}")
                
                if log.metadata:
                    st.json(log.metadata)
            
            st.caption(f"Confidence: {log.confidence} | Time: {log.timestamp}")
            
    st.divider()
    st.caption(f"Interaction ID: {interaction_id}")
