
# This file will be appended to dashboard.py later
import streamlit as st
import os

def render_knowledge_hub(memory):
    st.header("📚 Knowledge Hub")
    st.caption("Central repository for financial schemes and policies")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF or Text files", 
            type=['pdf', 'txt', 'md'], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("Process & Ingest"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name}...")
                    
                    # Save temporarily
                    temp_path = os.path.join("data/uploads", file.name)
                    os.makedirs("data/uploads", exist_ok=True)
                    
                    with open(temp_path, "wb") as f:
                        f.write(file.getbuffer())
                    
                    # Ingest with language metadata (enables optional language filtering in future)
                    meta = {"source": "upload", "original_name": file.name, "language": "multi"}
                    
                    # We need to run async ingest in sync streamlit
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    success = loop.run_until_complete(
                        memory.ingest_file(temp_path, collection_name="knowledge_base", metadata=meta)
                    )
                    loop.close()
                    
                    if success:
                        st.success(f"✅ Indexed {file.name}")
                    else:
                        st.error(f"❌ Failed to index {file.name}")
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("Ingestion Complete!")

    with col2:
        st.subheader("Knowledge Stats")
        try:
             stats = memory.get_collection_stats()
             kb_stats = stats.get("knowledge_base", {})
             st.metric("Indexed Documents", kb_stats.get("points_count", 0))
             
             if st.button("🔄 Re-seed Defaults"):
                 import subprocess
                 import sys
                 subprocess.Popen([sys.executable, "scripts/ingest_knowledge.py"])
                 st.info("Seeding started in background...")
                 
        except Exception as e:
            st.error(f"Could not fetch stats: {e}")

    st.divider()
    
    # Document Browser - Use scroll to get ALL recent docs
    st.subheader("Recent Knowledge Entries")
    try:
        # Use scroll API to get actual recent entries (not semantic search)
        scroll_result, _ = memory.client.scroll(
            collection_name="knowledge_base",
            limit=10,
            with_payload=True
        )
        
        if scroll_result:
            for point in scroll_result:
                payload = point.payload or {}
                source = payload.get('source', payload.get('original_name', 'Unknown'))
                with st.expander(f"📄 {source}"):
                    content = payload.get("content", "")[:500]
                    st.write(content + ("..." if len(payload.get("content", "")) > 500 else ""))
                    st.caption(f"Language: {payload.get('language', 'N/A')} | Chunks: {payload.get('total_chunks', 1)}")
        else:
            st.info("No documents in knowledge base yet. Upload some files above!")
                
    except Exception as e:
        st.warning(f"Could not fetch entries: {e}")
