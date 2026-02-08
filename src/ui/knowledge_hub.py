
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
    
    # Document Browser (Mockup for now as we can't easily query all docs without scroll API)
    st.subheader("Recent Knowledge Entries")
    try:
        # Just search for * to get some docs
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        recent = loop.run_until_complete(
            memory.search("knowledge_base", "scheme", limit=5)
        )
        loop.close()
        
        for doc in recent:
            with st.expander(f"📄 {doc.get('source', 'Unknown')} (Score: {doc.get('score', 0):.2f})"):
                st.write(doc.get("content", "")[:500] + "...")
                st.json(doc)
                
    except Exception as e:
        st.warning(f"No documents found or error: {e}")
