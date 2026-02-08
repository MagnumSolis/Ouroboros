# Sahayak Demo Guide

This guide outlines how to demonstrate the key capabilities of the **Sahayak Financial Sentinel** system, specifically focusing on **Document Ingestion**, **No Hallucination (RAG)**, and **Multimodal Interaction**.

## 1. Prerequisites
- Ensure the application is running:
  ```bash
  streamlit run dashboard.py
  ```
- Access the dashboard at `http://localhost:8501`.
- **Verify System Status**:
  - Look at the Sidebar.
  - Ensure **System Status** is **🟢 Online**.
  - Check that **Deepgram** is listed under "Active Providers" (this confirms the recent fix).

---

## 2. Feature Demo: Ingestion (Knowledge Hub)

**Objective**: Show how easily new knowledge (PDFs, text files) can be added to the system.

1.  **Navigate**: Click on the **"📚 Knowledge Hub"** tab at the top of the main area.
2.  **Upload**:
    -   Locate the **"Upload Documents"** section.
    -   Drag and drop a PDF file (e.g., a government scheme circular or a sample policy document).
    -   *Tip: Use a file that contains specific, non-public facts to better demonstrate retrieval.*
3.  **Ingest**:
    -   Click the **"Process & Ingest"** button.
    -   Observe the progress bar and status messages.
4.  **Verify**:
    -   Wait for the **"✅ Indexed [filename]"** success message.
    -   Check the **"Knowledge Stats"** panel on the right. The "Indexed Documents" count should increase.
    -   Expand the **"Recent Knowledge Entries"** section below to see chunks of your uploaded document.

---

## 3. Feature Demo: RAG & "No Hallucination"

**Objective**: Demonstrate that the system answers based *only* on facts and refuses to invent information.

### Step A: The "Unknown" Test (No Hallucination)
*Perform this test BEFORE ingesting a specific unique document, or ask about something fictitious.*

1.  Go to the **"💬 Chat"** tab.
2.  **Query**: Ask a question about a fictitious scheme or a detail not in the knowledge base.
    -   *Example*: "What is the 'Galactic Subsidy Scheme' for Mars colonization?"
3.  **Expected Result**:
    -   The system should **NOT** invent a scheme.
    -   It should reply along the lines of: *"I cannot find information about a 'Galactic Subsidy Scheme' in my knowledge base."* or provide a general answer stating it doesn't know.
    -   **This proves the "No Hallucination" guardrail is active.**

### Step B: The "Retrieval" Test (RAG)
*Perform this test AFTER ingesting a document (from Section 2).*

1.  **Query**: Ask a specific question whose answer is found *only* in the uploaded document.
    -   *Example (if you uploaded a PMJDY circular)*: "What is the overdraft limit for PMJDY accounts opened after August 2018?"
2.  **Expected Result**:
    -   The system should provide the **exact figure** (e.g., ₹10,000) citing the document context.
    -   It may show a "Thinking..." step where the **Retrieval Agent** is active.

---

## 4. Feature Demo: Multimodal (Speech-to-Text)

**Objective**: Show the fixed Deepgram integration.

1.  **Navigate**: Go to the **"💬 Chat"** tab.
2.  **Action**: Click the microphone icon labeled **"🎤 Speak now / बोलें"**.
3.  **Speak**: Ask a question in Hindi or English (e.g., "PM Kisan Samman Nidhi ke liye apply kaise karein?").
4.  **Verify**:
    -   The system should transcribe your audio accurately (displayed in the chat as user text).
    -   The **Orchestrator** should then process the query and provide an audio (TTS) and text response.
    -   *Note: If Deepgram was broken, this step would have failed or fallen back to a slower local model.*

---

## Troubleshooting
- **Deepgram Error**: If you see "Deepgram SDK not installed", ensure you have restarted the app after the latest code fix.
- **Ingestion Stuck**: If the progress bar doesn't move, refresh the page. This is a known limitation of the synchronous UI wrapper around async ingestion.
