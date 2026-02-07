# 🛡️ Sahayak - The Vernacular Financial Sentinel

A Multi-Agent System for Financial Inclusion and Fraud Protection, built for the Convolve Hackathon.

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.11+
- Docker (for Qdrant)

### 2. Start Qdrant

```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

### 3. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Mac Users (Apple Silicon)
# If installation fails, use:
# pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu


# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your API keys
```

### 4. Run Dashboard

```bash
streamlit run dashboard/app.py
```

## 🔑 API Keys Required

| Service | Purpose | Required | Get Key |
|---------|---------|----------|---------|
| **Groq** | Primary LLM | ⭐ Yes | [console.groq.com](https://console.groq.com) |
| **Gemini** | Multimodal backup | ⭐ Yes | [aistudio.google.com](https://aistudio.google.com) |
| Cohere | Embeddings | Optional | [dashboard.cohere.com](https://dashboard.cohere.com) |
| Deepgram | Real-time STT | Optional | [console.deepgram.com](https://console.deepgram.com) |

> 💡 **Note**: Embeddings and Speech-to-Text have local fallbacks (Sentence Transformers and Whisper) that work without API keys.

## 📁 Project Structure

```
sahayak/
├── sahayak/                 # Main package
│   ├── adapters/            # API integrations (LLM, Embeddings, Speech, Vision)
│   ├── agents/              # Multi-agent system
│   ├── config/              # Settings and configuration
│   ├── memory/              # Qdrant Blackboard
│   └── utils/               # Utilities
├── dashboard/               # Streamlit UI
├── data/                    # Demo data
└── scripts/                 # Setup scripts
```

## 🤖 Supported APIs

### LLM Providers
- **Groq** - Llama 3.3 70B (ultra-fast, primary)
- **Gemini** - gemini-1.5-flash (multimodal capable)
- **Grok** - via Puter.js (browser-only, for web dashboard)

### Embeddings
- **Cohere** - embed-multilingual-v3.0 (1024d, production)
- **Local** - all-MiniLM-L6-v2 (384d, no API needed)

### Speech-to-Text
- **Deepgram** - nova-2 (real-time, Hindi/English)
- **Whisper** - Local (offline, 100+ languages)

### Vision/OCR
- **EasyOCR** - Hindi/English text extraction

## 📊 Agent Architecture

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────┐
│   User      │────▶│   Orchestrator  │────▶│  Retrieval  │
│   Input     │     │   (Planner)     │     │   Agent     │
└─────────────┘     └─────────────────┘     └─────────────┘
                           │                      │
                           ▼                      ▼
                    ┌─────────────────────────────────┐
                    │      Qdrant Blackboard          │
                    │  (episodic, knowledge, fraud)   │
                    └─────────────────────────────────┘
                           │                      │
                           ▼                      ▼
                    ┌─────────────┐        ┌─────────────┐
                    │   Fraud     │        │   Critic    │
                    │   Agent     │        │   Agent     │
                    └─────────────┘        └─────────────┘
```

## 🧪 Testing

```bash
# Test API connections
python scripts/test_adapters.py

# Initialize Qdrant collections
python scripts/setup_qdrant.py

# Load demo data
python scripts/load_demo_data.py
```

## 📝 License

MIT
