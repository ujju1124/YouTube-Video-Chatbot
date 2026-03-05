# 🎥 YouTube Video Chatbot

> An AI-powered chatbot that lets you have a conversation with **any YouTube video** — paste a URL, and start asking questions. Powered by RAG (Retrieval-Augmented Generation).

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://youtube-video-chatbot-ujju1124.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green?logo=chainlink)
![GPT-4o](https://img.shields.io/badge/GPT--4o-GitHub%20Models-black?logo=github)
![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📸 Demo

| Load Any YouTube Video | Chat With The Video |
|---|---|
| Paste URL → auto-fetches transcript | Ask questions → answers from video only |

---

## ✨ Features

- 🔗 **Any YouTube URL** — supports `watch`, `youtu.be`, `shorts` formats
- 🌍 **Multi-language Support** — auto-detects transcript language (EN manual → EN auto → any language)
- 🖼️ **Video Thumbnail** — auto-displayed after loading
- ✨ **One-click Summarization** — structured summary of the full video
- 💬 **Multi-turn Chat** — remembers full conversation history (memory)
- 📄 **Citations** — shows exactly which part of the video was used to answer
- 🕘 **Video History** — switch between previously loaded videos instantly
- 🔒 **Secure Keys** — API keys stored in `.env`, never hardcoded
- 🆓 **Free Stack** — GPT-4o 

---

## 🏗️ Architecture

```
YouTube URL
    ↓
Extract Video ID (regex)
    ↓
Fetch Transcript (YouTube API) → Multi-language fallback
    ↓
Split into chunks (LangChain RecursiveCharacterTextSplitter)
    ↓
Embed chunks (HuggingFace all-MiniLM-L6-v2)
    ↓
Store in FAISS Vector Store
    ↓
    ├──→ Summarize Button → GPT-4o → Structured Summary
    └──→ User Question → Retrieve top chunks → GPT-4o → Answer + Citations
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit |
| **LLM** | GPT-4o via GitHub Models |
| **Embeddings** | `all-MiniLM-L6-v2` (HuggingFace) |
| **Vector Store** | FAISS (Facebook AI) |
| **Transcript** | `youtube-transcript-api` |
| **Text Splitting** | LangChain `RecursiveCharacterTextSplitter` |
| **Prompt Management** | LangChain `PromptTemplate` |
| **Environment** | `python-dotenv` |

---

## 🚀 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/ujju1124/YouTube-Video-Chatbot.git
cd YouTube-Video-Chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your `.env` file
Create a `.env` file in the root folder:
```
GITHUB_TOKEN=ghp_your_github_token_here
HUGGINGFACEHUB_API_TOKEN=hf_your_huggingface_token_here
```

> - Get your **GitHub Token** → [github.com/settings/tokens](https://github.com/settings/tokens) (needs `read:user` scope)
> - Get your **HuggingFace Token** → [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 4. Run the app
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
YouTube-Video-Chatbot/
│
├── app.py               ← Main application
├── requirements.txt     ← All dependencies
├── .env                 ← API keys (never commit this)
├── .gitignore           ← Excludes .env and cache
└── README.md            ← You are here
```

---

## 💡 How It Works (RAG Explained Simply)

Instead of sending the entire transcript to the LLM (too long, too expensive), this app uses **RAG**:

1. **Retrieval** — transcript is split into small chunks, embedded into vectors, stored in FAISS. When you ask a question, the top 4 most similar chunks are retrieved.
2. **Generation** — only those 4 chunks + your question are sent to GPT-4o. It answers using **only** that context — no hallucination, no outside knowledge.

---

## ⚙️ Supported YouTube URL Formats

```
https://www.youtube.com/watch?v=VIDEO_ID
https://youtu.be/VIDEO_ID
https://www.youtube.com/shorts/VIDEO_ID
```

---

## 🔒 Environment Variables

| Variable | Where to get it |
|---|---|
| `GITHUB_TOKEN` | [github.com/settings/tokens](https://github.com/settings/tokens) |
| `HUGGINGFACEHUB_API_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

---

## 📄 License

MIT License — feel free to use, modify, and share.

---

## 🙋‍♂️ Author

Built by **ujju1124** · Powered by the [GitHub Student Developer Pack](https://education.github.com/pack)
