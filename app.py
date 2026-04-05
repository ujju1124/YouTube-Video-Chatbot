# ============================================================
# INSTALL ALL DEPENDENCIES BEFORE RUNNING
# ============================================================
# --- Core Streamlit & Environment ---
# pip install streamlit python-dotenv
#
# --- YouTube Transcript ---
# pip install youtube-transcript-api
#
# --- LangChain ---
# pip install langchain-community langchain-text-splitters langchain-core langchain-huggingface
#
# --- Vector Store & Tokenizer ---
# pip install faiss-cpu tiktoken
#
# --- GitHub Models (GPT-4o) ---
# pip install openai
#
# --- HTTP Requests (Supadata fallback) ---
# pip install requests
# ============================================================


# ============================================================
# IMPORTS
# ============================================================
import os
import re
import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# LOAD API KEYSS
# ============================================================
load_dotenv()

# Try Streamlit secrets first (cloud), fall back to .env (local)
try:
    github_token = st.secrets["GITHUB_TOKEN"]
    hf_token     = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
except Exception:
    github_token = os.getenv("GITHUB_TOKEN")
    hf_token     = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# ============================================================
# STEP 1 — EXTRACT VIDEO ID FROM YOUTUBE URL
# ============================================================
def extract_video_id(url):
    """
    INPUT  : A full YouTube URL string
             Example: "https://www.youtube.com/watch?v=3MG4mtnJvAg"

    PROCESS: Uses regex to find the 11-character video ID
             from any common YouTube URL format

    OUTPUT : The 11-character video ID string, or None if not found
             Example: "3MG4mtnJvAg"

    SUPPORTED URL FORMATS:
      - https://www.youtube.com/watch?v=VIDEO_ID
      - https://youtu.be/VIDEO_ID
      - https://www.youtube.com/shorts/VIDEO_ID
    """
    pattern = (
        r'(?:https?://)?'
        r'(?:www\.)?'
        r'(?:youtu\.be/|youtube\.com/(?:watch\?v=|embed/|shorts/))'
        r'([\w-]{11})'
    )
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return None


# ============================================================
# STEP 2 — FETCH TRANSCRIPT FROM YOUTUBE (Production Ready)
# ============================================================
def fetch_transcript(video_id):
    """
    INPUT  : YouTube video ID string
             Example: "3MG4mtnJvAg"

    PROCESS: Tries transcript sources in this priority order:
             1. youtube-transcript-api — free, works locally
             2. Supadata API           — works on cloud, handles IP bans
             Each attempt silently falls back to the next if it fails

    OUTPUT : Tuple of (transcript, language_used, error)
             Success → ("transcript text...", "en", None)
             Failure → (None, None, "friendly error message")

             Example success via youtube-transcript-api:
             ("Today we discuss AI...", "en", None)

             Example success via Supadata fallback:
             ("Today we discuss AI...", "en (via Supadata)", None)

             Example failure (both failed):
             (None, None, "Could not fetch transcript...")
    """

    # ── Try 1: youtube-transcript-api (free, works locally) ──────────────
    try:
        api = YouTubeTranscriptApi()

        try:
            transcript_list = api.fetch(video_id, languages=['en'])
            language_used   = 'en'
        except Exception:
            try:
                transcript_list = api.fetch(video_id, languages=['en-US', 'en-GB', 'en-CA'])
                language_used   = 'en (auto-generated)'
            except Exception:
                available       = api.list(video_id)
                first_lang      = available[0].language_code
                transcript_list = api.fetch(video_id, languages=[first_lang])
                language_used   = first_lang

        transcript = " ".join(chunk.text for chunk in transcript_list)
        return transcript, language_used, None

    except Exception:
        pass  # silently move to Supadata

    # ── Try 2: Supadata API (dedicated transcript service) ────────────────
    # Supadata handles IP rotation internally — works on Streamlit Cloud
    try:
        try:
            supadata_key = st.secrets["SUPADATA_API_KEY"]
        except Exception:
            supadata_key = os.getenv("SUPADATA_API_KEY")

        if not supadata_key:
            raise Exception("No Supadata API key found")

        # NOTE: We do NOT pass text=true here
        # Without text=true → content is a LIST of {text, offset, duration} objects
        # With text=true    → content is a plain STRING (our old bug)
        # List format is safer — we can always join it ourselves
        response = requests.get(
            "https://api.supadata.ai/v1/youtube/transcript",
            headers={"x-api-key": supadata_key},
            params={"videoId": video_id, "lang": "en"},
            timeout=30
        )

        if response.status_code == 200:
            data     = response.json()
            content  = data.get("content", [])
            language = data.get("lang", "en")

            # content is a list of {"text": "...", "offset": 123, "duration": 456}
            # We only need the "text" field from each chunk
            if isinstance(content, list):
                transcript = " ".join(
                    chunk["text"] for chunk in content if "text" in chunk
                )
            elif isinstance(content, str):
                # Safety fallback — handles text=true response just in case
                transcript = content
            else:
                transcript = ""

            if transcript.strip():
                return transcript, f"{language} (via Supadata)", None

    except Exception:
        pass

    # ── Both options failed ───────────────────────────────────────────────
    return None, None, (
        "⚠️ Could not fetch transcript automatically. "
        "This video may not have captions, or the server is being blocked by YouTube. "
        "Try a different video."
    )


# ============================================================
# STEP 3 — BUILD VECTOR STORE FROM TRANSCRIPT
# ============================================================
def build_vector_store(transcript):
    """
    INPUT  : A full transcript string
             Example: "In this video we discuss football tactics..."

    PROCESS: 1. Splits transcript into 700-character chunks (80 char overlap)
             2. Converts each chunk into a number vector using HuggingFace embeddings
             3. Stores all vectors in a FAISS vector store for fast similarity search

    OUTPUT : A FAISS vector store object ready for similarity search
             Example: <FAISS vector_store with 142 chunks indexed>

    NOTE   : HuggingFace embeddings run locally — no API call needed here
    """
    splitter     = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=80)
    chunks       = splitter.create_documents([transcript])
    embeddings   = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


# ============================================================
# STEP 4 — SET UP THE LLM (GitHub Models — GPT-4o)
# ============================================================
def load_llm():
    """
    INPUT  : Nothing (reads GITHUB_TOKEN from secrets / .env automatically)

    PROCESS: Creates an OpenAI client pointed at GitHub Models endpoint
             instead of the default OpenAI endpoint

    OUTPUT : An OpenAI client object ready to make GPT-4o API calls
             Example: <OpenAI client → base_url=models.inference.ai.azure.com>
    """
    client = OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=github_token
    )
    return client


# ============================================================
# STEP 5 — SUMMARIZE THE FULL VIDEO
# ============================================================
def summarize_video(transcript, client):
    """
    INPUT  : 1. transcript — the full transcript string of the video
                Example: "Today we talk about AI and robots..."
             2. client — the OpenAI client from load_llm()

    PROCESS: 1. Takes the first 12000 characters of the transcript
             2. Sends it to GPT-4o with a summarization prompt
             3. GPT-4o returns a structured summary with key topics

    OUTPUT : A summary string with key topics and main points
             Example:
             "This video covers:
              - AI and its impact on jobs
              - Elon Musk's views on AGI..."

    NOTE   : Uses raw transcript directly (not vector store)
             because summarization needs the full video content,
             not just a few retrieved chunks
    """
    transcript_preview = transcript[:12000]

    summary_prompt = f"""
You are a helpful assistant. Read the following YouTube video transcript and provide a clear, structured summary.

Your summary must include:
- A brief 2-3 sentence overview of what the video is about
- A bullet point list of the key topics discussed
- Any important conclusions or takeaways mentioned

Keep the summary easy to read and well organised.

Transcript:
{transcript_preview}

Summary:
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": summary_prompt}],
        max_tokens=1024,
        temperature=0.5
    )

    return response.choices[0].message.content


# ============================================================
# STEP 6 — GENERATE SUGGESTED QUESTIONS
# ============================================================
def generate_suggested_questions(transcript, client):
    """
    INPUT  : 1. transcript — full transcript string
                Example: "Today we discuss AI and the future..."
             2. client — OpenAI client from load_llm()

    PROCESS: Sends first 3000 characters of transcript to GPT-4o
             Asks it to generate exactly 3 short, interesting questions
             that a user would naturally want to ask about this video

    OUTPUT : A list of 3 question strings
             Example: [
               "What is the main topic discussed?",
               "Who are the key people mentioned?",
               "What conclusions were reached?"
             ]

    NOTE   : Questions shown as clickable buttons in the UI
             so users immediately know what kind of questions to ask.
             Saved in video_history so switching back to a video
             restores them without re-calling the API.
    """
    prompt = f"""
Based on this transcript excerpt, generate exactly 3 short, interesting questions
a user might want to ask about this video.
Return ONLY the 3 questions, one per line, no numbering, no bullet points, no extra text.

Transcript:
{transcript[:3000]}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=120,
        temperature=0.7
    )

    raw       = response.choices[0].message.content.strip()
    questions = [q.strip() for q in raw.split("\n") if q.strip()]
    return questions[:3]


# ============================================================
# STEP 7 — ANSWER A QUESTION (with memory + citations)
# ============================================================
def answer_question(question, vector_store, client, chat_history):
    """
    INPUT  : 1. question     — the user's question string
                Example: "What did Elon say about AI?"
             2. vector_store — the FAISS vector store from build_vector_store()
             3. client       — the OpenAI client from load_llm()
             4. chat_history — list of past {question, answer} dicts for memory
                Example: [{"question": "What is AI?", "answer": "AI is..."}]

    PROCESS: 1. Detects if broad summary question → uses k=10 chunks
                Or specific question → uses k=4 chunks
             2. Retrieves most relevant chunks from the vector store
             3. Builds chat history text so LLM remembers past turns
             4. Fills in the prompt template with history + context + question
             5. Sends final prompt to GPT-4o and returns the answer

    OUTPUT : A tuple of (answer, retrieved_docs)
             answer         — string response from GPT-4o
             retrieved_docs — list of LangChain Document objects (the sources)

             Example:
             ("Elon discussed AGI timelines...", [Document(...), Document(...)])
    """

    # Broad questions need more chunks for better coverage
    summary_keywords = [
        "key topics", "summarize", "summary", "main points",
        "what was discussed", "highlight", "overview", "what is this video about"
    ]
    is_summary_question = any(keyword in question.lower() for keyword in summary_keywords)
    k_value = 10 if is_summary_question else 4

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': k_value}
    )

    retrieved_docs = retriever.invoke(question)
    context_text   = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Build memory string from past conversation turns
    history_text = ""
    for turn in chat_history:
        history_text += f"User: {turn['question']}\nAssistant: {turn['answer']}\n\n"

    prompt = PromptTemplate(
        template="""
You are a helpful assistant that answers questions strictly based on the content of a YouTube video transcript.

Rules you must follow:
- Answer ONLY using the transcript context provided below.
- Do NOT use any outside knowledge or make things up.
- If the answer is not found in the context, just say: "I could not find that information in the video."
- Keep your answers clear, concise, and easy to understand.

--- Previous Conversation (Memory) ---
{history}

--- Transcript Context (from the video) ---
{context}

--- Current Question ---
{question}

Answer:
""",
        input_variables=['history', 'context', 'question']
    )

    final_prompt = prompt.invoke({
        'history': history_text,
        'context': context_text,
        'question': question
    })

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": final_prompt.text}],
        max_tokens=512,
        temperature=0.5
    )

    return response.choices[0].message.content, retrieved_docs


# ============================================================
# STEP 8 — SWITCH TO A VIDEO FROM HISTORY
# ============================================================
def switch_to_video(video_id):
    """
    INPUT  : A video ID string that exists in st.session_state.video_history
             Example: "3MG4mtnJvAg"

    PROCESS: Pulls all saved data for that video from video_history
             and loads it back into the active session state variables.
             No API calls made — everything is loaded from memory.

    OUTPUT : Nothing returned — directly updates st.session_state
             and calls st.rerun() to refresh the UI.

             Effect: The app switches to the selected video instantly
             with its transcript, summary and suggested questions restored.
    """
    saved = st.session_state.video_history[video_id]

    st.session_state.vector_store        = saved["vector_store"]
    st.session_state.transcript          = saved["transcript"]
    st.session_state.summary             = saved["summary"]
    st.session_state.suggested_questions = saved["suggested_questions"]
    st.session_state.video_id            = video_id
    st.session_state.chat_history        = []
    st.session_state.video_loaded        = True

    st.rerun()


# ============================================================
# STREAMLIT UI
# ============================================================
def main():
    st.set_page_config(page_title="YouTube Video Chatbot", page_icon="🎥", layout="wide")

    st.title("🎥 YouTube Video Chatbot")
    st.caption("Paste any YouTube video URL and chat with its content. Powered by LangChain + FAISS + GPT-4o")

    # ---- Check tokens ----
    if not github_token:
        st.error("❌ GITHUB_TOKEN not found. Please add it to your .env file or Streamlit secrets.")
        st.stop()

    if not hf_token:
        st.error("❌ HUGGINGFACEHUB_API_TOKEN not found. Please add it to your .env file or Streamlit secrets.")
        st.stop()

    # ----------------------------------------------------------------
    # SIDEBAR
    # ----------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Setup")

        youtube_url = st.text_input(
            "YouTube Video URL",
            placeholder="https://www.youtube.com/watch?v=..."
        )

        load_button = st.button("🚀 Load Video", use_container_width=True)

        st.divider()
        st.markdown("**Supported URL formats:**")
        st.code("youtube.com/watch?v=VIDEO_ID\nyoutu.be/VIDEO_ID\nyoutube.com/shorts/VIDEO_ID")

        st.divider()
        st.markdown("**🤖 Model:** `GPT-4o` via GitHub Models")
        st.markdown("**🧠 Embeddings:** `all-MiniLM-L6-v2`")

        # ---- Video History ----
        # Shows all previously loaded videos as clickable buttons
        # Clicking one switches instantly — no re-fetching needed
        if "video_history" in st.session_state and st.session_state.video_history:
            st.divider()
            st.markdown("**🕘 Previously Loaded Videos:**")

            for past_id in st.session_state.video_history:
                thumb_url  = f"https://img.youtube.com/vi/{past_id}/default.jpg"
                col1, col2 = st.sidebar.columns([1, 2])
                with col1:
                    st.image(thumb_url)
                with col2:
                    is_active = (past_id == st.session_state.get("video_id", ""))
                    label     = f"✅ {past_id}" if is_active else f"▶️ {past_id}"
                    if st.button(label, key=f"history_{past_id}"):
                        switch_to_video(past_id)

        # ---- About Section ----
        st.divider()
        with st.expander("ℹ️ About this app"):
            st.markdown("""
            **YouTube Video Chatbot** uses **RAG**
            (Retrieval-Augmented Generation) to answer
            questions strictly from video content —
            no hallucination, no outside knowledge.

            **How it works:**
            1. Transcript is fetched from YouTube
            2. Split into chunks & embedded into vectors
            3. Your question retrieves the most relevant chunks
            4. GPT-4o answers using only those chunks

            **Built with:**
            - 🔗 LangChain
            - 🧠 FAISS Vector Store
            - 🤖 GPT-4o (GitHub Models)
            - 🎥 YouTube Transcript API + Supadata
            - ☁️ Streamlit Community Cloud

            **Built by:** [ujju1124](https://github.com/ujju1124)

            ⭐ [Star on GitHub](https://github.com/ujju1124/YouTube-Video-Chatbot)
            """)

    # ----------------------------------------------------------------
    # SESSION STATE INITIALIZATION
    # Streamlit forgets everything on every rerun — session state
    # keeps all important data alive across reruns
    # ----------------------------------------------------------------
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "client" not in st.session_state:
        st.session_state.client = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "video_loaded" not in st.session_state:
        st.session_state.video_loaded = False
    if "video_id" not in st.session_state:
        st.session_state.video_id = ""
    if "transcript" not in st.session_state:
        st.session_state.transcript = ""
    if "summary" not in st.session_state:
        st.session_state.summary = ""
    if "language_used" not in st.session_state:
        st.session_state.language_used = ""
    if "video_history" not in st.session_state:
        # Format: { video_id: { vector_store, transcript, summary, suggested_questions } }
        st.session_state.video_history = {}
    if "suggested_questions" not in st.session_state:
        st.session_state.suggested_questions = []
    if "auto_question" not in st.session_state:
        # Stores a question clicked from suggested buttons
        # Processed as a real chat message in the next rerun
        st.session_state.auto_question = None

    # ----------------------------------------------------------------
    # LOAD VIDEO BUTTON LOGIC
    # ----------------------------------------------------------------
    if load_button:
        if not youtube_url:
            st.sidebar.error("Please enter a YouTube URL.")
        else:
            video_id = extract_video_id(youtube_url)

            if not video_id:
                st.sidebar.error("Could not find a valid video ID. Please check the URL.")
            else:
                with st.spinner("📥 Fetching transcript and building knowledge base..."):

                    transcript, language_used, error = fetch_transcript(video_id)

                    if error:
                        st.sidebar.error(error)
                    else:
                        vector_store = build_vector_store(transcript)
                        client       = load_llm()

                        # Save everything to active session state
                        st.session_state.vector_store        = vector_store
                        st.session_state.client              = client
                        st.session_state.transcript          = transcript
                        st.session_state.language_used       = language_used
                        st.session_state.chat_history        = []
                        st.session_state.summary             = ""
                        st.session_state.suggested_questions = []   # reset for new video
                        st.session_state.auto_question       = None
                        st.session_state.video_loaded        = True
                        st.session_state.video_id            = video_id

                        # Save to video history for future switching
                        st.session_state.video_history[video_id] = {
                            "vector_store":        vector_store,
                            "transcript":          transcript,
                            "summary":             "",
                            "suggested_questions": []
                        }

                        st.sidebar.success(f"✅ Video loaded! (ID: `{video_id}`)")

    # ----------------------------------------------------------------
    # MAIN AREA — only shown after a video is loaded
    # ----------------------------------------------------------------
    if st.session_state.video_loaded:

        # ── Thumbnail + Info Header ───────────────────────────────────────
        # YouTube provides a free thumbnail for every video at this URL pattern
        thumbnail_url = f"https://img.youtube.com/vi/{st.session_state.video_id}/0.jpg"

        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(thumbnail_url, use_container_width=True)
        with col2:
            st.success(f"📺 **Video ID:** `{st.session_state.video_id}`")
            st.info(f"🌍 **Transcript language:** `{st.session_state.language_used}`")
            st.markdown("Ask anything about this video below 👇")

        st.divider()

        # ── Summarization Button ──────────────────────────────────────────
        # Appears right after video loads — above the chat section
        # Summary is cached so it only generates once per video
        st.subheader("📋 Video Summary")

        if st.session_state.summary:
            # Already generated — just display it
            st.markdown(st.session_state.summary)
        else:
            if st.button("✨ Summarize This Video", use_container_width=True):
                with st.spinner("📝 Summarizing the video..."):
                    summary = summarize_video(
                        transcript=st.session_state.transcript,
                        client=st.session_state.client
                    )
                    # Cache in both session state and video history
                    st.session_state.summary = summary
                    st.session_state.video_history[st.session_state.video_id]["summary"] = summary
                    st.rerun()

        st.divider()

        # ── Suggested Questions ───────────────────────────────────────────
        # Auto-generates 3 starter questions after video loads
        # Clicking any button auto-asks it as a real chat message

        # Generate only if not yet done for this video
        if not st.session_state.suggested_questions:
            with st.spinner("💡 Generating suggested questions..."):
                questions = generate_suggested_questions(
                    transcript=st.session_state.transcript,
                    client=st.session_state.client
                )
                st.session_state.suggested_questions = questions
                # Cache in history too so switching back restores them
                st.session_state.video_history[st.session_state.video_id]["suggested_questions"] = questions
                st.rerun()

        if st.session_state.suggested_questions:
            st.markdown("**💡 Suggested questions — click to ask:**")
            cols = st.columns(3)
            for i, sq in enumerate(st.session_state.suggested_questions):
                with cols[i]:
                    if st.button(sq, key=f"sq_{i}", use_container_width=True):
                        # Store clicked question — picked up below as active_question
                        st.session_state.auto_question = sq
                        st.rerun()

        st.divider()

        # ── Chat History Display ──────────────────────────────────────────
        if st.session_state.chat_history:
            st.subheader("💬 Conversation")
            for turn in st.session_state.chat_history:

                with st.chat_message("user"):
                    st.write(turn["question"])

                with st.chat_message("assistant"):
                    st.write(turn["answer"])

                    with st.expander("📄 View Sources (Citations)"):
                        for i, source in enumerate(turn["sources"]):
                            st.markdown(f"**Source {i+1}:**")
                            st.info(source)

        # ── Question Input ────────────────────────────────────────────────
        st.divider()
        question = st.chat_input("Ask anything about the video...")

        # Handle either typed question OR suggested question button click
        active_question = question or st.session_state.auto_question

        if active_question:
            # Clear auto_question so it doesn't re-trigger on next rerun
            st.session_state.auto_question = None

            with st.spinner("🤔 Thinking..."):
                answer, retrieved_docs = answer_question(
                    question=active_question,
                    vector_store=st.session_state.vector_store,
                    client=st.session_state.client,
                    chat_history=st.session_state.chat_history
                )

            st.session_state.chat_history.append({
                "question": active_question,
                "answer":   answer,
                "sources":  [doc.page_content for doc in retrieved_docs]
            })

            st.rerun()

        # ── Clear Conversation Button ─────────────────────────────────────
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Conversation"):
                st.session_state.chat_history = []
                st.rerun()

    else:
        # ── Landing Screen ────────────────────────────────────────────────
        st.info("👈 Paste a YouTube URL in the sidebar and click **Load Video** to start.")

        st.markdown("""
        ### 💡 How it works:
        1. 🔗 Paste **any** YouTube video URL in the sidebar
        2. 🚀 Click **Load Video** — transcript is fetched automatically (any language!)
        3. 🖼️ See the video thumbnail and transcript language detected
        4. 💡 Click any **suggested question** to instantly ask it
        5. ✨ Click **Summarize This Video** to get a full structured summary
        6. 💬 Ask your own question — the AI answers **strictly from the video**
        7. 📄 Expand **View Sources** to see exactly which part of the video was used
        8. 🕘 Switch between previously loaded videos instantly from the sidebar
        """)


# ============================================================
# RUN THE APP
# ============================================================
if __name__ == "__main__":
    main()
