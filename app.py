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
# ============================================================


# ============================================================
# IMPORTS
# ============================================================
import os
import re
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# LOAD API KEYS FROM .env FILE
# ============================================================
load_dotenv()
github_token = os.getenv("GITHUB_TOKEN")
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")  # still needed for embeddings only


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
# STEP 2 — FETCH TRANSCRIPT FROM YOUTUBE (Multi-language)
# ============================================================
def fetch_transcript(video_id):
    """
    INPUT  : A YouTube video ID string
             Example: "3MG4mtnJvAg"

    PROCESS: Tries to fetch captions in this order:
             1. English manual captions ('en')
             2. English auto-generated variants ('en-US', 'en-GB', 'en-CA')
             3. Any available language on the video (fallback)
             Each attempt is wrapped in a try/except so it moves
             to the next option if the current one fails

    OUTPUT : A tuple of (transcript, language_used, error)
             Success → ("full transcript text...", "en", None)
             Failure → (None, None, "error message string")

             Example success:
             ("Today we talk about AI...", "en", None)

             Example fallback success:
             ("Hoy hablamos de IA...", "es", None)

             Example failure:
             (None, None, "No captions available for this video.")
    """
    try:
        api = YouTubeTranscriptApi()

        # Try 1 — English manual captions
        try:
            transcript_list = api.fetch(video_id, languages=['en'])
            language_used = 'en'

        except Exception:

            # Try 2 — Auto-generated English variants
            try:
                transcript_list = api.fetch(video_id, languages=['en-US', 'en-GB', 'en-CA'])
                language_used = 'en (auto-generated)'

            except Exception:

                # Try 3 — Whatever language is available on the video
                available = api.list(video_id)
                first_lang = available[0].language_code
                transcript_list = api.fetch(video_id, languages=[first_lang])
                language_used = first_lang

        transcript = " ".join(chunk.text for chunk in transcript_list)
        return transcript, language_used, None

    except TranscriptsDisabled:
        return None, None, "No captions available for this video."
    except Exception as e:
        return None, None, f"Error fetching transcript: {str(e)}"


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
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=80)
    chunks = splitter.create_documents([transcript])

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store


# ============================================================
# STEP 4 — SET UP THE LLM (GitHub Models — GPT-4o)
# ============================================================
def load_llm():
    """
    INPUT  : Nothing (reads GITHUB_TOKEN from .env automatically)

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
                (GPT-4o has a token limit so we cap it to be safe)
             2. Sends it to GPT-4o with a summarization prompt
             3. GPT-4o returns a structured summary with key topics

    OUTPUT : A summary string with key topics and main points
             Example:
             "This video covers:
              - AI and its impact on jobs
              - Elon Musk's views on AGI
              ..."

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
        messages=[
            {"role": "user", "content": summary_prompt}
        ],
        max_tokens=1024,
        temperature=0.5
    )

    summary = response.choices[0].message.content
    return summary


# ============================================================
# STEP 6 — ANSWER A QUESTION (with memory + citations)
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
             answer        — string response from GPT-4o
             retrieved_docs — list of LangChain Document objects (the sources)

             Example:
             ("Elon discussed AGI timelines...", [Document(...), Document(...)])
    """

    # Detect broad summary-type questions → need more chunks
    summary_keywords = [
        "key topics", "summarize", "summary", "main points",
        "what was discussed", "highlight", "overview", "what is this video about"
    ]
    is_summary_question = any(keyword in question.lower() for keyword in summary_keywords)
    k_value = 10 if is_summary_question else 4

    # --- Retriever ---
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': k_value}
    )

    # --- Retrieve relevant chunks ---
    retrieved_docs = retriever.invoke(question)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # --- Build chat history string for memory ---
    history_text = ""
    for turn in chat_history:
        history_text += f"User: {turn['question']}\nAssistant: {turn['answer']}\n\n"

    # --- Prompt Template ---
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

    # --- Build the final prompt ---
    final_prompt = prompt.invoke({
        'history': history_text,
        'context': context_text,
        'question': question
    })

    # --- Get answer from GPT-4o via GitHub Models ---
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": final_prompt.text}
        ],
        max_tokens=512,
        temperature=0.5
    )
    answer = response.choices[0].message.content

    return answer, retrieved_docs


# ============================================================
# STEP 7 — SWITCH TO A VIDEO FROM HISTORY
# ============================================================
def switch_to_video(video_id):
    """
    INPUT  : A video ID string that exists in st.session_state.video_history
             Example: "3MG4mtnJvAg"

    PROCESS: Pulls all saved data for that video from video_history
             and loads it back into the active session state variables
             No API calls made — everything is loaded from memory

    OUTPUT : Nothing returned — directly updates st.session_state
             and calls st.rerun() to refresh the UI

             Effect: The app switches to the selected video instantly
    """
    saved = st.session_state.video_history[video_id]

    st.session_state.vector_store = saved["vector_store"]
    st.session_state.transcript   = saved["transcript"]
    st.session_state.summary      = saved["summary"]
    st.session_state.video_id     = video_id
    st.session_state.chat_history = []       # fresh chat for switched video
    st.session_state.video_loaded = True

    st.rerun()


# ============================================================
# STREAMLIT UI
# ============================================================
def main():
    st.set_page_config(page_title="YouTube RAG Chatbot", page_icon="🎥", layout="wide")

    st.title("🎥 YouTube Video Chatbot")
    st.caption("Paste any YouTube video URL and chat with its content. Powered by LangChain + FAISS + GPT-4o")

    # ---- Check tokens loaded from .env ----
    if not github_token:
        st.error("❌ GITHUB_TOKEN not found. Please add it to your .env file.")
        st.stop()

    if not hf_token:
        st.error("❌ HUGGINGFACEHUB_API_TOKEN not found. Please add it to your .env file.")
        st.stop()

    # ---- Sidebar ----
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
        st.markdown("**🤖 Model:** `GPT-4o`")
        st.markdown("**🧠 Embeddings:** `all-MiniLM-L6-v2`")

        # ---- Video History Section in Sidebar ----
        # Shows all previously loaded videos as clickable buttons
        # Clicking one switches to that video instantly — no re-fetching
        if "video_history" in st.session_state and st.session_state.video_history:
            st.divider()
            st.markdown("**🕘 Previously Loaded Videos:**")

            for past_id in st.session_state.video_history:

                # Show thumbnail + button side by side for each past video
                thumb_url = f"https://img.youtube.com/vi/{past_id}/default.jpg"

                col1, col2 = st.sidebar.columns([1, 2])
                with col1:
                    st.image(thumb_url)
                with col2:
                    # Highlight the currently active video
                    is_active = (past_id == st.session_state.get("video_id", ""))
                    label = f"▶️ {past_id}" if not is_active else f"✅ {past_id}"
                    if st.button(label, key=f"history_{past_id}"):
                        switch_to_video(past_id)

    # ---- Session State initialization ----
    # All important data is stored here so Streamlit remembers it across reruns
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
        # Stores all previously loaded videos
        # Format: { video_id: { vector_store, transcript, summary } }
        st.session_state.video_history = {}

    # ---- Load Video Button Logic ----
    if load_button:
        if not youtube_url:
            st.sidebar.error("Please enter a YouTube URL.")
        else:
            video_id = extract_video_id(youtube_url)

            if not video_id:
                st.sidebar.error("Could not find a valid video ID in that URL. Please check the URL.")
            else:
                with st.spinner("📥 Fetching transcript and building knowledge base..."):

                    # fetch_transcript now returns 3 values including language used
                    transcript, language_used, error = fetch_transcript(video_id)

                    if error:
                        st.sidebar.error(error)
                    else:
                        vector_store = build_vector_store(transcript)
                        client = load_llm()

                        # Save to active session state
                        st.session_state.vector_store  = vector_store
                        st.session_state.client        = client
                        st.session_state.transcript    = transcript
                        st.session_state.language_used = language_used
                        st.session_state.chat_history  = []
                        st.session_state.summary       = ""
                        st.session_state.video_loaded  = True
                        st.session_state.video_id      = video_id

                        # Save to video history for future switching
                        # summary is empty string for now — gets filled when user clicks summarize
                        st.session_state.video_history[video_id] = {
                            "vector_store": vector_store,
                            "transcript":   transcript,
                            "summary":      ""
                        }

                        st.sidebar.success(f"✅ Video loaded! (ID: `{video_id}`)")

    # ---- Main Chat Area ----
    if st.session_state.video_loaded:

        # --------------------------------------------------------
        # VIDEO THUMBNAIL + INFO HEADER
        # YouTube provides a free thumbnail image for every video
        # URL format: https://img.youtube.com/vi/VIDEO_ID/0.jpg
        # --------------------------------------------------------
        thumbnail_url = f"https://img.youtube.com/vi/{st.session_state.video_id}/0.jpg"

        col1, col2 = st.columns([1, 3])
        with col1:
            # Display the video thumbnail
            st.image(thumbnail_url, use_container_width=True)
        with col2:
            st.success(f"📺 **Video ID:** `{st.session_state.video_id}`")
            st.info(f"🌍 **Transcript language:** `{st.session_state.language_used}`")
            st.markdown("Ask anything about this video below 👇")

        st.divider()

        # --------------------------------------------------------
        # SUMMARIZATION BUTTON
        # Appears right after video loads — above the chat section
        # Summary is cached in session state so it only generates once
        # --------------------------------------------------------
        st.subheader("📋 Video Summary")

        if st.session_state.summary:
            # Already generated — just show it
            st.markdown(st.session_state.summary)
        else:
            if st.button("✨ Summarize This Video", use_container_width=True):
                with st.spinner("📝 Summarizing the video..."):
                    summary = summarize_video(
                        transcript=st.session_state.transcript,
                        client=st.session_state.client
                    )
                    # Save summary to session state AND to video history
                    st.session_state.summary = summary
                    st.session_state.video_history[st.session_state.video_id]["summary"] = summary
                    st.rerun()

        st.divider()

        # --------------------------------------------------------
        # CHAT SECTION
        # Shows full conversation with citations
        # --------------------------------------------------------
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

        # Question input
        st.divider()
        question = st.chat_input("Ask anything about the video...")

        if question:
            with st.spinner("🤔 Thinking..."):
                answer, retrieved_docs = answer_question(
                    question=question,
                    vector_store=st.session_state.vector_store,
                    client=st.session_state.client,
                    chat_history=st.session_state.chat_history
                )

            sources = [doc.page_content for doc in retrieved_docs]

            st.session_state.chat_history.append({
                "question": question,
                "answer":   answer,
                "sources":  sources
            })

            st.rerun()

        # Clear conversation button
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Conversation"):
                st.session_state.chat_history = []
                st.rerun()

    else:
        # Landing screen
        st.info("👈 Paste a YouTube URL in the sidebar and click **Load Video** to start.")

        st.markdown("""
        ### 💡 How it works:
        1. 🔗 Paste **any** YouTube video URL in the sidebar
        2. 🚀 Click **Load Video** — transcript is fetched automatically (any language!)
        3. 🖼️ See the video thumbnail and transcript language detected
        4. ✨ Click **Summarize This Video** to get a full structured summary
        5. 💬 Ask any question — the AI answers **strictly from the video**
        6. 📄 Expand **View Sources** to see exactly which part of the video was used
        7. 🕘 Switch between previously loaded videos instantly from the sidebar
        """)


# ============================================================
# RUN THE APP
# ============================================================
if __name__ == "__main__":
    main()