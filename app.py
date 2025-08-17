import os
import time
import random
import tempfile
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq

from htmlTemplates import css, bot_template, user_template
from gtts import gTTS
from deep_translator import GoogleTranslator  # ✅ for translation

# ---------------------- Load environment variables ----------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Please set GROQ_API_KEY in your .env file")
    st.stop()

# ---------------------- TTS helpers ----------------------
LANGS = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
}

def tts_to_mp3(text: str, lang_code: str) -> str:
    """Convert text to mp3 (translated if needed) and return path."""
    if not text.strip():
        return ""

    # Translate if selected language is not English
    if lang_code != "en":
        try:
            text = GoogleTranslator(source="auto", target=lang_code).translate(text)
        except Exception as e:
            st.error(f"Translation error: {e}")
            return ""

    tts = gTTS(text=text, lang=lang_code, slow=False)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp.close()
    tts.save(tmp.name)
    return tmp.name

# ---------------------- PDF Handling ----------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        except Exception as e:
            st.warning(f"Could not read one file: {e}")
    return text.strip()

def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

# ---------------------- Conversation Chain with Retry ----------------------
def get_conversation_chain(vectorstore, retries=3, delay=2):
    for attempt in range(retries):
        try:
            llm = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name="llama3-70b-8192",
                temperature=0.5,
                max_tokens=512,
            )
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            return ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
                memory=memory
            )
        except Exception as e:
            st.warning(f"Groq API error: {e} (attempt {attempt+1}/{retries})")
            if attempt < retries - 1:
                wait_time = delay * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
            else:
                st.error("Groq service unavailable. Please try again later.")
                return None

# ---------------------- User Input ----------------------
def handle_userinput(user_question, speak_lang_code):
    try:
        response = st.session_state.conversation({"question": user_question})
    except Exception as e:
        st.error(f"Error fetching answer: {e}")
        return

    st.session_state.chat_history = response["chat_history"]

    for message in st.session_state.chat_history:
        if message.type in ("human", "user"):
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

    last_ai = next((m.content for m in reversed(st.session_state.chat_history)
                    if m.type not in ("human", "user")), "")
    if last_ai:
        st.session_state.last_ai_message = last_ai

# ---------------------- Main App ----------------------
def main():
    st.set_page_config(page_title="AskSphere", page_icon=":books:", layout="wide")
    st.write(css, unsafe_allow_html=True)

    # Session state defaults
    for key, default in [
        ("conversation", None),
        ("chat_history", None),
        ("raw_text", ""),
        ("last_ai_message", ""),
        ("mp3_path", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    st.header("AskSphere :books:")

    # Sidebar
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True,
            type=["pdf"]
        )
        speak_lang = st.selectbox("🎧 Speech language", list(LANGS.keys()), index=0)
        read_entire = st.button("📖 Read Entire PDF")

        if st.button("🔊 Speak Last Answer"):
            if st.session_state.last_ai_message:
                st.session_state.mp3_path = tts_to_mp3(
                    st.session_state.last_ai_message, LANGS[speak_lang]
                )
            else:
                st.warning("No answer to speak yet.")

    # Process PDFs
    if st.button("Process") and pdf_docs:
        with st.spinner("Processing"):
            raw_text = get_pdf_text(pdf_docs)
            if not raw_text:
                st.error("No extractable text found in the uploaded PDFs.")
                st.stop()
            text_chunks = get_text_chunks(raw_text)
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation = get_conversation_chain(vectorstore)
            st.session_state.raw_text = raw_text
            if st.session_state.conversation:
                st.success("✅ Using Groq LLM.")

    # Chat
    user_question = st.text_input("Ask a question about your documents:")
    if user_question and st.session_state.conversation:
        handle_userinput(user_question, LANGS[speak_lang])

    # Play stored audio
    if st.session_state.mp3_path:
        st.audio(st.session_state.mp3_path, format="audio/mp3")

    # Read entire PDF
    if read_entire:
        if not st.session_state.raw_text:
            st.warning("Please upload and process PDFs first.")
        else:
            st.info(f"Generating audio in {speak_lang}...")
            mp3_path = tts_to_mp3(st.session_state.raw_text, LANGS[speak_lang])
            st.audio(mp3_path, format="audio/mp3")

if __name__ == "__main__":
    main()
