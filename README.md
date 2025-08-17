📘 AskSphere – Chat with Your PDFs (with Voice & Translation)
AskSphere is an interactive Streamlit application that allows you to chat with your PDF documents using Groq’s Llama 3 LLM.
It supports multilingual text-to-speech (TTS) so the answers can be read aloud in English, Hindi, Spanish, French, and German.

🚀 Features
📂 Upload and process multiple PDF documents

🤖 Ask questions and get context-aware answers powered by Groq LLM

🧠 Uses LangChain + FAISS for efficient text chunking & retrieval

🔄 Built-in retry logic for Groq API requests

🎧 Text-to-Speech (TTS) with translation in multiple languages

🔊 Option to listen to answers or entire PDFs as audio

💬 Chat history preserved during the session

🛠️ Tech Stack
Frontend / App: Streamlit

LLM: Groq (Llama 3) via langchain_groq

Embeddings: HuggingFace MiniLM

Vector Store: FAISS

Text-to-Speech: gTTS

Translation: Deep Translator

PDF Processing: PyPDF2
