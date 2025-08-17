📘 AskSphere – Chat with Your PDFs (with Voice & Translation)
AskSphere is an interactive Streamlit application that allows you to chat with your PDF documents using Groq’s Llama 3 LLM.
It supports multilingual text-to-speech (TTS) so the answers can be read aloud in English, Hindi, Spanish, French, and German.

Features

1-📂 Upload and process multiple PDF documents

2-🤖 Ask questions and get context-aware answers powered by Groq LLM

3-🧠 Uses LangChain + FAISS for efficient text chunking & retrieval

4-🔄 Built-in retry logic for Groq API requests

5-🎧 Text-to-Speech (TTS) with translation in multiple languages

6-🔊 Option to listen to answers or entire PDFs as audio

7-💬 Chat history preserved during the session

⚡ Tech Stack:-

1-Frontend / App: Streamlit
2-LLM: Groq (Llama 3) via langchain_groq
3-Embeddings: HuggingFace MiniLM
4-Vector Store: FAISS
5-Text-to-Speech: gTTS
6-Translation: Deep Translator
7-PDF Processing: PyPDF2
