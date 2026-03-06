🤖 Private-PDF Brain: Context-Aware RAG System
A high-performance Retrieval-Augmented Generation (RAG) system built with FastAPI and Gemini. This project allows users to "chat" with their local PDF documents while maintaining conversational memory and high retrieval accuracy.
--
🚀 Key Engineering Features
Custom Vector DB: Implemented a persistent similarity search engine using NumPy for vectorized operations and SentenceTransformers for semantic embeddings.

Conversational Memory: Utilized a deque-based sliding window to manage chat history, ensuring the model maintains context without exceeding token limits.

Contextual Query Rewriting: Solves the "Pronoun Problem" by using an LLM to rewrite follow-up questions (e.g., "What is his salary?") into standalone search queries (e.g., "What is Tim Cook's salary?") before querying the database.

Streamlit Interface: A clean, reactive frontend for document indexing and real-time chat.

In-Memory PDF Processing: Efficiently handles PDF streams using PyMuPDF, avoiding unnecessary disk I/O during ingestion.
--
🛠️ Tech Stack
Backend: FastAPI, Uvicorn

Frontend: Streamlit

AI Models: Gemini 2.5 Flash (LLM), all-MiniLM-L6-v2 (Embeddings)

Parsing: PyMuPDF (fitz)
--
🧠 System Architecture
The system follows a three-stage pipeline:

Ingestion: PDF text is extracted, cleaned, and split into overlapping chunks to preserve semantic context.

Retrieval: User queries are vectorized and compared against the chunk library using Cosine Similarity.

Synthesis: The LLM receives a prompt containing the Retrieved Context + Conversation History + Current Question to generate a grounded response.
--
🔧 Installation & Setup
Clone the repository:

Set up Environment Variables:
Create a .env file in the root directory:

Install Dependencies:

Run the Application:
Terminal 1 (Backend): fastapi dev main.py

Terminal 2 (Frontend): streamlit run app.py