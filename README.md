# Clinical Agentic System: Autonomous Nursing Support

I have made an Nurse agent  with fastapi and streamlit, it is designed to assist the hospital/clinic staff .It  can handlemanual retrieval, dosage calculations, and patient data logging using a tool-calling architecture.

## Core Architectural Features

### 1. Agentic Routing
This main agent calls tools which are appropriate for the given task.

### 2. Custom RAG Pipeline (Built from Scratch)
I have made the RAG pipeline from scratch without high-level libraries like LangChain or LlamaIndex.
* **Ingestion:** Used `PyMuPDF` to extract text from the uploaded medical manuals.
* **Vectorization & Storage:** Implemented a custom in-memory vector database (`NanoVectorDB`) using `numpy` for chunking, embedding, and cosine similarity calculations.
* **Retrieval:** Fetches the top-k most relevant chunks in the documents.

### 3. Context-Aware Query Rewriting
Instead of relying on the user prompt which might not be complete, i have added a query rewriter that will change the user query based on the history of the conversation. 
e.g., *"Update his heart rate to 90"* -> *"Update Jake's heart rate to 90"*

### 4. Deterministic Execution Sandbox (AI Safety)
LLMs are prone to hallucinations, which is unacceptable in clinical settings.So the LLM is only used as a parameter extractor. the calculations are hard-coded to an O(1). `Pydantic` bounds are used to ensure formatting

## The Agentic Toolset

The main agent has access to the following bounded tools:

1. **`send_question` (Knowledge Retrieval):** Triggers the custom RAG pipeline to search the medical PDFs and answer clinical questions with source attribution.
2. **`dosage_object` (Deterministic Calculator):** Extracts medication name and patient weight, and calculates precise dosages while checking against maximum safe limits.
3. **`all_details` (Vitals Logging):** Extracts structured patient data (name, age, gender, heart rate, BP, weight) and validates bounds (e.g., Systolic BP 70-250) via Pydantic before logging it in a JSONL log.
4. **`emergency_object` (Critical Escalation):** Bypasses standard logging to immediately record critical patient name and the reasons into a dedicated emergency file.

## Tech Stack
* **Backend:** FastAPI, Python 3.x
* **Frontend:** Streamlit
* **LLM Orchestration:** Gemini models
* **Data Validation:** Pydantic
* **Document Processing:** PyMuPDF
* **Vector Math:** Numpy

## Installation & Setup

Follow these steps to run the application locally.

### 1. Clone the Repository
`git clone https://github.com/yourusername/clinical-nursing-agent.git`
`cd clinical-nursing-agent`

### 2. Install Dependencies
Ensure you have Python installed, then install the required packages:
`pip install -r requirements.txt`

### 3. Environment Variables
Create a `.env` file in the root directory and add your Gemini API key:
`GEMINI_API_KEY=your_actual_api_key_here`

### 4. Run the Backend (FastAPI)
Start the FastAPI server. This will host the agentic endpoints and the custom vector database.
`fastapi dev main.py`

### 5. Run the Frontend (Streamlit)
Open a new terminal window, ensure you are in the project directory, and start the UI:
`streamlit run app.py`

The Streamlit interface will open in your browser, allowing you to upload PDF manuals and interact with the clinical agent.