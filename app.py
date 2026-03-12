import streamlit as st
import requests
import uuid

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Clinical AI Agent", page_icon="🏥", layout="wide")
st.title("🏥 Clinical Nursing Assistant")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Knowledge Base")
    st.caption(f"Session ID: `{st.session_state.session_id[:8]}`") 
    
    uploaded_file = st.file_uploader("Upload Medical Manual (PDF)", type="pdf")
    
    if uploaded_file and st.button("Index Document"):
        with st.spinner("Processing and vectorizing PDF..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = requests.post(f"{API_BASE_URL}/upload/", files=files)
                response.raise_for_status()
                st.success(f"'{uploaded_file.name}' indexed successfully!")
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to backend. Is FastAPI running?")
            except requests.exceptions.RequestException as e:
                st.error(f"Upload failed: {e}")

st.info("Ask medical queries, log patient vitals, or calculate dosages.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter patient details, dosage queries, or medical questions..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing intent and executing tools..."):
            try:
                payload = {
                    "user_query": prompt,
                    "session_id": st.session_state.session_id
                }
                response = requests.post(f"{API_BASE_URL}/question/", params=payload)
                response.raise_for_status() 
                
                answer = response.json().get("response", "Error: Malformed API response.")
                st.markdown(answer)
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except requests.exceptions.ConnectionError:
                st.error("🚨 Critical Error: Cannot connect to the FastAPI backend. Ensure `uvicorn main:app --reload` is running.")
            except requests.exceptions.HTTPError as e:
                st.error(f"🚨 HTTP Error {e.response.status_code}: {e.response.text}")
            except Exception as e:
                st.error(f"🚨 Unexpected System Error: {str(e)}")