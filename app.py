import streamlit as st
import requests

st.set_page_config(page_title="PDF Private Brain", page_icon="🤖")
st.title("🤖 Ask me about your PDFs")

with st.sidebar:
    st.header("Upload Center")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file and st.button("Index PDF"):
        with st.spinner("Processing PDF..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            response = requests.post("http://127.0.0.1:8000/uploadfile/", files=files)
            if response.status_code == 200:
                st.success("File Indexed Successfully!")

st.info("Ask anything based on the documents you've uploaded.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is in this document?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating..."):
            res = requests.post(f"http://127.0.0.1:8000/question/?user_query={prompt}")
            answer = res.json().get("response", "Error connecting to AI.")
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})