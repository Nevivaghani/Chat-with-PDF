import streamlit as st
import requests

# FastAPI endpoints
UPLOAD_ENDPOINT = "http://127.0.0.1:8000/upload_pdf/"
QUESTION_ENDPOINT = "http://127.0.0.1:8000/ask_question/"

st.title("📄 PDF Q&A Chatbot")

# --- Upload Section ---
st.header("Step 1: Upload Your PDF")

with st.form("upload_form"):
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    infer_table_structure = st.checkbox("Infer table structure", value=True)
    strategy = st.selectbox("PDF parsing strategy", ["hi_res", "fast", "ocr_only"])
    chunking_strategy = st.selectbox("Chunking Strategy", ["by_title", "default"])
    
    max_characters = st.number_input("Max characters per chunk", value=10000)
    combine_text_under_n_chars = st.number_input("Combine text under N chars", value=2000)
    new_after_n_chars = st.number_input("New chunk after N chars", value=6000)

    submitted = st.form_submit_button("Upload and Process")

    if submitted and uploaded_file:
        with st.spinner("Uploading and processing..."):
            files = {'file': (uploaded_file.name, uploaded_file, "application/pdf")}
            data = {
                'infer_table_structure': str(infer_table_structure),
                'strategy': strategy,
                'chunking_strategy': chunking_strategy,
                'max_characters': max_characters,
                'combine_text_under_n_chars': combine_text_under_n_chars,
                'new_after_n_chars': new_after_n_chars,
            }
            response = requests.post(UPLOAD_ENDPOINT, files=files, data=data)
            if response.status_code == 200:
                session_id = response.json().get("session_id")
                st.success("PDF processed successfully! Your session ID: " + session_id)
                st.session_state["session_id"] = session_id
            else:
                st.error("Failed to process file: " + response.text)

# --- Q&A Section ---
if "session_id" in st.session_state:
    st.header("Step 2: Ask a Question")
    question = st.text_input("Enter your question about the uploaded PDF")

    if st.button("Get Answer"):
        with st.spinner("Thinking..."):
            payload = {
                'session_id': st.session_state["session_id"],
                'question': question,
            }
            response = requests.post(QUESTION_ENDPOINT, data=payload)
            if response.status_code == 200:
                answer = response.json().get("answer")
                st.success("Answer:")
                st.markdown(answer)
            else:
                st.error("Failed to get an answer: " + response.text)
