import streamlit as st
import requests
import os
import base64
import json

st.set_page_config(page_title="Chat with PDF", layout="wide")
st.title("Chat with your PDF (Text + Visual QA)")

# --- Upload PDF ---
st.subheader("1. Upload your PDF")
pdf_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if pdf_file:
    with st.form("upload_form"):
        infer_table_structure = st.checkbox("Infer Table Structure", value=True)
        strategy = st.selectbox("PDF Parsing Strategy", ["hi_res", "fast", "ocr_only"], index=0)
        chunking_strategy = st.selectbox("Chunking Strategy", ["by_title", "default"], index=0)
        
        max_characters = st.number_input("Max characters per chunk", value=10000)
        combine_text_under_n_chars = st.number_input("Combine text under N chars", value=2000)
        new_after_n_chars = st.number_input("New chunk after N chars", value=6000)

        submit_button = st.form_submit_button("Upload & Process")

        if submit_button:
            with st.spinner("Uploading and processing your PDF..."):
                files = {"file": (pdf_file.name, pdf_file, "application/pdf")}
                data = {
                    "infer_table_structure": str(infer_table_structure),
                    "strategy": strategy,
                    "chunking_strategy": chunking_strategy,
                    "max_characters": max_characters,
                    "combine_text_under_n_chars": combine_text_under_n_chars,
                    "new_after_n_chars": new_after_n_chars,
                }
                res = requests.post(f"http://127.0.0.1:8000/upload_pdf/", files=files, data=data)

                if res.status_code == 200:
                    resp = res.json()
                    st.success("PDF processed successfully!")
                    st.session_state["session_id"] = resp["session_id"]
                    
                    # Load and cache images
                    session_id = resp["session_id"]
                    image_file_path = f"./chroma_db/{session_id}_images.txt"
                    if os.path.exists(image_file_path):
                        with open(image_file_path, "r") as f:
                            content = f.read()
                            images = content.split("---IMAGE_SEPARATOR---")
                            st.session_state["cached_images"] = [img.strip() for img in images if img.strip()]
                else:
                    st.error("Failed to process file: " + res.text)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# --- Ask Questions ---
if "session_id" in st.session_state:
    st.subheader("2. Ask a Question about the PDF")
    
    # Display conversation history
    for entry in st.session_state["chat_history"]:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.write(entry["answer"])
            # Display related images if any
            if "images" in entry and entry["images"]:
                for img in entry["images"]:
                    st.image(f"data:image/jpeg;base64,{img}", use_column_width=True)
    
    # Question input
    question = st.chat_input("Ask about the PDF")

    if question:
        # Add user question to chat UI
        with st.chat_message("user"):
            st.write(question)
        
        # Get response
        with st.spinner("Generating answer..."):
            payload = {
                "session_id": st.session_state["session_id"],
                "question": question
            }
            res = requests.post(f"http://127.0.0.1:8000/ask_question/", data=payload)

            if res.status_code == 200:
                response_data = res.json()
                answer = response_data["answer"]
                
                # Get references to images if available
                related_images = []
                if "source_documents" in response_data:
                    for doc in response_data["source_documents"]:
                        # Check if doc is an image (base64)
                        try:
                            if isinstance(doc, str) and doc.startswith("/9j"):
                                related_images.append(doc)
                        except:
                            pass
                
                # If no images found directly, try to find relevant images by content matching
                if not related_images and "cached_images" in st.session_state:
                    # Simple keyword matching - could be improved
                    keywords = question.lower().split()
                    if any(keyword in answer.lower() for keyword in ["image", "figure", "picture", "diagram", "wavelet", "svm"]):
                        # In a real app, you might want more sophisticated matching
                        related_images = st.session_state["cached_images"][:1]  # Just show first image as example
                
                # Display answer with assistant chat bubble
                with st.chat_message("assistant"):
                    st.write(answer)
                    # Display any related images
                    for img in related_images:
                        # st.image(f"data:image/jpeg;base64,{img}", use_column_width=True)
                        # When displaying images in the app
                        st.image(f"data:image/jpeg;base64,{img}", use_container_width=True, width=300)  # Limit width to 300 pixels
                
                # Save to history
                st.session_state["chat_history"].append({
                    "question": question,
                    "answer": answer,
                    "images": related_images
                })
            else:
                with st.chat_message("assistant"):
                    st.error("Error while answering: " + res.text)