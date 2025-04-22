import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pipeline import run_pdf_pipeline
from model_chain import get_qa_chain
import base64

app = FastAPI()

# In-memory store for retrievers
retrievers = {}

def load_existing_retriever(session_id):
    """Load a retriever from disk based on session_id"""
    try:
        # Check if collection info exists
        info_path = f"./chroma_db/{session_id}_info.txt"
        if not os.path.exists(info_path):
            return None
            
        # Read collection name
        with open(info_path, "r") as f:
            collection_name = f.read().strip()
        
        # Recreate vector store
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain.storage import InMemoryStore
        from langchain.retrievers.multi_vector import MultiVectorRetriever
        
        vectorstore = Chroma(
            collection_name=f"multi-model-rag-{session_id}",  # Make collection name unique per session
            embedding_function=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'),
            persist_directory="./chroma_db"
        )
        
        # Create docstore and retriever
        store = InMemoryStore()
        id_key = "doc_id"
        retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key)
        
        # Note: In a production environment, you'd need to reload the documents into docstore
        # This simplified version will work for vector similarity search but might not 
        # return full document content properly
        
        return retriever
    except Exception as e:
        print(f"Error loading retriever: {e}")
        return None

@app.post("/upload_pdf/")
async def upload_pdf(
    file: UploadFile = File(...),
    infer_table_structure: bool = Form(True),
    strategy: str = Form("hi_res"),
    chunking_strategy: str = Form("by_title"),
    max_characters: int = Form(10000),
    combine_text_under_n_chars: int = Form(2000),
    new_after_n_chars: int = Form(6000)
):
    try:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Run pipeline
        retriever, session_id = run_pdf_pipeline(
            tmp_path,
            infer_table_structure=infer_table_structure,
            strategy=strategy,
            chunking_strategy=chunking_strategy,
            max_characters=max_characters,
            combine_text_under_n_chars=combine_text_under_n_chars,
            new_after_n_chars=new_after_n_chars,
        )

        retrievers[session_id] = retriever

        return JSONResponse(content={"message": "File processed successfully.", "session_id": session_id})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/ask_question/")
async def ask_question(session_id: str = Form(...), question: str = Form(...)):
    try:
        retriever = retrievers.get(session_id)
        
        # If retriever not found in memory, try to load from disk
        if not retriever:
            retriever = load_existing_retriever(session_id)
            if retriever:
                retrievers[session_id] = retriever
            else:
                return JSONResponse(status_code=404, content={"error": "Invalid session_id or session expired."})

        # Get or create memory for this session
        if "memories" not in globals():
            globals()["memories"] = {}
            
        if session_id not in globals()["memories"]:
            from langchain.memory import ConversationBufferMemory
            globals()["memories"][session_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            
        memory = globals()["memories"][session_id]
        
        # Create the QA chain
        qa_chain = get_qa_chain(retriever)
        
        # Get the answer - Use 'query' instead of 'question'
        response = qa_chain.invoke({"query": question})
        answer = response.get("result", "")
        
        # Process source documents differently
        source_docs = []
        text_contents = []
        relevant_images = []
        
        if "source_documents" in response:
            for doc in response["source_documents"]:
                if hasattr(doc, "page_content"):
                    content = doc.page_content
                    
                    # Check if this is an image
                    try:
                        # Attempt to decode as base64
                        base64.b64decode(content)
                        relevant_images.append(content)
                    except:
                        # Not an image, add to text contents
                        text_contents.append(content)
                        source_docs.append(content)
        
        # If no images were found directly, search for images by name matching
        if not relevant_images:
            # Extract potential names from the question
            import re
            name_match = re.search(r"(?:who is|about|regarding|Dr\.|Mr\.|Mrs\.|Prof\.) ([A-Z][a-z]+ [A-Z][a-z]+)", question)
            if name_match:
                person_name = name_match.group(1).lower()
                
                # Get all images
                image_file_path = f"./chroma_db/{session_id}_images.txt"
                all_images = []
                if os.path.exists(image_file_path):
                    with open(image_file_path, "r") as f:
                        content = f.read()
                        all_images = content.split("---IMAGE_SEPARATOR---")
                        all_images = [img.strip() for img in all_images if img.strip()]
                
                # Get all text chunks to check for name mentions
                from chromadb import PersistentClient
                
                try:
                    db = PersistentClient(path="./chroma_db")
                    info_path = f"./chroma_db/{session_id}_info.txt"
                    if os.path.exists(info_path):
                        with open(info_path, "r") as f:
                            collection_name = f.read().strip()
                        
                        collection = db.get_or_create_collection(collection_name)
                        
                        # Search for the name in the collection
                        results = collection.query(
                            query_texts=[person_name],
                            n_results=5
                        )
                        
                        # Get index of chunks that mentioned the name
                        mentioned_indices = []
                        for i, doc in enumerate(results["documents"][0]):
                            if person_name in doc.lower():
                                mentioned_indices.append(i)
                        
                        # If we found mentions of the name, use that to find nearby images
                        if mentioned_indices and all_images:
                            # For simplicity, just use the first image
                            # In a real system, you'd want to be smarter about finding the right image
                            relevant_images.append(all_images[0])
                except Exception as e:
                    print(f"Error searching for name in collection: {e}")
        
        return JSONResponse(content={
            "answer": answer, 
            "source_documents": source_docs,
            "relevant_images": relevant_images[:1]  # Limit to one image to avoid information overload
        })
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})