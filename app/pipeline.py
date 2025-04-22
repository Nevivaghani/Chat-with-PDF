import os
import uuid
import re
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Table
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever

from utils import get_images_base64
from model_chain import get_summarize_chain_groq, get_image_description_chain


def sanitize_collection_name(name):
    """
    Ensure collection name meets Chroma's requirements:
    1. Contains 3-63 characters
    2. Starts and ends with an alphanumeric character
    3. Otherwise contains only alphanumeric characters, underscores or hyphens
    4. Contains no two consecutive periods
    5. Is not a valid IPv4 address
    """
    # Remove any non-alphanumeric characters except underscores and hyphens
    name = re.sub(r'[^a-zA-Z0-9_\-]', '', name)
    
    # Ensure it starts and ends with alphanumeric
    if not name[0].isalnum():
        name = 'a' + name
    if not name[-1].isalnum():
        name = name + 'z'
    
    # Ensure minimum length of 3
    while len(name) < 3:
        name += 'x'
    
    # Ensure maximum length of 63
    if len(name) > 63:
        name = name[:63]
        # Make sure it still ends with alphanumeric
        if not name[-1].isalnum():
            name = name[:-1] + 'z'
    
    return name


def find_closest_text_to_image(chunks, img):
    """Find text chunks that are close to an image in the document"""
    # This is a simplified approach - in reality, you'd need to analyze the PDF structure
    for i, chunk in enumerate(chunks):
        if "Image" in str(type(chunk)):
            if chunk.metadata.image_base64 == img:
                # Look at surrounding chunks
                context = []
                # Look at previous chunk
                if i > 0:
                    context.append(str(chunks[i-1]))
                # Look at next chunk
                if i < len(chunks) - 1:
                    context.append(str(chunks[i+1]))
                return " ".join(context)
    return ""

def run_pdf_pipeline(
    file_path,
    infer_table_structure=True,
    strategy="hi_res",
    extract_image_block_types=["image"],
    extract_image_block_to_payload=True,
    chunking_strategy="by_title",
    max_characters=10000,
    combine_text_under_n_chars=2000,
    new_after_n_chars=6000,
):
    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=infer_table_structure,
        strategy=strategy,
        extract_image_block_types=extract_image_block_types,
        extract_image_block_to_payload=extract_image_block_to_payload,
        chunking_strategy=chunking_strategy,
        max_characters=max_characters,
        combine_text_under_n_chars=combine_text_under_n_chars,
        new_after_n_chars=new_after_n_chars,
    )

    texts = [chunk for chunk in chunks if "CompositeElement" in str(type(chunk))]
    tables = []
    for chunk in chunks:
        for el in chunk.metadata.orig_elements:
            if isinstance(el, Table):
                tables.append(el)

    images = get_images_base64(chunks)

    summarize_chain = get_summarize_chain_groq()
    
    text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})
    tables_html = [table.metadata.text_as_html for table in tables]
    table_summaries = summarize_chain.batch(tables_html, {"max_concurrency": 3})

    # Image summary
    # img_chain = get_image_description_chain()
    img_chain = get_image_description_chain()
    img_metadata = []
    for img in images:
        try:
            # Get basic description
            result = img_chain.invoke({"image": img})
            
            # Try to find a nearby text chunk that might describe the image
            # This is a simplified approach - you might need more complex logic
            closest_text = find_closest_text_to_image(chunks, img)
            
            # Store both the description and nearby text as metadata
            img_metadata.append({
                "description": result,
                "nearby_text": closest_text,
                "image": img
            })
        except Exception as e:
            print("Error processing image:", e)
    results = []
    for img in images:
        try:
            result = img_chain.invoke({"image": img})
            results.append(result)
        except Exception as e:
            print("Retrying due to:", e)
            continue

    # Generate a session ID - use UUID for reliable uniqueness
    raw_session_id = os.path.basename(file_path).replace(".pdf", "")
    session_id = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID for cleaner ID
    
    # Create a valid collection name
    collection_name = sanitize_collection_name(f"pdf-{session_id}")
    
    # Save the collection name for future retrieval
    os.makedirs("./chroma_db", exist_ok=True)
    with open(f"./chroma_db/{session_id}_info.txt", "w") as f:
        f.write(collection_name)
    
    # Setup vectorstore with proper collection name
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'),
        persist_directory="./chroma_db"
    )

    store = InMemoryStore()
    id_key = "doc_id"
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key)

    doc_ids = [str(uuid.uuid4()) for _ in texts]
    vectorstore.add_documents([Document(page_content=summary, metadata={id_key: doc_ids[i]})
                               for i, summary in enumerate(text_summaries)])
    # store.mset(list(zip(doc_ids, texts)))
    store.mset([(doc_id, Document(page_content=str(text), metadata={"doc_id": doc_id})) 
            for doc_id, text in zip(doc_ids, texts)])

    table_ids = [str(uuid.uuid4()) for _ in tables]
    vectorstore.add_documents([Document(page_content=summary, metadata={id_key: table_ids[i]})
                               for i, summary in enumerate(table_summaries)])
    # store.mset(list(zip(table_ids, tables)))
    store.mset([(table_id, Document(page_content=str(table), metadata={"doc_id": table_id})) 
            for table_id, table in zip(table_ids, tables)])

    img_ids = [str(uuid.uuid4()) for _ in images]
    vectorstore.add_documents([Document(page_content=summary, metadata={id_key: img_ids[i]})
                               for i, summary in enumerate(results)])
    # store.mset(list(zip(img_ids, images)))
    store.mset([(img_id, Document(page_content=img, metadata={"doc_id": img_id})) 
            for img_id, img in zip(img_ids, images)])

    # Also store images in a separate file for easy access from streamlit
    if images:
        with open(f"./chroma_db/{session_id}_images.txt", "w") as f:
            for img in images:
                f.write(f"{img}\n---IMAGE_SEPARATOR---\n")

    return retriever, session_id