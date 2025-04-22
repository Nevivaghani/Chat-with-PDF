
# 📄 Chat with Your PDF (Text + Visual QA)

This project enables interactive question-answering with PDFs through both textual and visual content. Upload a PDF, extract structured and unstructured data (including tables and images), and ask intelligent questions using a powerful multi-modal pipeline backed by LangChain, HuggingFace, and Gemini/Groq LLMs.


## Features

🖼 Extracts text, tables, and images from PDFs

💬 Asks questions about content using LangChain QA chains

📊 Summarizes tables and nearby image content using Groq/Gemini

🧠 Remembers chat history using conversational memory

🖥️ FastAPI backend + Streamlit frontend

🗃 Stores vector embeddings with ChromaDB for retrieval




## How It Works

1. Upload a PDF via the Streamlit UI

2. The backend:

   - Parses the PDF using unstructured

   - Summarizes content with Groq LLM

   - Extracts and describes images using Gemini

   - Stores vectors in ChromaDB for fast retrieval

3. Ask questions — get responses powered by LangChain and Gemini

4. If relevant, visual images related to your question will be shown!


## 🧩 Tech Stack

**Frontend** : Streamlit

**Backend** : FastAPI

**Vector Store** : ChromaDB

**Embeddings** : sentence-transformers/all-MiniLM-L6-v2

**LLMs** : Groq (LLaMA) and Google Gemini (via langchain_google_genai)

**PDF Parsing** : unstructured

**RAG** : LangChain RetrievalQA with multi-vector retriever


## 🛠️ Setup Instructions

Clone the project

```bash
  git clone https://github.com/Nevivaghani/Chat-with-PDF.git
```

Go to the chat with pdf 4

```bash
  cd chat with pdf 4
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Set environment variables

Create a .env file and add:

```bash
GENAI_API_KEY=your_google_genai_api_key
GROQ_API_KEY=your_groq_api_key
```

Run FastAPI server
```bash
uvicorn main:app --reload
```

Run Streamlit frontend
```bash
streamlit run streamlit_app.py

```
## 📌 Example Use Cases


- Explore research papers and get text + image-based explanations

- Summarize and extract tables from technical PDFs

- Ask about specific figures or visual diagrams

- Find model comparisons, metrics, or methods mentioned



## 🧪 Sample Question Prompts

- “What does the table say about model accuracy?”

- “Describe the bar plot in Figure 2.”

- “Who is the author of this paper?”

- “What technique outperforms SVM?”


## Screenshots

![App Screenshot][def]

![App Screenshot][def2]

![App Screenshot][def3]

![App Screenshot][def4]

![App Screenshot][def5]

[def]: ./assets/chat%20with%20pdf1.png

[def2]: ./assets/chat%20with%20pdf2.png

[def3]: ./assets/chat%20with%20pdf3.png

[def4]: ./assets/chat%20with%20pdf4.png

[def5]: ./assets/chat%20with%20pdf5.png


