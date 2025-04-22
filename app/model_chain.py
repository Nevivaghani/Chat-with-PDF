import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()

def get_summarize_chain_groq():
    from langchain_groq import ChatGroq
    model = ChatGroq(temperature=0.5, model="meta-llama/llama-4-scout-17b-16e-instruct", api_key=os.getenv("GROQ_API_KEY"))
    prompt = ChatPromptTemplate.from_template("""
        You are an assistant tasked with summarizing tables and text.
        Give a concise summary of the table or text.
        Respond only with the summary.
        Table or text chunk: {element}
    """)
    return {"element": lambda x: x} | prompt | model | StrOutputParser()

def get_image_description_chain():
    prompt_template = """Describe the image in detail. For context,
    the image is part of a research paper explaining the transformers
    architecture. Be specific about graphs, such as bar plots."""

    prompt = ChatPromptTemplate.from_messages([
        (
            "user",
            [
                {"type": "text", "text": prompt_template},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image}"}},
            ],
        )
    ])

    model = GoogleGenerativeAI(
        google_api_key=os.environ["GENAI_API_KEY"],
        model="gemini-2.0-flash",
        temperature=0
    )

    return prompt | model | StrOutputParser()

def get_qa_chain(retriever, memory=None):
    """Create a QA chain with optional memory support"""

    # Create a better prompt
    template = """You are an expert at answering questions based on PDF documents. 
    I will provide you with context information extracted from a PDF, and you should answer the question using ONLY this information.
    
    Context information:
    {context}
    
    Question: {question}
    
    Instructions:
    1. Answer the question thoroughly using the provided context.
    2. If the answer is explicitly stated in the context, provide it precisely.
    3. Include specific numbers, percentages, and statistics from the context if relevant.
    4. If the context includes information about models like SVM, random forest, or logistic regression, include their performance metrics.
    5. If there are images relevant to the question, reference and describe them.
    6. If the information to answer the question is not in the context, state clearly that you cannot find this information in the provided document.
    
    Your answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GENAI_API_KEY"),
        temperature=0
    )

    # Use the simpler RetrievalQA chain for all cases
    from langchain.chains import RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    
    # If we have memory, update it manually after getting the response
    if memory is not None:
        # We'll handle memory updates in the FastAPI endpoint
        pass
    
    return qa_chain