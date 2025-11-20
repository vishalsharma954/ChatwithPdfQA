import streamlit as st
from PyPDF2 import PdfReader
import openai
from dotenv import load_dotenv
import os

# Load API key from .env file
load_dotenv()

# Initialize OpenAI client (auto picks key from env)
openai.api_key = os.getenv("OPENAI_API_KEY")

# LangChain imports (correct for LangChain 1.x)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📘 PDF RAG App (LangChain + OpenAI + Chroma)")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:

    # -----------------------------
    # Extract PDF Text
    # -----------------------------
    pdf_reader = PdfReader(uploaded_file)
    raw_text = ""

    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text + "\n"

    st.success("PDF Loaded Successfully!")

    # -----------------------------
    # Split into Chunks
    # -----------------------------
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(raw_text)
    st.write(f"📑 Total Chunks: {len(chunks)}")

    # -----------------------------
    # Create Embeddings & Vector DB
    # -----------------------------
    embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.getenv("OPENAI_API_KEY")
    )

    vectordb = Chroma.from_texts(
        chunks,
        embedding=embeddings,
        collection_name="pdf_rag_store"
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    # -----------------------------
    # Ask Question
    # -----------------------------
    query = st.text_input("Ask a question from this PDF")

    if query:
        # Step 1: Retrieve relevant chunks
        # docs = retriever.get_relevant_documents(query)
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Step 2: Create prompt
        prompt = f"""
Use the following PDF content to answer the question.
If the answer is not in the document, say:
"I cannot find this information in the PDF."

PDF Content:
{context}

Question:
{query}

Answer:
"""

        # Step 3: Call OpenAI LLM
        response = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.choices[0].message.content


        st.subheader("🧠 Answer:")
        st.success(answer)
