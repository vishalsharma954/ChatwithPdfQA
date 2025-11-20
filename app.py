import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os

# Load env
load_dotenv()

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# LangChain old stable imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

st.title("📘 PDF RAG App (LangChain + OpenAI)")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)
    raw_text = ""

    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text + "\n"

    st.success("PDF Loaded Successfully!")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )

    vectordb = Chroma.from_texts(
        chunks, embedding=embeddings, collection_name="pdf_rag"
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    query = st.text_input("Ask a question:")

    if query:
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
Use this PDF content to answer the question.
If answer not found, say "Not in PDF".

PDF Content:
{context}

Question:
{query}

Answer:
"""

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.choices[0].message.content
        st.success(answer)
