import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Import OpenAI client (works perfectly with openai==1.26.0)
from openai import OpenAI

client = OpenAI()  # auto-picks API key from environment

# LangChain imports (stable versions)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_text(raw_text)
    st.write(f"📑 Total Chunks: {len(chunks)}")

    # -----------------------------
    # Create Embeddings & Vector DB
    # -----------------------------
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"
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
        # Retrieve relevant chunks
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Create prompt
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

        # Call OpenAI LLM
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        answer = response.choices[0].message.content

        st.subheader("🧠 Answer:")
        st.success(answer)
