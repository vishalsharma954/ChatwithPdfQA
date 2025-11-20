import streamlit as st
import fitz
import os
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken
import chromadb
from chromadb.utils import embedding_functions

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create Chroma client (in-memory, deployment safe)
chroma_client = chromadb.Client()

st.title("📘 PDF RAG App (OpenAI + Chroma, No LangChain)")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

def chunk_text(text, chunk_size=1000, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

if uploaded_file:

    # Extract text using PyMuPDF (best on cloud)
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    raw_text = ""
    for page in doc:
        raw_text += page.get_text()

    st.success("PDF Loaded Successfully!")

    # Chunk text manually
    chunks = chunk_text(raw_text)
    st.write(f"📑 Total Chunks: {len(chunks)}")

    # Create embedding function
    embed_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-large"
    )

    # Create a Chroma collection
    collection = chromadb.Client().create_collection(
        name="pdf_rag",
        embedding_function=embed_fn
    )

    # Insert chunks
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)

    query = st.text_input("Ask a question from this PDF")

    if query:
        # Retrieve top 4 matching chunks
        results = collection.query(query_texts=[query], n_results=4)
        retrieved_chunks = results["documents"][0]
        context = "\n\n".join(retrieved_chunks)

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

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.choices[0].message.content

        st.subheader("🧠 Answer:")
        st.success(answer)
