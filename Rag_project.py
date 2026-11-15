import fitz
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

load_dotenv()
st.set_page_config(page_title="AICADEMY HELPER", layout="wide", page_icon="🎓")
st.title("🎓AICADEMY HELPER")
st.markdown("Your Smart AI Study Assistant | Ask Questions From Your Own Notes & PDFs")
st.markdown("Upload your class PDF → Ask any question → Get answer ONLY from your PDF")
st.divider()
with st.sidebar:
    st.header("⚙️ Control Panel")
    st.markdown("How it works?")
    st.markdown("1. Upload PDF\n2. AI reads & understands it\n3. Ask question\n4. Get answer with source")
    st.divider()
    st.markdown(" 📚 Demo PDFs")
    st.markdown("- Module 04 - Generative AI\n- Module 03 - Python Basics\n- Module 02 - AI Concepts")
    st.divider()
    st.info("Built with RAG Pipeline | Aicademy 360° Assignment #3")

    def extract_text_from_pdf(pdf_file):
     """Extract all text from uploaded PDF"""
     doc =fitz.open(stream=pdf_file.read(), filetype="pdf")
     full_text =     ""
     for page in doc:
      full_text +=        page.get_text() + "\n"
     return full_text

def split_into_chunks(text, chunk_size=800):
    """Split long text into small chunks for RAG"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) < chunk_size:
            current += sentence + " "
        else:
            if current.strip():
                chunks.append(current.strip())
            current = sentence + " "
    if current.strip():
        chunks.append(current.strip())
    return chunks

def find_best_chunks(question, chunks, top_k=3):
    """Find most relevant chunks using TF-IDF + Cosine Similarity"""
    if not chunks:
        return []
    vectorizer = TfidfVectorizer().fit(chunks + [question])
    vectors = vectorizer.transform(chunks + [question])
    scores = cosine_similarity(vectors[-1], vectors[:-1])[0]
    top_indices = scores.argsort()[-top_k:][::-1]
    return [(chunks[i], round(float(scores[i]), 3)) for i in top_indices]

if "history" not in st.session_state:
    st.session_state.history = []
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None

left, right = st.columns([1, 1.2])

with left:
    st.subheader("📄 1. Upload Your Notes")
    uploaded = st.file_uploader("Upload Class PDF", type=["pdf"])

    if uploaded:
        if st.session_state.file_name != uploaded.name:
            with st.spinner("AI is reading your PDF..."):
                text = extract_text_from_pdf(uploaded)
                chunks = split_into_chunks(text)
                st.session_state.chunks = chunks
                st.session_state.file_name = uploaded.name
                st.session_state.total_text = text
                st.session_state.history = []

        st.success(f"✅ {uploaded.name} uploaded!")
        col_a, col_b = st.columns(2)
        col_a.metric("Chunks Created", len(st.session_state.chunks))
        col_b.metric("Total Words", len(st.session_state.total_text.split()))
    else:
        st.warning("Please upload a PDF to start")
        st.markdown("**Try Demo Without PDF:**")
        if st.button("Load Demo Notes"):
            demo_text = (
                "RAG stands for Retrieval Augmented Generation. "
                "RAG has 4 steps. First we upload documents. "
                "Second we split them into chunks. "
                "Third we search relevant chunks using embeddings. "
                "Fourth AI generates answer from those chunks only. "
                "A good frontend should be modern and user friendly. "
                "A good product solves a real world problem."
            )
            st.session_state.chunks = split_into_chunks(demo_text)
            st.session_state.file_name = "Demo Notes"
            st.session_state.total_text = demo_text
            st.session_state.history = []
            st.success("Demo Notes Loaded!")

with right:
    st.subheader("💬 2. Ask Question")
    question = st.text_input(
        "Your Question",
        placeholder="Example: What is RAG?",
    )

    if st.button("Ask AI 🚀"):
        if st.session_state.chunks is None:
            st.error("Please upload a PDF first or load Demo Notes!")
        elif not question.strip():
            st.error("Please write a question!")
        else:
            best_chunks = find_best_chunks(question, st.session_state.chunks)

            # ----- HALLUCINATION BLOCKER -----
            # If relevance score is too low, answer is NOT in PDF
            # So we DO NOT generate a fake answer
            if not best_chunks or best_chunks[0][1] < 0.15:
                answer = (
                    "⚠️ Sorry, this question's answer is not available in your uploaded PDF. "
                    "Please ask a question related to your PDF only."
                )
                st.session_state.history.append((question, answer, []))
            else:
                # Answer ONLY from PDF chunks, no outside knowledge
                main_answer = best_chunks[0][0]
                extra_detail = best_chunks[1][0] if len(best_chunks) > 1 else ""
                answer = (
                    f"**According to your PDF:**\n\n{main_answer}\n\n"
                    f"---\n**Extra Detail from PDF:** {extra_detail}"
                )
                st.session_state.history.append((question, answer, best_chunks))

    # Show Chat History
    for q, a, sources in reversed(st.session_state.history):
        st.markdown(f"**🧑 You:** {q}")
        st.markdown(f"**🤖 Aicademy Helper:** {a}")
        if sources:
            with st.expander("📖 View Sources from PDF"):
                for i, (chunk, score) in enumerate(sources):
                    st.markdown(f"**Source {i+1} - Relevance Score: {score}**")
                    st.write(chunk[:600])
        st.divider()

st.divider()
st.markdown("🌟 Why Aicademy Helper is a Real Product?")
c1, c2, c3 = st.columns(3)
c1.markdown("🎯 For Students**\n\nNo more ratta. Understand from your own notes.")
c2.markdown("🏫 For Academies**\n\nEvery academy can have its own AI assistant.")
c3.markdown("💼 Business Model**\n\nMonthly subscription per academy. Marketable Product.")
st.caption("Made for Aicademy 360° | Module 04 | Assignment #3 | RAG-Based AI Application")

