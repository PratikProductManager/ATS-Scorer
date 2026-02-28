import os
import shutil
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA

# -------------------------------------------------
# Setup
# -------------------------------------------------
load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))

embedding = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")

llm = ChatNVIDIA(
    model="meta/llama-3.3-70b-instruct",
    temperature=0
)

VECTOR_DB_PATH = f"{working_dir}/doc_vectorstore"


# -------------------------------------------------
# 1. Load Resume + ATS Guidelines into Chroma
# -------------------------------------------------
def process_documents_to_chroma(resume_file):

    documents = []

    # ---- Load Resume ----
    resume_loader = PyPDFLoader(f"{working_dir}/{resume_file}")
    documents.extend(resume_loader.load())

    # ---- Load ATS Guidelines ----
    guidelines_loader = TextLoader(f"{working_dir}/ats_guidelines.txt")
    documents.extend(guidelines_loader.load())

    # ---- Split text ----
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )
    texts = splitter.split_documents(documents)

    # ---- Store in Chroma ----
    if os.path.exists(VECTOR_DB_PATH):
        shutil.rmtree(VECTOR_DB_PATH)

    Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=VECTOR_DB_PATH
    )

    return "Documents processed and stored in vector DB."


# -------------------------------------------------
# 2. Create ATS Evaluation Chain
# -------------------------------------------------
def create_ats_chain(retriever):

    prompt = PromptTemplate.from_template(
        """
You are an advanced Applicant Tracking System (ATS).

Evaluate the resume against the job description using ATS best practices.

JOB DESCRIPTION:
{question}

RETRIEVED CONTEXT (Resume + ATS Guidelines):
{context}

Return:

1. ATS Score (0-100)
2. Missing / Non-matching Skills
3. Resume Strengths
4. Scope of Improvement
5. Clear Recommendations

Be concise, structured, and professional.
"""
    )

    ats_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return ats_chain


# -------------------------------------------------
# 3. Evaluate Resume Against Job Description
# -------------------------------------------------
def evaluate_resume(job_description):

    vectordb = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embedding
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 6})

    ats_chain = create_ats_chain(retriever)

    result = ats_chain.invoke(job_description)

    return result
