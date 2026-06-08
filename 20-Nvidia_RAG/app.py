import os 
import streamlit as st
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnableLambda

from dotenv import load_dotenv
load_dotenv()

st.title("RAG pipline with NVIDIA inference")

def vector_embedding():
    if "vectors" not in st. session_state:
        st.session_state.embedding = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("20-Nvidia_RAG/research_papers")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embedding)


llm = ChatNVIDIA(model = "meta/llama-3.3-70b-instruct")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)


query =st.text_input("Enter Your Question From Doduments")


if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")


if prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    chain = create_retrieval_chain(retriever, document_chain)
    parser = StrOutputParser()
    chains = chain| RunnableLambda(lambda x: x["answer"]) | parser
    answer = chains.invoke({"input": query})

    st.write(answer)




# streamlit run 20-Nvidia_RAG/app.py