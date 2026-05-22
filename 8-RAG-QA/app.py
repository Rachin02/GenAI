import streamlit as st
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model = "llama-3.1-8b-instant")

# prompt template for llm input
prompt = ChatPromptTemplate.from_template(
    """
        Answer the questions based on the provided context only. Please provide the
        most accurate response based on the question
        <context>
        {context}
        <context>
        Question: {input}

    """
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embedding = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("8-RAG-QA/research_papers")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        print("Documents:", st.session_state.final_documents)
        print("Document count:", len(st.session_state.final_documents))
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embedding)

st.title("RAG Document Q&A with Groq")

user_question = st.text_input("Enter your question about the research paper")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector database is ready.")


import time

if user_question:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retriever_chain.invoke({"input":user_question})
    print(f"Response time : {time.process_time() - start}")

    st.write(response['answer'])


    ## with a streamlit expander
    with st.expander("Document similarity search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("----"*10)





# streamlit run 8-RAG-QA/app.py