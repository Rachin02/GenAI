import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model = "llama-3.1-8b-instant")
embedding = OpenAIEmbeddings()


st.title("Conversational RAG with PDF upload and chat history")
st.write("Upload PDF and chat with their content")

session_id = st.text_input("Session ID", value = "default_session")

if 'store' not in st.session_state:
    st.session_state.store = {}


uploaded_files = st.file_uploader("Choose a PDF file", type = "pdf", accept_multiple_files=True)


# process upload file
if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        tempPdf = "9-RAG-chat-with-pdf/temp.pdf"
        with open(tempPdf, "wb") as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name

        loader = PyPDFLoader(tempPdf)
        docs = loader.load()
        documents.extend(docs)


    # split and create embedding for the documents
    # print(documents)
    st.write(docs[0].page_content[:1000])
    text_splitters = RecursiveCharacterTextSplitter(chunk_size= 1000, chunk_overlap = 500)
    final_documents = text_splitters.split_documents(documents)
    vectorstore = Chroma.from_documents(final_documents, embedding)
    retriever = vectorstore.as_retriever()



    contextualize_q_system_prompt = (
        """
    Given a chat history and the latest user question, which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do not answer the question, just reformulate it if needed and otherwise return it as is.
    """

    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("user","{input}")
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # prompt for answering question
    system_prompt = (
        """ 
    You are an assistant for question answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.\n\n
    {context}

    """
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",system_prompt),
            MessagesPlaceholder("chat_history"),
            ("user","{input}")
        ]
    )

    question_answering_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answering_chain)



    def get_session_history(session_id:str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        
        return st.session_state.store[session_id]



    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key= "chat_history",
        output_messages_key= "answer"
    )

    user_question = st.text_input("Your question : ")

    if user_question:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_question },
            config = {
                "configurable":{"session_id":session_id} #constructs a key like abc123 in store
            }
            
        )

        st.write(st.session_state.store)
        st.write("AI response",response['answer'])
        st.write("chat History: ", session_history.messages)






# streamlit run 9-RAG-chat-with-pdf/app.py