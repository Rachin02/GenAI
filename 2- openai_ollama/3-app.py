from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser



prompt = ChatPromptTemplate.from_messages(
    [
        ("system","you are a helpful assistant. Please response to the question asked"),
        ("user","Question:{question}")

    ]
    )

llm = ChatOllama(model = "gemma:2b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser


st.title("Chat wih ollama... ")
input_text = st.text_input("What question you have in mind?")


if input_text:
    st.write(chain.invoke({"question":input_text}))


# streamlit run "2- openai_ollama"/3-app.py