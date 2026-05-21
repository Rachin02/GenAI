import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

# langchain tracking
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = "QA chatbot"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistante. always answer the user question simply."),
        ("user","Question : {question}")
    ]
)

def generate_response(question, llm, temperature, max_token):
    model = ChatOllama(model = llm, temperature= temperature, max_tokens = max_token)
    output_parser = StrOutputParser()

    chain = prompt | model | output_parser

    response = chain.invoke({"question":question})
    return response


# title of of web
st.title("Chatbot with ollama")

# select the ollama model
llm = st.sidebar.selectbox("Select opensourc model", ["gemma:2b", "llama3.1:latest","mxbai-embed-large:latest"])

# select temperature
temperature = st.sidebar.slider("temperature", min_value= 0.0 , max_value= 1.0 ,value=0.7)
# max token
max_token = st.sidebar.slider("Max token", min_value= 30, max_value= 200, value = 100)

# main interface for user input
st.write("Ask any question")
text = st.text_input("You : ")

if text:
    if st.button("Generate"):
        response = generate_response(text, llm, temperature, max_token)
        st.write(response)
else:
    st.info("Enter a question to start")



# python3 7-chatbot/ollamaChat.py

# streamlit run 7-chatbot/ollamaChat.py