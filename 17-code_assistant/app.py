from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate



st.set_page_config(page_title="Code Assistant", page_icon="💻")
st.title("Code Assistant")
st.caption("Powered by OpenAI + Langchain - Ask any coding question")


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role":"assistant","content": "Give me the coding question"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


s_prompt = """ 
 You are an expert coding assistant.
 when asked a coding question, always provide: \n
 1. A short explaination
 2. clear working code in a code block
 3. A brief breakdown of how the code works
"""

prompt = ChatPromptTemplate(
    [
        ("system",s_prompt),
        ("human","{question}")
    ]
)

model = ChatOpenAI(model = "gpt-5.4-nano", temperature = 0.9)

chain = prompt | model

# messages = []


user_qus = st.chat_input("Ask Question")

if user_qus:
    st.session_state.messages.append({"role":"user","content": user_qus})
    # messages.extend(HumanMessage(content=user_qus))
    st.chat_message("user").write(user_qus)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke({"question":st.session_state.messages})
            st.session_state.messages.append({"role":"assistant","content":response.content})
            # messages.extend(AIMessage(content= response.content))

            st.write(response.content)
            



# streamlit run 17-code_assistant/app.py