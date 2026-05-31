import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import Tool
from langchain_classic.agents.initialize import initialize_agent
from langchain_classic.chains import LLMMathChain, LLMChain
from langchain_classic.agents.agent_types import AgentType


st.set_page_config(page_title="Text To MAth Problem Solver And Data Serach Assistant",page_icon="🧮")
st.title("Text to Math problem solver")

gpt_api = st.sidebar.text_input("enter GPT API key", type = "password")

if not gpt_api:
    st.info("Please first provide your groq api")
    st.stop()

llm = ChatOpenAI(model = "gpt-4o-mini", api_key= gpt_api)

wiki_wrapper = WikipediaAPIWrapper()
wiki_tool = Tool(
    name = "Wikipedia tool",
    func= wiki_wrapper.run,
    description= "A tool for searching the Internet to find the various information on the topics mentioned"
)

math_chain = LLMMathChain.from_llm(llm= llm)
math_tool = Tool(
    name = "calculator",
    func= math_chain.run,
    description= "A tools for answering math related questions. Only input mathematical expression need to be provided"
)

prompt = """ 
You are an agent assign for solving users mathematical question. Logically arrive at the solution and provide a detailed explanation
and display it point wise for the question below
Question:{question}
Answer:
"""
prompt_template = PromptTemplate(
    input_variables= ["question"],
    template = prompt
)

chain = LLMChain(llm= llm , prompt= prompt_template)

reasoning_tool = Tool(
    name = "reasoning tool",
    func= chain.run,
    description= "A tool for answering logic-based and reasoning questions."
)

## combine all tool and chain
agent = initialize_agent(
    llm= llm,
    tools= [wiki_tool, math_tool, reasoning_tool],
    agent= AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)
agent.handle_parsing_errors = True


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role":"assistant", "content":"Hi! I am a math chatbot who can answer your math question"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# query = st.text_area("Enter your question")
query = st.chat_input("Enter your question")

if query:
    if query:
        st.session_state.messages.append({"role":"user","content":query})
        st.chat_message("user").write(query)

        with st.spinner("Generating asnwer...."):
            st_cb = StreamlitCallbackHandler(st.container())
            response = agent.run(query, callbacks= [st_cb])

        st.session_state.messages.append({"role":"assistant", "content": response})
        st.write(response)
    else:
        st.write("Please enter question")



# streamlit run 14-MathSolver/ap.py
