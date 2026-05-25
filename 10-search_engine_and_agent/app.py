import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun, DuckDuckGoSearchResults
from langchain.agents import create_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from dotenv import load_dotenv
load_dotenv()


# wikipedia tool
wiki_wrapper = WikipediaAPIWrapper(top_k_results= 1, doc_content_chars_max= 300)
wiki = WikipediaQueryRun(api_wrapper= wiki_wrapper)

# arxiv tool
arxiv_wrapper = ArxivAPIWrapper(top_k_results= 1, doc_content_chars_max= 300)
arxiv = ArxivQueryRun(api_wrapper= arxiv_wrapper)

search = DuckDuckGoSearchResults(name = "search")


st.title("🔎 LangChain - Chat with search")

# In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.

st.sidebar.title("Setting")
api_key = st.sidebar.text_input("Enter your GPT API key: ", type = "password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role":"assistant","content":"Hi,I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

prompt = """

You are a helpful AI assistant.

Always explain the user's question in simple and easy-to-understand words.
Use proper structure, bullet points, and small examples if needed.
If useful, use simple text-based figures or diagrams for explanation.
Keep responses clear, beginner-friendly, and within a maximum of 10 sentences.

You MUST use tools before answering whenever external information is needed.

Use:
- wikipedia for factual knowledge
- arxiv for research papers
- search for latest/current information

For recent topics, research papers, unknown facts, or web information, always use the appropriate tool first before answering.
After using tools, summarize the information in simple words. use maximum 10 sentence for answer the question.

"""
model = ChatOpenAI(model = "gpt-5-nano",streaming=True) #model
tools = [wiki, arxiv, search] # tool
agent = create_agent(model=model, tools= tools, system_prompt= prompt, verbose = True ) # agent


if prompt:= st.chat_input(placeholder="Ask Question: "):

    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):

        # THIS SHOWS TOOL CALLS
        st_callback = StreamlitCallbackHandler(st.container())

        response = agent.invoke(
            {
                "messages": [
                    {"role": "user","content": prompt}
                            ]
            },
            {
                "callbacks": [st_callback]
            }
        )

        final_response = response["messages"][-1].content

        st.write(final_response)

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": final_response
    })


# streamlit run 10-search_engine_and_agent/app.py