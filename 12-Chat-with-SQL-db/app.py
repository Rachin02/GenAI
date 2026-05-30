import streamlit as st
import sqlite3
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_classic.agents.agent_types import AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine


st.set_page_config(page_title="Langchain: chat with SQL database", page_icon="🦜")
st.title("🦜 LangChain: Chat with SQL DB")

LOCALDB = "USE_LOCALDB"
MYSQL ="USE_MYSQL"

ratio_opt = ["Use sqlite 3 Database - student.db","Connect to your sql database"]
select_opt = st.sidebar.radio(label = "Choose the database which you want to chat", options=ratio_opt)


if ratio_opt.index(select_opt) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("provide MySQL host")
    mysql_user = st.sidebar.text_input("MySQL user")
    mysql_pass = st.sidebar.text_input("MySQL password", type = "password")
    mysql_db = st.sidebar.text_input("MySQL Database")
else:
    db_uri = LOCALDB


api_key = st.sidebar.text_input("Enter your gpt api key", type = "password")

if not db_uri:
    st.info("Please enter the database information and uri")
if not api_key:
    st.info("Please add you api key")
    st.stop()



## llm model
model = ChatOpenAI(model = "gpt-4o-mini", api_key = api_key, streaming = True)


@st.cache_resource(ttl = "2h")
def config_db(db_uri, mysql_host = None, mysql_user = None, mysql_pass = None, mysql_db = None):
    if db_uri == LOCALDB:
        # db_path = (Path(__file__).parent/"student.db").absolute()
        db_path = (Path(__file__).parent.parent/"student.db").absolute()
        creator = lambda : sqlite3.connect(f"file:{db_path}?mode = ro",uri = True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))

        # st.write(db_path)
    elif db_uri == MYSQL:
        if not (mysql_host and mysql_pass and mysql_user and mysql_db):
            st.error("Please provide all MySQL connection details.")
            st.stop()

        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_pass}@{mysql_host}/{mysql_db}"))


if db_uri == MYSQL:
    db = config_db(db_uri, mysql_host, mysql_user, mysql_pass, mysql_db)
else:
    db = config_db(db_uri)

# tool kits
toolkit = SQLDatabaseToolkit(db=db, llm= model)

agent = create_sql_agent(llm = model, toolkit= toolkit, verbose= True, agent_type= AgentType.ZERO_SHOT_REACT_DESCRIPTION)
agent.handle_parsing_errors = True

if "messages" not in st.session_state or st.button("clear message history"):
    st.session_state["messages"] = [{"role":"assistant", "content":"How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder= "Ask anything from the database")

if user_query:
    st.session_state.messages.append({"role":"user","content":user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query, callbacks=[streamlit_callback])
        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response)

        # response = agent.invoke(
        #     {"input": user_query},
        #     callbacks=[streamlit_callback]
        # )

        # output = response["output"]

        # st.session_state.messages.append({
        #     "role": "assistant",
        #     "content": output
        # })

        # st.write(output)

        # st.write(response["output"])






# python3 Chat-with-SQL-db/app.py
# streamlit run Chat-with-SQL-db/app.py