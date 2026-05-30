import validators
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_classic.chains.summarize import load_summarize_chain



st.set_page_config(page_title= "URL summarizer", page_icon = "🦜" )
st.title("Langchain: Summarize text from Youtube video or website")
st.subheader("Summarize URL")

input_url = st.text_input("Enter URL", label_visibility= "collapsed")


with st.sidebar:
    grop_api = st.text_input("Enter grok API", type = "password")

if not grop_api:
    st.info("Provide your grop API key")
    st.stop()

llm = ChatGroq(model = "llama-3.1-8b-instant", api_key= grop_api)

prompt_template = """
Provide a summary of the following content in 300 words.
Content: {text}
"""

prompt = PromptTemplate(template= prompt_template, input_variables= ["text"])

if st.button("summarize"):
    if not grop_api.strip() or not input_url.strip():
        st.error("Provide the information to get start")
    elif not validators.url(input_url):
        st.error("Enter an valid URL address")

    else:
        try:
            with st.spinner("Summarizing....."):
                if "youtube.com" in input_url:
                    loader = YoutubeLoader.from_youtube_url(input_url, add_video_info = True)
    
                else:
                    loader = UnstructuredURLLoader(urls = [input_url], ssl_verify = False,  headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})

                
                docs = loader.load()
                if not docs:
                    st.info("Could not extract any content from this URL.")
              
                
                # chain for summarization
                chain = load_summarize_chain(llm , prompt = prompt, chain_type= "stuff")
                summary = chain.run(docs)
                st.success(summary)


        except Exception as e:
            st.exception(f"Exception: {e}")


    




# streamlit run 13-youtube_video_summarization/app.py