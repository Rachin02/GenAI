from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI
from langserve import add_routes

client = ChatOpenAI(model = "gpt-5-nano")

# creates prompt template
system_prompt = "translate the following sentences to {language}"
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("user","{text}")
    ]
)

# output parser
parser = StrOutputParser()

#create chain
chain = prompt | client | parser

result = chain.invoke({"language":"Bangla", "text":"Hi, how are you?"})

# app definition
app = FastAPI(title = "Langchain server",
              version = "1.0",
              description = "A simple API server using langchain runnable interface" )

## adding chain route

add_routes(
    app,
    chain,
    path = "/chian"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host = "127.0.0.1", port = 8000)

# python3 3-simpleLLMApp/serve.py