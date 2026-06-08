import sys
import pytubefix
sys.modules['pytube'] = pytubefix  

from crewai import Agent, Task, Crew, LLM
from crewai_tools import YoutubeChannelSearchTool
from dotenv import load_dotenv
load_dotenv()

llm = LLM(model= "openai/gpt-4o-mini")

yt_tool = YoutubeChannelSearchTool(youtube_channel_handle="https://www.youtube.com/@campusx-official")

# researcher agent
blog_researcher = Agent(
    llm= llm,
    role = "Youtube channel video researcher",
    goal= "Find and extract important information about {topic} from youtube video transcript.",
    verbose = True,
    backstory= "You are an expert researcher in AI, machine learning, Deep learning, Generative AI. You analyze videos and extract useful information.",
    tools= [yt_tool],
    allow_delegation = False,
    max_iter=3,         
    max_retry_limit=2 
)
# blog writer agent
blog_writer = Agent(
    llm = llm,
    role = "Technical blog writer",
    goal = "Create an engaging blog article from researched information about {topic}.",
    verbose= True,
    backstory= "You simplify complex AI topics and write clear, educational technical blog.",
    allow_delegation = False,
    max_iter=3,         
    max_retry_limit=2 

)


researcher_task = Task(
    description= "Search the video about {topic} in the youtube channel. Extract the transcript and important details",
    expected_output= "A detailed 3 paragraph research report containing important information from the video.",
    agent= blog_researcher
)

writer_task = Task(
    description= "Using the research report, create a well structured technical report about {topic}.",
    expected_output= "A complete blog article with title, introduction, main explanation and conclusion.",
    agent = blog_writer,
    output_file= "21-crewAI_agent//new_blog.md"
)


crew  = Crew(
    agents= [blog_researcher, blog_writer],
    tasks = [researcher_task, writer_task],
    memory= True,
    cache= True,
    max_rpm= 10
)

result = crew.kickoff(inputs = {"topic":"generative ai vs agentic ai"})

print(result)


# python3 21-crewAI_agent/app.py