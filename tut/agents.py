from crewai import Agent,LLM
from tools import yt_tool
from apikey import GROQ_KEY



llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=GROQ_KEY,
 
    temperature=0.7
)


# creating a blog reasearch agent

blog_researcher= Agent(
    role='Blog creator from youtube videos',
    goal='get the relevant video content for the topic{topic} from yt channel',
    name="Blog Researcher",
    verbose=True,
    memory=True,
    backstory=("Expert in understanding the youtube videos pertaining to computer science  and providing suggestions for the blog content "),
    tools=[yt_tool],
    allow_delegation=True,
    llm=llm
    
)

# creating a blog writer agent with yt tool
blog_writer=Agent(
    role='Writer',
    goal="narrate compelling tech stories about the video {topic} from YT channel",
    verbose=True,
    memory=True,
    backstory=(""" 
               The blog writer is focused on making tech topics engaging and accessible. With a flair for storytelling, the writer simplifies complex ideas, turning them into compelling narratives that both inform and inspire a wide audience. The goal is to blend technical depth with a relatable, exciting narrative style.
               """),
    tools=[yt_tool],
    allow_delegation=False,
    llm=llm
)