from crewai import Task
from tools import yt_tool
from agents import blog_researcher,blog_writer


research_task=Task(
    
    description="identity the video {topic} and get detailed information about the video from the channel",
    expected_output="A comprehensive 3 paragraph summary based on the {topic} from  the video content ",
    tools=[yt_tool],
    agent=blog_researcher
)

writing_task=Task(
    description=("get the info about the video{topic} and write a blog post about the video"),
    expected_output="Summarise the info from the youtube channel video on the {topic} and create the content for the blog",
    tools=[yt_tool],
    agent=blog_writer,
    output_file="blogpost.md",
    async_execution=False
)
