from crewai import Crew,Process
from agents import blog_researcher,blog_writer
from tools import yt_tool
from task import research_task,writing_task



crew= Crew(
    process=Process.sequential,
    agents=[blog_researcher,blog_writer],
    tasks=[research_task,writing_task],
    memory=False,
    
    
    cache=True,
    max_rpm=100,
    share_crew=True)



result=crew.kickoff(inputs={'topic':'Kernels'})

print(result)
