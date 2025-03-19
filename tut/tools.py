from crewai_tools import YoutubeChannelSearchTool
from apikey import GROQ_KEY  # Ensure this contains GROQ_API_KEY

yt_tool = YoutubeChannelSearchTool(
    youtube_channel_handle='@StanfordOnline',
    verbose=True,
    config={
        "llm": {
            "provider": "groq",
            "config": {
                "model": "llama3.3-70b-versatile",  # Groq's official model name
                "api_key": GROQ_KEY,
                "temperature": 0.7
            }
        },
        "embedder": {
            "provider": "google",  # Alternative to OpenAI
            "config": {
                "model": "models/embedding-001",
                "task_type": "retrieval_document"
            }
        }
    }
)
